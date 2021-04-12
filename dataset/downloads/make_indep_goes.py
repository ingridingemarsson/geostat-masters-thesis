

import numpy as np
import yaml
import os
import re
import warnings
import datetime
import argparse
import shutil
from pathlib import Path

import xarray as xr
from pyresample import kd_tree, geometry, load_area
from satpy import Scene

from pansat.download.providers.goes_aws import GOESAWSProvider
from pansat.products.satellite.goes import GOES16L1BRadiances


# GLOBAL VARIABLES ################################################################################################################################################################

global channels
channels = list(range(8,17))
channels.remove(12)

global number_of_pixels
number_of_pixels = 128

global storage_path_temp
storage_path_temp = '../temp'
if not Path(storage_path_temp).exists():
	os.mkdir(storage_path_temp)
	
global storage_path_final
storage_path_final = '../origin/independent_goes/'
if not Path(storage_path_final).exists():
	os.mkdir(storage_path_final)
	

area_path='files/areas.yaml'

global area_def
area_def = load_area(area_path, 'full_disk')

area_file = open(area_path)
parsed_area_file = yaml.load(area_file, Loader=yaml.FullLoader)
area_dict_full_disk = parsed_area_file['full_disk']
area_dict_region = parsed_area_file['region']
area_file.close()

global region_corners
region_corners = area_dict_region['area_extent']

global shape_full_disk
shape_full_disk = area_dict_full_disk['shape']

global projection
projection = area_dict_full_disk['projection']


def get_ind_extents_from_center_proj_coord(center_x, center_y, number_of_boxes_x, number_of_boxes_y):
	
	projcoords_x, projcoords_y  = area_def.get_proj_vectors()
	
	projcoords_x_diff = np.abs(projcoords_x - center_x)
	center_x_idx = np.argmin(projcoords_x_diff)
	
	projcoords_y_diff = np.abs(projcoords_y - center_y)
	center_y_idy = np.argmin(projcoords_y_diff)

	box_numbers_x = np.array(range(number_of_boxes_x))-int(number_of_boxes_x/2) #for iteration
	box_numbers_y = np.array(range(number_of_boxes_y))-int(number_of_boxes_y/2) #for iteration

	box_shift_x = 0
	if ((number_of_boxes_x % 2) == 0):
		box_shift_x = np.int(number_of_pixels/2) #correction for region center y index for odd number of boxes
		
	box_shift_y = 0
	if ((number_of_boxes_y % 2) == 0):
		box_shift_y = np.int(number_of_pixels/2) #correction for region center y index for odd number of boxes

	center_x_idx = center_x_idx-box_shift_x
	center_y_idy = center_y_idy-box_shift_y 
	
	box_ind_extents = []
	for box_y in box_numbers_y:
		for box_x in box_numbers_x:
	
			box_ind_extent = [center_x_idx+(box_x-1)*int(number_of_pixels), center_y_idy+box_y*int(number_of_pixels),
						center_x_idx+box_x*int(number_of_pixels), center_y_idy+(box_y-1)*int(number_of_pixels)]
						
			box_ind_extents.append(box_ind_extent)	
	
	return(box_ind_extents)
	

def goes_filename_extract_datetime(mystr):
	'''
	Extracting start and end datetime from GOES combined product filename.
	
	Args:
		mystr: filename for GOES combined product
		
	Returns:
		start: datetime for measurement start
		end: datetime for measurement end
	'''
	
	
	startmatch = re.findall(r'(?:s\d{14})', mystr)[0]
	endmatch = re.findall(r'(?:e\d{14})', mystr)[0]
	
	start = datetime.datetime.strptime(startmatch[1:-1],"%Y%j%H%M%S")
	end = datetime.datetime.strptime(endmatch[1:-1],"%Y%j%H%M%S")
	
	if(start.hour > end.hour):
		end += datetime.timedelta(days=1)

	return([start, end])		
					


def download_cached(time_in, time_out, channels, no_cache=False, time_tol = 480):
	"""
	Download all files from the GOES satellite combined product with
	channels in channels in a given time range but avoid redownloading
	files that are already present.

	Args:
		no_cache: If this is set to True, it forces a re-download of files even 
		if they are already present.

	Returns:
		List of pathlib.Path object pointing to the available data files
		in the request time range.
	"""

	channel = channels[0]
		
	p = GOES16L1BRadiances("F", channel)
	dest = Path(storage_path_temp)
	dest.mkdir(parents=True, exist_ok=True)

	provider = GOESAWSProvider(p)
	filenames0 = provider.get_files_in_range(time_in, time_out, start_inclusive=True)	
	
	datetimes = [goes_filename_extract_datetime(filename) for filename in filenames0]
	
	print(datetimes)
	
	files_list = []
	for datetime in datetimes:
		files = []
		for channel in channels:

			p = GOES16L1BRadiances("F", channel)
			provider = GOESAWSProvider(p)
			filenames = provider.get_files_in_range(datetime[0], datetime[1], start_inclusive=True)
			if(len(filenames)==0):
				files.append(None)
			else:
				f = filenames[0]
				path = dest / f
				
				if not path.exists() or no_cache:
					data = provider.download_file(f, path)
				files.append(path)		
		files_list.append(files)					
	return(files_list)	
		
				

def goes_data_process_independent(filenames_goes, box_ind_extent, filename_new):
	'''
	Read and resample goes data.
	'''
	
	
	files_goes = map(str, filenames_goes)
	goes_scn = Scene(reader='abi_l1b', filenames=files_goes)
	av_dat_names = goes_scn.available_dataset_names()

	# This is a warning regarding loss of projection information when converting to a PROJ string
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		goes_scn.load((av_dat_names))

		
	goes_time_in = [str(goes_scn[av_dat_name].attrs['start_time']) for av_dat_name in av_dat_names]
	goes_time_out = [str(goes_scn[av_dat_name].attrs['end_time']) for av_dat_name in av_dat_names]
	

	# This is a warning regarding loss of projection information when converting to a PROJ string
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')	
		goes_scn = goes_scn.resample(goes_scn.min_area(), resampler = 'native')
		
	height, width = shape_full_disk
	ref_height = goes_scn[av_dat_names[0]].y.shape[0]
	ref_width = goes_scn[av_dat_names[0]].x.shape[0]


	goes_scn = goes_scn.aggregate(x=np.int(ref_width/width), y=np.int(ref_height/height), func='mean')

	keys = av_dat_names
	box_idx_low, box_idy_high, box_idx_high, box_idy_low = box_ind_extent
	
	# RuntimeWarning: invalid value encountered in true_divide x = np.divide(x1, x2, out)
	# Caused by doing mean of nan-values in aggregate
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		values = []
		for av_dat_name in av_dat_names:
			av_dat_vals = goes_scn[av_dat_name].values[box_idy_low:box_idy_high, box_idx_low:box_idx_high]
			values.append((["y", "x"], av_dat_vals)) 

	projcoords_x, projcoords_y  = area_def.get_proj_vectors()
	box_area_extent = [projcoords_x[box_ind_extent[0]], projcoords_y[box_ind_extent[1]], projcoords_x[box_ind_extent[2]], projcoords_y[box_ind_extent[3]]]		
	
	data_vars_dict = dict(zip(keys, values))
	#print(data_vars_dict)
	box_dataset = xr.Dataset(
		data_vars = data_vars_dict, 
			attrs = dict(
				ind_extent = box_ind_extent,
				area_extent = box_area_extent,
				shape = [number_of_pixels, number_of_pixels],
				goes_time_in = goes_time_in,
				goes_time_out = goes_time_out,
				filenames_goes = [str(os.path.basename(filename_goes)) for filename_goes in filenames_goes]))
	box_dataset = box_dataset.astype(np.float32)
	
	#print(box_dataset)
	box_dataset_filename = filename_new
	box_dataset.to_netcdf(box_dataset_filename)
	box_dataset.close()

		
start_time = datetime.datetime(2021,2,16,10,10,15)
end_time = datetime.datetime(2021,2,16,10,20,15)	
	
files_in_range = download_cached(start_time, end_time, channels)

region_center_x = (region_corners[0]+region_corners[2])/2
region_center_y = (region_corners[1]+region_corners[3])/2
inds = get_ind_extents_from_center_proj_coord(region_center_x, region_center_y, 5, 5)

for elem in files_in_range:
	if (None in elem): 
		pass
	st = goes_filename_extract_datetime(str(elem[0]))[0].strftime('%Y%m%d-%H%M%S')
	for ind_ext in inds:
		print(ind_ext)
		if not Path(storage_path_final+'GOES'+st).exists():
			os.mkdir(storage_path_final+'GOES'+st)
	
		goes_data_process_independent(elem, ind_ext, storage_path_final+'GOES'+st+'/'+'GOES'+st+'x'+str(ind_ext[0])+'y'+str(ind_ext[1])+'.nc')
		
		

import numpy as np
import datetime
import re
import warnings
import os

import pyproj
import xarray as xr
from pyresample import kd_tree, geometry, load_area
from satpy import Scene

from pansat.products.satellite.gpm import l2b_gpm_cmb
from pansat.products.satellite.goes import GOES16L1BRadiances
from goes_downloads_with_cache import download_cached
import settings



# Extracts date from resulting link in the earth data search
def gpm_extract_datetime(mystr):
	'''
	Extracting start and end datetime from Earth data search GPM downloading link.
	
	Args:
		mystr: link for GPM product download
		
	Returns:
		start: datetime for measurement start
		end: datetime for measurement end
	'''
	
	datematch = re.findall(r'(?:\.\d{8})', mystr)[0]
	startmatch = re.findall(r'(?:S\d{6})', mystr)[0]
	endmatch = re.findall(r'(?:E\d{6})', mystr)[0]
	
	start = datetime.datetime.strptime(datematch[1:]+startmatch[1:],"%Y%m%d%H%M%S")
	end = datetime.datetime.strptime(datematch[1:]+endmatch[1:],"%Y%m%d%H%M%S")
	
	if(start.hour > end.hour):
		end += datetime.timedelta(days=1)

	return([start, end])
	
	
	
def extract_latlon_limits_from_region():
	'''
	Computing approximate latitude and longitude borders 
	of the choosen projection rectangle.
	
	Returns:
		cS: South border 
		cN: North border
		cW: West border
		cE: East border
	'''

	lower_left_x, lower_left_y, upper_right_x, upper_right_y = settings.region_corners
	
	p = pyproj.Proj(settings.projection)
	a_lon, a_lat = p(lower_left_x, lower_left_y, inverse=True)
	b_lon, b_lat = p(upper_right_x, lower_left_y, inverse=True)
	c_lon, c_lat = p(lower_left_x, upper_right_y, inverse=True)
	d_lon, d_lat = p(upper_right_x, upper_right_y, inverse=True)
	
	cS = np.mean([a_lat, b_lat]) 
	cN = np.mean([c_lat, d_lat]) 
	cW = np.mean([a_lon, c_lon])
	cE = np.mean([b_lon, d_lon]) 
	
	return([cS, cN, cW, cE])

	
	
def create_box_dataset(box_num, gpm_transformed_data, files_gpm):
	'''
	Extracting data from transformed GPM data to create new xarray.Dataset
	for the current box.
	
	Args:
		box_num: Index of box defining coordinates along GPM swath
		gpm_transformed_data: GPM data resampled into GOES projection coordinates,
			surface_precipitation and time
		files_gpm: filenames of downloaded files
	
	Returns: 
		dataset: xarray.Dataset
	
	'''

	box_idx_le, box_idy_do, box_idx_ri, box_idy_up = get_box_ind(box_num, gpm_transformed_data)

	box_data = gpm_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,0]
	box_t_0 = np.nanmin(gpm_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,1]).astype('datetime64[ns]')
	box_t_1 = np.nanmax(gpm_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,1]).astype('datetime64[ns]')

	projvec = settings.area_def.get_proj_vectors()
	projcoords_x, projcoords_y = projvec
	box_ind_extent = [box_idx_le, box_idy_up, box_idx_ri, box_idy_do]
	box_area_extent = [projcoords_x[box_idx_le], projcoords_y[box_idy_up], projcoords_x[box_idx_ri], projcoords_y[box_idy_do]]

	dataset = xr.Dataset(
			#coords = dict(
			#	y = (["y"], projcoords_y[box_idy_do:box_idy_up]), 
			#	x = (["x"], projcoords_x[box_idx_le:box_idx_ri])),
			data_vars = dict(gpm_precipitation = (["y","x"], box_data)), 
			attrs = dict(
				ind_extent = box_ind_extent,
				area_extent = box_area_extent,
				shape = [settings.number_of_pixels, settings.number_of_pixels],
				gpm_time_in = str(box_t_0), 
				gpm_time_out = str(box_t_1),
				filename_gpm = str(files_gpm[0])
					))
					
	return(dataset)
		
		
		
def gpm_data_processing(gpm_file_time):
	'''
	Download of GPM product in specific time interval, transformation into 
	GOES projection coordinates and cropped into several smaller 'box' datasets.
	
	Args:
		gpm_file_time: [start, end] time for measurements in GPM file
		
	Returns:
		datasets: list containing multiple xarray.Dataset
	'''

	# DOWNLOAD gpm data
	files_gpm = l2b_gpm_cmb.download(gpm_file_time[0], gpm_file_time[1])
	dataset_gpm = l2b_gpm_cmb.open(files_gpm[0])
	
	# Resample coordinates for GPM data into GOES data format
	repeted_time = np.transpose(np.tile(dataset_gpm['scan_time'].astype(np.int),(49,1)))
	gpm_data = np.stack((dataset_gpm['surface_precipitation'].values, repeted_time), axis = 2)
	swath_def = geometry.SwathDefinition(lons=dataset_gpm['longitude'], lats=dataset_gpm['latitude'])
	gpm_transformed_data = kd_tree.resample_nearest(swath_def, gpm_data, 
	settings.area_def, radius_of_influence=3700, epsilon=0, fill_value=np.nan) #3600 --> missing pixels

	
	region_center_y = np.mean([settings.region_corners[1],settings.region_corners[3]])
	projvec = settings.area_def.get_proj_vectors() #tuple (X,Y)
	projcoords_y = projvec[1]
	projcoords_x = projvec[0]
	projcoords_y_diff = np.abs(projcoords_y - region_center_y)
	region_center_y_idy = np.argmin(projcoords_y_diff)
	
	box_nums = get_box_num()
	
	datasets = []
	for box_num in box_nums:
		dataset_box = create_box_dataset(box_num, gpm_transformed_data, files_gpm)
		datasets.append(dataset_box)
		
	for files in files_gpm:	
		if os.path.exists(files):
  			os.remove(files)
		else:
  			pass
	
	
	return(datasets)
	
	
	
def get_box_ind(box_num, gpm_transformed_data):
	'''
	TODO
	'''
	
	region_center_y = np.mean([settings.region_corners[1],settings.region_corners[3]])
	projvec = settings.area_def.get_proj_vectors() #tuple (X,Y)
	projcoords_y = projvec[1]
	projcoords_x = projvec[0]
	projcoords_y_diff = np.abs(projcoords_y - region_center_y)
	region_center_y_idy = np.argmin(projcoords_y_diff)
	
	box_idy_up = region_center_y_idy+int((box_num+1)*settings.number_of_pixels)
	box_idy_do = region_center_y_idy+int(box_num*settings.number_of_pixels)

	swath_idx_up = np.where(np.isnan(gpm_transformed_data[box_idy_up,:,0]) == False)[0]
	swath_idx_do = np.where(np.isnan(gpm_transformed_data[box_idy_do,:,0]) == False)[0]

	box_middle_x = int(np.mean([min(min(swath_idx_up), min(swath_idx_do)), max(max(swath_idx_up), max(swath_idx_do))]))
	box_idx_le = box_middle_x-int(0.5*settings.number_of_pixels)
	box_idx_ri = box_middle_x+int(0.5*settings.number_of_pixels)
	
	return([box_idx_le, box_idy_do, box_idx_ri, box_idy_up])
	
	
	
def get_box_num():
	'''
	TODO
	'''
	
	projvec = settings.area_def.get_proj_vectors() #tuple (X,Y)
	projcoords_y = projvec[1]
	h0 = np.argmin(np.abs(projcoords_y-settings.region_corners[1]))
	h1 = np.argmin(np.abs(projcoords_y-settings.region_corners[3])) 
	box_height = np.abs(h1-h0)
	n = box_height/(2*settings.number_of_pixels)
	box_num_ind = 2*np.int(np.round(n))
	box_nums = np.array(range(box_num_ind))-int(box_num_ind/2)
	
	return(box_nums)
	
	
def goes_data_processing(time, ind_extent):
	'''
	TODO
	'''

	box_idx_le, box_idy_up, box_idx_ri, box_idy_do = ind_extent
	height, width = settings.shape_full_disk
	filenames_goes = download_cached(time[0], time[1], settings.channels)
	files_goes = map(str, filenames_goes)
	goes_scn = Scene(reader='abi_l1b', filenames=files_goes)
	av_dat_names = goes_scn.available_dataset_names()

	# This is a warning regarding loss of projection information when converting to a PROJ string
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		goes_scn.load((av_dat_names))


	# This is a warning regarding loss of projection information when converting to a PROJ string
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')	
		goes_scn = goes_scn.resample(goes_scn.min_area(), resampler = 'native')
		
	ref_height = goes_scn[av_dat_names[0]].y.shape[0]
	ref_width = goes_scn[av_dat_names[0]].x.shape[0]

	goes_scn = goes_scn.aggregate(x=np.int(ref_width/width), 
	y=np.int(ref_height/height), func='mean')

	# RuntimeWarning: invalid value encountered in true_divide x = np.divide(x1, x2, out)
	# Caused by doing mean of nan-values in aggregate
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')

		keys = av_dat_names
		values = [(["y", "x"], goes_scn[av_dat_name].values[box_idy_do:box_idy_up, box_idx_le:box_idx_ri]) for av_dat_name in av_dat_names]
		data_vars_dict = dict(zip(keys, values))
		
		projvec = settings.area_def.get_proj_vectors()
		projcoords_x, projcoords_y = projvec


		dataset = xr.Dataset(
					#coords = dict(
					#	y = (["y"], projcoords_y[box_idy_do:box_idy_up]), 
					#	x = (["x"], projcoords_x[box_idx_le:box_idx_ri])),
					data_vars = data_vars_dict, 
					attrs = dict(
						goes_time_in = str(goes_scn[av_dat_names[0]].attrs['start_time']), 
						goes_time_out = str(goes_scn[av_dat_names[0]].attrs['end_time']),
						filenames_goes = [str(filename_goes) for filename_goes in filenames_goes]
							))

	return(dataset)
	
	

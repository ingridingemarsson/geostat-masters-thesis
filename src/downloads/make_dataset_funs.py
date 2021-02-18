import numpy as np
import datetime
import re
import warnings
from pathlib import Path
import os

from pyresample import kd_tree, geometry, load_area
from satpy import Scene

import settings



def gpm_link_extract_datetime(mystr):
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
	
	
	
def label_data_transform(label_dataset):
	'''
	TODO
	'''
	
	rep_num = label_dataset['matched_pixels'].shape[0]
	times = label_dataset['scan_time'].values.astype(np.int)
	repeted_times = np.transpose(np.tile(times,(rep_num,1)))

	precip = label_dataset['surface_precipitation'].values
	label_data = np.stack((precip, repeted_times), axis = 2)

	swath_def = geometry.SwathDefinition(lons=label_dataset['longitude'], lats=label_dataset['latitude'])
	label_transformed_data = kd_tree.resample_nearest(swath_def, label_data, settings.area_def,
		                                        radius_of_influence=3700, epsilon=0, 
		                                        fill_value=np.nan)
		                                        
	return(label_transformed_data)
	
	
	
def label_data_crop(box_idy_do_center, box_number, label_transformed_data):
	'''
	TODO
	'''

	box_idy_do = box_idy_do_center + box_number*settings.number_of_pixels
	box_idy_up = box_idy_do + settings.number_of_pixels 

	swath_idx_up = np.where(np.isnan(label_transformed_data[box_idy_up,:,0]) == False)[0]
	swath_idx_do = np.where(np.isnan(label_transformed_data[box_idy_do,:,0]) == False)[0]

	box_middle_x = int(np.mean([min(min(swath_idx_up), min(swath_idx_do)), max(max(swath_idx_up), max(swath_idx_do))]))
	box_idx_le = box_middle_x-int(0.5*settings.number_of_pixels)
	box_idx_ri = box_middle_x+int(0.5*settings.number_of_pixels)
	
	box_data = label_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,0]
	label_time_in = np.nanmin(label_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,1]).astype('datetime64[ns]')
	label_time_in = datetime.datetime.strptime(str(label_time_in)[:-3],"%Y-%m-%dT%H:%M:%S.%f")
	label_time_out = np.nanmax(label_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,1]).astype('datetime64[ns]')
	label_time_out = datetime.datetime.strptime(str(label_time_out)[:-3],"%Y-%m-%dT%H:%M:%S.%f")

	box_ind_extent = [box_idx_le, box_idy_up, box_idx_ri, box_idy_do]
	return([box_data, box_ind_extent, label_time_in, label_time_out])

	
	
def calculate_boxes(projcoords_y):
	'''
	TODO
	'''
       
	region_center_y = np.mean([settings.region_corners[1], settings.region_corners[3]])
	projcoords_y_diff = np.abs(projcoords_y - region_center_y)
	region_center_y_idy = np.argmin(projcoords_y_diff)

	region_min_y = np.argmin(np.abs(projcoords_y-settings.region_corners[1]))
	region_max_y = np.argmin(np.abs(projcoords_y-settings.region_corners[3])) 
	box_height = np.abs(region_max_y-region_min_y)

	number_of_boxes = np.int(np.round(box_height/settings.number_of_pixels-1))
	
	offset_range = box_height-number_of_boxes*settings.number_of_pixels
	offset = np.random.randint(offset_range)-np.int(offset_range/2)

	region_center_y_idy += offset
	
	box_numbers = np.array(range(number_of_boxes))-int(number_of_boxes/2)

	box_shift = 0
	if ((number_of_boxes % 2) != 0):
		box_shift = np.int(settings.number_of_pixels/2)

	box_idy_do_center = region_center_y_idy-box_shift
	
	return([box_idy_do_center, box_numbers])
	
	
	
def input_data_process(box_ind_extent, filenames_goes):
	'''
	TODO
	'''
	
	files_goes = map(str, filenames_goes)
	goes_scn = Scene(reader='abi_l1b', filenames=files_goes)
	av_dat_names = goes_scn.available_dataset_names()

	# This is a warning regarding loss of projection information when converting to a PROJ string
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		goes_scn.load((av_dat_names))
		
	input_time_in = str(goes_scn[av_dat_names[0]].attrs['start_time'])
	input_time_out = str(goes_scn[av_dat_names[0]].attrs['end_time'])


	# This is a warning regarding loss of projection information when converting to a PROJ string
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')	
		goes_scn = goes_scn.resample(goes_scn.min_area(), resampler = 'native')
		
	height, width = settings.shape_full_disk
	ref_height = goes_scn[av_dat_names[0]].y.shape[0]
	ref_width = goes_scn[av_dat_names[0]].x.shape[0]

	
	goes_scn = goes_scn.aggregate(x=np.int(ref_width/width), y=np.int(ref_height/height), func='mean')

	keys = av_dat_names
	box_idx_le, box_idy_up, box_idx_ri, box_idy_do = box_ind_extent
	
	# RuntimeWarning: invalid value encountered in true_divide x = np.divide(x1, x2, out)
	# Caused by doing mean of nan-values in aggregate
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		values = [(["y", "x"], goes_scn[av_dat_name].values[box_idy_do:box_idy_up, box_idx_le:box_idx_ri]) for av_dat_name in av_dat_names]
		
		
	return([keys, values, input_time_in, input_time_out])
	
	
	
def get_dataset_filename(box_number, label_file_start, filetype):
	'''
	TODO
	'''

	parent_dir = settings.path_to_store_processed_data + '/' + settings.linkfile.replace(".txt", "")
	
	if not Path(settings.path_to_store_processed_data).exists():
		os.mkdir(settings.path_to_store_processed_data)
		
	if not Path(parent_dir).exists():
		os.mkdir(parent_dir)
	
	common_filename = '/GPMGOES-' 
	common_filename += 'oS' + str(label_file_start).replace(" ", "T") 
	common_filename += '-c' + str(settings.channels).replace(" ", "") 
	common_filename += '-p' + str(settings.number_of_pixels) 


	total_path = parent_dir + common_filename + '-b' + str(box_number) + filetype
	return(total_path)









	
	
	

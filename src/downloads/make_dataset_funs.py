import numpy as np
import datetime
import re
import warnings
from pathlib import Path
import os

from pyresample import kd_tree, geometry, load_area
from satpy import Scene

import downloads.settings as st



def gpm_link_extract_datetime(mystr):
	'''
	Extracting start and end datetime from Earth data search 2BCMB downloading link.
	
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
	label_transformed_data = kd_tree.resample_nearest(swath_def, label_data, st.area_def,
		                                        radius_of_influence=3700, epsilon=0, 
		                                        fill_value=np.nan)
		                                        
	return(label_transformed_data)
	
	
		
def calculate_boxes(projcoords_y):
	'''
	TODO
	'''
       
	region_center_y = np.mean([st.region_corners[1], st.region_corners[3]]) # lower left y coordinate (SW), upper right y coordinate (NE)
	projcoords_y_diff = np.abs(projcoords_y - region_center_y)
	region_center_y_idy = np.argmin(projcoords_y_diff)

	region_min_y = np.argmin(np.abs(projcoords_y-st.region_corners[1])) #lowest y-index in region (Northmost)
	region_max_y = np.argmin(np.abs(projcoords_y-st.region_corners[3])) #highest y-index in region (Southmost)
	region_height = np.abs(region_max_y-region_min_y) #y extent of region in pixels

	number_of_boxes = np.int(np.round(region_height/st.number_of_pixels-1))
	
	offset_range = region_height-number_of_boxes*st.number_of_pixels #how much extra y extent is there to play with
	offset = np.random.randint(offset_range)-np.int(offset_range/2) #how much to shift the region center y index
	region_center_y_idy += offset
	
	box_numbers = np.array(range(number_of_boxes))-int(number_of_boxes/2) #for iteration

	box_shift = 0
	if ((number_of_boxes % 2) != 0):
		box_shift = np.int(st.number_of_pixels/2) #correction for region center y index for odd number of boxes

	box_idy_low_center = region_center_y_idy-box_shift #The lowest y-index border of the 'middle' box (Northmost border)
	
	return([box_idy_low_center, box_numbers])
	
	
	
def label_data_crop(box_idy_low_center, box_number, label_transformed_data):
	'''
	TODO
	'''

	box_idy_low = box_idy_low_center + box_number*st.number_of_pixels #The lowest y-index border of the current box (Northmost border)
	box_idy_high = box_idy_low + st.number_of_pixels #The highest y-index border of the current box (Southmost border)

	#Check which x-indices is in the swath at the North and South box border
	swath_idx_low = np.where(np.isnan(label_transformed_data[box_idy_low,:,0]) == False)[0]
	swath_idx_high = np.where(np.isnan(label_transformed_data[box_idy_high,:,0]) == False)[0]

	box_idx_low_swath = min(min(swath_idx_high), min(swath_idx_low)) #The lowest x-index in the swath (Westmost)
	box_idx_high_swath = max(max(swath_idx_high), max(swath_idx_low)) #The highest x-index in the swath (Eastmost)
	box_middle_x = int(np.mean([box_idx_low_swath, box_idx_high_swath])) # Center of swath in x direction
	box_idx_low = box_middle_x-int(0.5*st.number_of_pixels) #The lowest x-index of the current box (Westmost border)
	box_idx_high = box_middle_x+int(0.5*st.number_of_pixels) #The highest x-index of the current box (Eastmost border)
	
	box_data = label_transformed_data[box_idy_low:box_idy_high, box_idx_low:box_idx_high,0] #Cropping the whole data to the current box
	label_time_in = np.nanmin(label_transformed_data[box_idy_low:box_idy_high, box_idx_low:box_idx_high,1]).astype('datetime64[ns]')
	label_time_in = datetime.datetime.strptime(str(label_time_in)[:-3],"%Y-%m-%dT%H:%M:%S.%f")
	label_time_out = np.nanmax(label_transformed_data[box_idy_low:box_idy_high, box_idx_low:box_idx_high,1]).astype('datetime64[ns]')
	label_time_out = datetime.datetime.strptime(str(label_time_out)[:-3],"%Y-%m-%dT%H:%M:%S.%f")

	box_ind_extent = [box_idx_low, box_idy_high, box_idx_high, box_idy_low] # In form of area extent: lower left x, lower left y, upper right x, upper right y --> West, South, East, North
	
	return([box_data, box_ind_extent, label_time_in, label_time_out])

	

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
		
	height, width = st.shape_full_disk
	ref_height = goes_scn[av_dat_names[0]].y.shape[0]
	ref_width = goes_scn[av_dat_names[0]].x.shape[0]

	
	goes_scn = goes_scn.aggregate(x=np.int(ref_width/width), y=np.int(ref_height/height), func='mean')

	keys = av_dat_names
	box_idx_low, box_idy_high, box_idx_high, box_idy_low = box_ind_extent
	
	# RuntimeWarning: invalid value encountered in true_divide x = np.divide(x1, x2, out)
	# Caused by doing mean of nan-values in aggregate
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		values = [(["y", "x"], goes_scn[av_dat_name].values[box_idy_low:box_idy_high, box_idx_low:box_idx_high]) for av_dat_name in av_dat_names]
		
		
	return([keys, values, input_time_in, input_time_out])
	
	
	
def get_dataset_filename(box_number, label_file_start, filetype):
	'''
	TODO
	'''

	parent_dir = st.path_to_store_processed_data + '/' + st.linkfile.replace(".txt", "")
	
	if not Path(st.path_to_store_processed_data).exists():
		os.mkdir(st.path_to_store_processed_data)
		
	if not Path(parent_dir).exists():
		os.mkdir(parent_dir)
	
	common_filename = '/GPMGOES-' 
	common_filename += 'oS' + str(label_file_start).replace(" ", "T") 
	common_filename += '-c' + str(st.channels).replace(" ", "") 
	common_filename += '-p' + str(st.number_of_pixels) 

	common_dir = parent_dir + common_filename

	if not Path(common_dir).exists():
		os.mkdir(common_dir)
	
	total_path = common_dir + common_filename + '-b' + str(box_number) + filetype
	return(total_path)









	
	
	

import numpy as np
import warnings
import shutil
import time
import os
from pathlib import Path

import xarray as xr

from pansat.products.satellite.gpm import l2b_gpm_cmb
from make_dataset_funs import gpm_link_extract_datetime, label_data_transform, label_data_crop, calculate_boxes, input_data_process, get_dataset_filename
from goes_downloads_with_cache import download_cached
from gpm_plots import region_plot
import settings

# Clock the program
start_timing = time.time()

settings.parse_arguments()
settings.initial_load()

# Get list of Earth data search gpm product links from file
link_file = open('links/linkfiles/' + settings.linkfile, "r") 
link_list = link_file.readlines()
link_file.close()

N = len(link_list)
if (settings.test == True):
	N = 2

# For each gpm product link
for j in range(0,N):	
	
	print('start # ' + str(j))
			
	label_file_link = link_list[j]

	label_file_start, label_file_end = gpm_link_extract_datetime(label_file_link) #FUN

	label_files = l2b_gpm_cmb.download(label_file_start, label_file_end)
	label_dataset = l2b_gpm_cmb.open(label_files[0])

	label_transformed_data = label_data_transform(label_dataset)  #FUN
	projcoords_x, projcoords_y = settings.area_def.get_proj_vectors() 
	box_idy_low_center, box_numbers = calculate_boxes(projcoords_y)  #FUN

	box_datasets = []
	for box_number in box_numbers:

		label_box_data, box_ind_extent, label_time_in, label_time_out = label_data_crop(box_idy_low_center, box_number, label_transformed_data)  #FUN
		box_area_extent = [projcoords_x[box_ind_extent[0]], projcoords_y[box_ind_extent[1]], projcoords_x[box_ind_extent[2]], projcoords_y[box_ind_extent[3]]]
		if not (box_area_extent[0] < settings.region_corners[0] or box_area_extent[2] > settings.region_corners[2]):
		
			filenames_goes = download_cached(label_time_in, label_time_out, settings.channels)
			keys, values, input_time_in, input_time_out = input_data_process(box_ind_extent, filenames_goes)
			keys.append('gpm_precipitation')
			values.append((["y","x"], label_box_data))
			data_vars_dict = dict(zip(keys, values))

			box_dataset = xr.Dataset(
						data_vars = data_vars_dict, 
						attrs = dict(
								ind_extent = box_ind_extent,
								area_extent = box_area_extent,
								shape = [settings.number_of_pixels, settings.number_of_pixels],
								gpm_time_in = str(label_time_in), 
								gpm_time_out = str(label_time_out),
								goes_time_in = str(input_time_in),
								goes_time_out = str(input_time_out),
								filename_gpm = str(label_files[0]),
								filenames_goes = [str(filename_goes) for filename_goes in filenames_goes]))
			box_dataset = box_dataset.astype(np.float32)
			box_dataset_filename = get_dataset_filename(box_number, label_file_start,'.nc')
			box_dataset.to_netcdf(box_dataset_filename)
			box_dataset.close()

			box_datasets.append(box_dataset)



	if (settings.make_box_plot == True and len(box_datasets) > 0):
		# This is a warning regarding loss of projection information when converting to a PROJ string
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			for key in keys:
				region_plot(box_datasets, key, get_dataset_filename(box_number, label_file_start, str(key)+'.png'))
		
	if (settings.used_remove == True):
		for files in label_files:	
			if os.path.exists(files):
				os.remove(files)
			else:
				pass
			
	print('end # ' + str(j))
			
if (settings.used_remove == True):	
	dir_path = Path(settings.path_to_store_goes_data) / Path(settings.linkfile.replace(".txt", ""))
	try:
	    shutil.rmtree(dir_path)
	except OSError as e:
	    print("Error: %s : %s" % (dir_path, e.strerror))
	    

end_timing = time.time()
total_timing = end_timing-start_timing
print('total time: ' + str(total_timing)+ ' seconds')		




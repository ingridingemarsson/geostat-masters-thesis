# TODO
import numpy as np
import datetime
import os
import shutil
from pathlib import Path

import xarray as xr

import downloads.settings as st
from downloads.make_dataset_funs import input_data_process, get_dataset_filename
from downloads.goes_downloads_with_cache import download_cached
from visualize.plot_dataset_funs import pars_dataset_filename


'''

Takes a linkfile folder in Datasets, opens each box-file and adds specified channels. Saves to new 
file in Datasets_appended.

Reuses parsing in settings, can run with -c, -lf, -s

'''


st.parse_arguments()
st.initial_load('downloads/areas.yaml')

parentdir = st.path_to_store_processed_data + '/' + st.linkfile.replace(".txt", "") + '/'

for subdir in [f.name for f in os.scandir(parentdir) if f.is_dir()]:
	for filename in os.listdir(Path(parentdir+subdir)):
		if filename.endswith('.nc'):
			label_file_start, channels_old, __, box_number = pars_dataset_filename(filename)

			dataset = xr.open_dataset(parentdir+subdir+'/'+filename)
			dataset.close()

			new_channels = []
			for channel in st.channels:
				if not ('C'+str(channel).zfill(2) in list(dataset.data_vars.keys())):
					new_channels.append(channel)

			if (len(new_channels) > 0):

				time_in = datetime.datetime.strptime(dataset.goes_time_in[:-7],"%Y-%m-%d %H:%M:%S") 
				time_out = datetime.datetime.strptime(dataset.goes_time_out[:-7],"%Y-%m-%d %H:%M:%S") 

				filenames_goes = download_cached(time_in, time_out, new_channels)

				new_keys, new_values, __, __ = input_data_process(dataset.ind_extent, filenames_goes)
				
				for i in range(len(new_channels)):
					dataset[new_keys[i]] = new_values[i]

				dataset = dataset.astype(np.float32)
				dataset_filename = get_dataset_filename(box_number, label_file_start,'.nc', channels_old=channels_old, ext = '_extended')
				dataset.to_netcdf(dataset_filename)
				dataset.close()
			
				
if (st.used_remove == True):	
	dir_path = Path(st.path_to_store_goes_data) / Path(st.linkfile.replace(".txt", ""))
	if os.path.exists(dir_path):
		try:
		    shutil.rmtree(dir_path)
		except OSError as e:
		    print("Error: %s : %s" % (dir_path, e.strerror))
					
				
				
				
				
				
				
				
				
				
				
				


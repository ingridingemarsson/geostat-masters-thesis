import numpy as np
import datetime
import warnings
import os 
import yaml
import time
import shutil
from pathlib import Path

import xarray as xr
from pyresample import kd_tree, geometry, load_area
from satpy import Scene

from pansat.products.satellite.gpm import l2b_gpm_cmb
from pansat.products.satellite.goes import GOES16L1BRadiances
from goes_downloads_with_cache import download_cached
from funs_gpm_downloads import (gpm_extract_datetime, extract_latlon_limits_from_region, 
	create_box_dataset, gpm_data_processing, get_box_ind, get_box_num, goes_data_processing)
from gpm_plots import region_plot2
import settings
from settings import parse_arguments, initial_load


def create_GPMGOES_dataset(gpm_start_time, gpm_end_time):
	'''
	TODO
	'''

	latlon_limits = extract_latlon_limits_from_region()
	datasets_gpm = gpm_data_processing(gpm_file_time)
	
	# TODO: rewrite this	
	parent_dir = settings.path_to_store_processed_data + '/' + settings.linkfile.replace(".txt", "")
	common_dir = parent_dir + '/GPMGOES-' + 'oS' + str(gpm_file_time[0]).replace(" ", "T") + '-c' + str(settings.channels).replace(" ", "") + '-p' + str(settings.number_of_pixels) 
	
	if not Path(settings.path_to_store_processed_data).exists():
		os.mkdir(settings.path_to_store_processed_data)
		
	if not Path(parent_dir).exists():
		os.mkdir(parent_dir)
	
	if not Path(common_dir).exists():
		os.mkdir(common_dir) 

	# TODO: rewrite this	
	common_filename = '/GPMGOES-' + 'oS' + str(gpm_file_time[0]).replace(" ", "T") + '-c' + str(settings.channels).replace(" ", "") + '-p' + str(settings.number_of_pixels) 
	filetype = '.nc'

	for i in range(len(datasets_gpm)):
		time_in = datetime.datetime.strptime(datasets_gpm[i].attrs['gpm_time_in'][:-3],"%Y-%m-%dT%H:%M:%S.%f")
		time_out = datetime.datetime.strptime(datasets_gpm[i].attrs['gpm_time_out'][:-3],"%Y-%m-%dT%H:%M:%S.%f")
		ind_extent = datasets_gpm[i].attrs['ind_extent']		
		
		dataset_goes = goes_data_processing([time_in, time_out], ind_extent)
		
		dataset = xr.merge([datasets_gpm[i], dataset_goes], combine_attrs = "no_conflicts")
		dataset = dataset.where(np.isnan(dataset['gpm_precipitation'].values) == False) 
		
		dataset_filename = common_dir + common_filename + '-nS' + datasets_gpm[i].attrs['gpm_time_in']+ '-b' + str(i) + filetype
		dataset.to_netcdf(dataset_filename)
		dataset.close()

		
	## This is a warning regarding loss of projection information when converting to a PROJ string
	#with warnings.catch_warnings():
	#	warnings.simplefilter('ignore')
	#	region_plot2(datasets_gpm,'gpm_precipitation', common_dir + common_filename+'image'+'.pdf')
		
		
start_timing = time.time()
	
parse_arguments()

link_file = open('linkfiles/' + settings.linkfile, "r") 
link_list = link_file.readlines()
link_file.close()
		
initial_load()





N = len(link_list)
for j in range(0,N):
	gpm_file_time = gpm_extract_datetime(link_list[j])
	create_GPMGOES_dataset(gpm_file_time[0], gpm_file_time[1])
	
dir_path = Path(settings.path_to_store_goes_data) / Path(settings.linkfile.replace(".txt", ""))
try:
    shutil.rmtree(dir_path)
except OSError as e:
    print("Error: %s : %s" % (dir_path, e.strerror))

	
end_timing = time.time()
total_timing = end_timing-start_timing
print('total time: ' + str(total_timing)+ ' seconds')














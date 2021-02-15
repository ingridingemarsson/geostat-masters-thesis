from funs_gpm_downloads import *
from pansat.products.satellite.gpm import l2b_gpm_cmb
from pansat.products.satellite.goes import GOES16L1BRadiances
from goes_downloads_with_cache import *

from pyresample import kd_tree, geometry
from satpy import Scene
from pathlib import Path
import xarray as xr
from pyresample import load_area
import yaml
import numpy as np
import datetime
import time
import warnings
import os 

############################### CONSTANTS ###############################

start_timing = time.time()

# GOES channels to download
channels = [8,13] #[8, 13]
number_of_pixels = 256 #512

# Paths
linkfile = "link_file_test.txt" #"2017-12-18--2018-12-17-with-shapefile.txt" 
path_to_store_processed_data = "Dataset"


##################### Download, transform, crop, save #####################

link_file = open(linkfile, "r") #File with all linkes from the Earth data search
link_list = link_file.readlines()
link_file.close()



def create_GPMGOES_dataset(gpm_start_time, gpm_end_time, channels, number_of_pixels, path_to_store_processed_data):

	# GOES data projection 
	area_def = load_area('areas.yaml', 'full_disk')
	new_area_def = load_area('areas.yaml', 'region')
	area_file = open('areas.yaml')
	parsed_area_file = yaml.load(area_file, Loader=yaml.FullLoader)
	area_dict_full_disk = parsed_area_file['full_disk']
	area_dict_region = parsed_area_file['region']
	area_file.close()

	area_extent = area_dict_full_disk['area_extent']
	projection = area_dict_full_disk['projection']
	height = area_dict_full_disk['height']
	width = area_dict_full_disk['width']
	region_corners = area_dict_region['area_extent']

	latlon_limits = extract_latlon_limits_from_region(projection, region_corners)


	datasets_gpm = gpm_data_processing(number_of_pixels, gpm_file_time, area_def, region_corners)
	common_dir = path_to_store_processed_data + '/GPMGOES-' + 'oS' + str(gpm_file_time[0]).replace(" ", "T") + '-c' + str(channels).replace(" ", "") + '-p' + str(number_of_pixels) 
	
	if not Path(common_dir).exists():
		os.mkdir(common_dir) 

	common_filename = '/GPMGOES-' + 'oS' + str(gpm_file_time[0]).replace(" ", "T") + '-c' + str(channels).replace(" ", "") + '-p' + str(number_of_pixels) 
	filetype = '.nc'

	for i in range(len(datasets_gpm)):
		dataset_filename = common_dir + common_filename + '-nS' + datasets_gpm[i].attrs['gpm_time_in']+ '-b' + str(i) + filetype
		time_in = datetime.datetime.strptime(datasets_gpm[i].attrs['gpm_time_in'][:-3],"%Y-%m-%dT%H:%M:%S.%f")
		time_out = datetime.datetime.strptime(datasets_gpm[i].attrs['gpm_time_out'][:-3],"%Y-%m-%dT%H:%M:%S.%f")
		ind_extent = datasets_gpm[i].attrs['ind_extent']		
		dataset_goes = goes_data_processing([time_in, time_out], channels, ind_extent, area_def, [height, width])
		dataset = xr.merge([datasets_gpm[i], dataset_goes], combine_attrs = "no_conflicts")
		dataset = dataset.where(np.isnan(dataset['gpm_precipitation'].values) == False) 
		dataset.to_netcdf(dataset_filename)
		dataset.close()

		

	# This is a warning regarding loss of projection information when converting to a PROJ string
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		region_plot2(area_def, datasets_gpm,'gpm_precipitation', number_of_pixels, common_dir + common_filename+'image'+'.pdf')
		


N = 6 # len(link_list)
for j in range(0,N,2):
	gpm_file_time = gpm_extract_datetime(link_list[j])
	create_GPMGOES_dataset(gpm_file_time[0], gpm_file_time[1], channels, number_of_pixels, path_to_store_processed_data)
end_timing = time.time()
total_timing = end_timing-start_timing
print('total time: ' + str(total_timing)+ ' seconds')


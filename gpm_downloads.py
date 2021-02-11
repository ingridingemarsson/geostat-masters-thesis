from funs_gpm_downloads import *
from pansat.products.satellite.gpm import l2b_gpm_cmb
from pansat.products.satellite.goes import GOES16L1BRadiances

from pyresample import kd_tree, geometry
from satpy import Scene
from pathlib import Path
import xarray as xr
from pyresample import load_area
import yaml
import numpy as np
from datetime import datetime
import time
import warnings

############################### CONSTANTS ###############################

start_timing = time.time()

# GOES channels to download
channels = [8,13] #[8, 13]

# Paths
linkfile = "link_file_test.txt" #"2017-12-18--2018-12-17-with-shapefile.txt" 
path_to_download_storage_gpm = "GPM" #Path to where to store the gpm files
path_to_download_storage_goes = "GOES-16" #Path to where to store the goes files
path_to_store_processed_data = "Dataset"

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

##################### Download, transform, crop, save #####################

link_file = open(linkfile, "r") #File with all linkes from the Earth data search
link_list = link_file.readlines()
link_file.close()


for i in range(0,2,8): 
	
	# ----------------------------- FOR EACH LINK -----------------------------
	
	print(i)
	gpm_file_t_0, gpm_file_t_1 = gpm_extract_datetime(link_list[i])
	print('GPM time: ' + str(gpm_file_t_0) + ', ' + str(gpm_file_t_1))


	# DOWNLOAD gpm data
	files_gpm = l2b_gpm_cmb.download(gpm_file_t_0, gpm_file_t_1, path_to_download_storage_gpm)
	dataset_gpm = l2b_gpm_cmb.open(files_gpm[0])

	# TEST
	#dataset_gpm = l2b_gpm_cmb.open("GPM/2B.GPM.DPRGMI.2HCSHv4-1.20180314-S194719-E211954.022965.V06A.HDF5") #TEST

	# Arrival and departure times
	in_brazil_t_0, in_brazil_t_1 = extract_time_in_brazil(latlon_limits,
							 	dataset_gpm)
							 	
	print('In Brazil: ' + str(in_brazil_t_0) + ', ' + str(in_brazil_t_1))
	
	
	# DOWNLOAD goes data
	files_goes = []
	files_tmp = GOES16L1BRadiances("F", channels[0]).download(in_brazil_t_0, in_brazil_t_1, path_to_download_storage_goes)
	goes_index = 0
		
	if (len(files_tmp) > 1):
		time_list_goes = [GOES16L1BRadiances("F",
		channels[0]).filename_to_date(file_tmp) for file_tmp in files_tmp]
		mean_time_brazil = in_brazil_t_0 + datetime.timedelta(
		seconds = (in_brazil_t_1 - in_brazil_t_0).seconds/2)
		timediff = [np.abs((time_goes-mean_time_brazil).seconds) 
		for time_goes in time_list_goes]				
		goes_index = np.argmin(timediff)
	elif (len(files_tmp) == 0):
		continue    
	
	files_goes.append(files_tmp[goes_index])
		
		
	for channel in channels[1:]:
		files_tmp = GOES16L1BRadiances("F", channel).download(in_brazil_t_0, 
		in_brazil_t_1, path_to_download_storage_goes)
		files_goes.append(files_tmp[goes_index])
		
	print('goes files downloaded')	
	
	# TEST
	#files_08 = [Path('GOES-16/OR_ABI-L1b-RadF-M3C08_G16_s20180732000427_e20180732011193_c20180732011242.nc')] #TEST
	#files_13 = [Path('GOES-16/OR_ABI-L1b-RadF-M3C13_G16_s20180732000427_e20180732011205_c20180732011259.nc')] #TEST
	#files_goes = [files_08[0], files_13[0]] #TEST

	# Resample coordinates for GPM data into GOES data format
	swath_def = geometry.SwathDefinition(lons=dataset_gpm['longitude'], lats=dataset_gpm['latitude'])
	gpm_transformed_data = kd_tree.resample_nearest(swath_def, dataset_gpm['surface_precipitation'].values, 
	area_def, radius_of_influence=50000, epsilon=0.5, fill_value=None)
	print('gpm transformed')
	

	files_goes = map(str, files_goes)
	goes_scn = Scene(reader='abi_l1b', filenames=files_goes)
	av_dat_names = goes_scn.available_dataset_names()

	# This is a warning regarding loss of projection information when converting to a PROJ string
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		goes_scn.load((av_dat_names))
	
	print('Channels: ' +str(av_dat_names))
	
	# This is a warning regarding loss of projection information when converting to a PROJ string
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')	
		goes_scn = goes_scn.resample(goes_scn.min_area(), resampler = 'native')
		
	ref_height = goes_scn['C'+str(channels[0]).zfill(2)].y.shape[0]
	ref_width = goes_scn['C'+str(channels[0]).zfill(2)].x.shape[0]
	
	goes_scn = goes_scn.aggregate(x=np.int(ref_width/width), 
	y=np.int(ref_height/height), func='mean')
	
	# RuntimeWarning: invalid value encountered in true_divide x = np.divide(x1, x2, out)
	# Caused by doing mean of nan-values in aggregate
	with warnings.catch_warnings():
		warnings.simplefilter('ignore')
		# CREATE combined dataset (full disk)
		keys = ['C'+str(i).zfill(2) for i in channels]
		values = [(["y", "x"], goes_scn['C' +str(i).zfill(2)].values) for i in channels]
		keys.append('gpm_precipitation')
		values.append((["y", "x"], gpm_transformed_data))
		data_vars_dict = dict(zip(keys, values))

		dataset = xr.Dataset(
			coords = dict(
				y = (["y"], goes_scn['C'+str(channels[0]).zfill(2)].y), 
				x = (["x"], goes_scn['C'+str(channels[0]).zfill(2)].x)),
			data_vars = data_vars_dict, 
			attrs = dict(
				time_goes = str({'start_time': goes_scn['C'+str(channels[0]).zfill(2)].attrs['start_time'],
					'end_time': goes_scn['C'+str(channels[0]).zfill(2)].attrs['end_time']}), 
				time_gpm = str({'start_time': gpm_file_t_0, 'end_time': gpm_file_t_1, 
					'approx_start_time_region': in_brazil_t_0, 'approx_end_time_region': in_brazil_t_1})
					))
	print('dataset created')
	
	# CROPPING dataset to interesting region
	mask_x = (dataset.x > region_corners[0])*(dataset.x < region_corners[2])
	mask_y = (dataset.y > region_corners[1])*(dataset.y < region_corners[3])
	sliced_dataset = dataset.where(mask_x & mask_y, drop = True)

	print('dataset cropped')

	# Writing dataset to file
	filename = path_to_store_processed_data+'/datasetS'+str(gpm_file_t_0).replace(" ", "")+'E'+str(gpm_file_t_1).replace(" ", "")+str(channels)+'created'+str(datetime.now()).replace(" ", "")+'.nc'
	sliced_dataset.to_netcdf(filename)
	
	print('dataset saved')
	
	# -------------------------------------------------------------------------
	
end_timing = time.time()
total_timing = end_timing-start_timing
print('total time: ' + str(total_timing)+ ' seconds')


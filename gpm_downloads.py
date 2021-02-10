from datetime import datetime
from funs_gpm_downloads import gpm_extract_datetime, extract_time_in_brazil, region_plot
from pansat.products.satellite.gpm import l2b_gpm_cmb
from pansat.products.satellite.goes import GOES16L1BRadiances
import numpy as np
from pyresample import kd_tree, geometry
from satpy import Scene
from pathlib import Path
import xarray as xr


############################### CONSTANTS ###############################

# GOES channels to download
channels = [8, 13]

# Defining rectangle
lower_left_x = 90000.0000
lower_left_y = -3400000.0000
upper_right_x = 3920000.0000
upper_right_y = 600000.0000 

# GOES data projection 
area_id = 'GOES-East'
description = '2km at nadir'
proj_id = 'abi_fixed_grid'
width =  2712 #5424 #Changing resolution from ~2 km to ~4 km
height =  2712 #5424 #Changing resolution from ~2 km to ~4 km
area_extent = [-5434894.8851, -5434894.8851, 5434894.8851, 5434894.8851]
projection = "+ellps=GRS80 +h=35786023 +lon_0=-75 +no_defs=None +proj=geos +sweep=x +type=crs +units=m +x_0=0 +y_0=0"

# Paths
linkfile = "2017-12-18--2018-12-17-with-shapefile.txt" 
path_to_download_storage_gpm = "GPM" #Path to where to store the gpm files
path_to_download_storage_goes = "GOES-16" #Path to where to store the goes files
path_to_store_processed_data = "Dataset"



###############################  ###############################


f = open(linkfile, "r") #File with all linkes from the Earth data search
gpm_file_t_0, gpm_file_t_1 = gpm_extract_datetime(f.readline())


# DOWNLOAD gpm data
'''
files_gpm = l2b_gpm_cmb.download(gpm_file_t_0, gpm_file_t_1, path_to_download_storage_gpm)
dataset = l2b_gpm_cmb.open(files_gpm[0])
'''
dataset_gpm = l2b_gpm_cmb.open("GPM/2B.GPM.DPRGMI.2HCSHv4-1.20180314-S194719-E211954.022965.V06A.HDF5") #TEST

# Arrival and departure times
in_brazil_t_0, in_brazil_t_1 = extract_time_in_brazil(projection,
							[lower_left_x, lower_left_y, upper_right_x, upper_right_y],
						 	dataset_gpm)

# Resample coordinates for GPM data into GOES data format
area_def = geometry.AreaDefinition(area_id, description, proj_id, projection, width, height, area_extent)
swath_def = geometry.SwathDefinition(lons=dataset_gpm['longitude'], lats=dataset_gpm['latitude'])
gpm_transformed_data = kd_tree.resample_nearest(swath_def, dataset_gpm['surface_precipitation'].values, area_def, radius_of_influence=50000, epsilon=0.5, fill_value=None)


# DOWNLOAD goes data
'''
files_goes = []
for channel in channels:
	files_tmp = GOES16L1BRadiances("F", channel).download(in_brazil_t_0, in_brazil_t_1, path_to_download_storage_goes)
	# TODO: What if there are several files that match the time?
	files_goes.append(files_tmp[0])
'''	


files_08 = [Path('GOES-16/OR_ABI-L1b-RadF-M3C08_G16_s20180732000427_e20180732011193_c20180732011242.nc')] #TEST
files_13 = [Path('GOES-16/OR_ABI-L1b-RadF-M3C13_G16_s20180732000427_e20180732011205_c20180732011259.nc')] #TEST
files_goes = [files_08[0], files_13[0]] #TEST



files_goes = map(str, files_goes)
goes_scn = Scene(reader='abi_l1b', filenames=files_goes)
goes_scn.load((goes_scn.available_dataset_names()))
goes_scn = goes_scn.aggregate(x=2, y=2, func='mean') # yields x = np.divide(x1, x2, out) error trying to get mean of nan values?


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
		area = str(area_def), 
		time_goes = str({'start_time': goes_scn['C'+str(channels[0]).zfill(2)].attrs['start_time'],
			'end_time': goes_scn['C'+str(channels[0]).zfill(2)].attrs['end_time']}), 
		time_gpm = str({'start_time': gpm_file_t_0, 'end_time': gpm_file_t_1, 
			'approx_start_time_region': in_brazil_t_0, 'approx_end_time_region': in_brazil_t_1})
			))


# CROPPING dataset to interesting region
new_area_extent = [lower_left_x, lower_left_y, upper_right_x, upper_right_y]
mask_x = (dataset.x > lower_left_x)*(dataset.x < upper_right_x)
mask_y = (dataset.y > lower_left_y)*(dataset.y < upper_right_y)
sliced_dataset = dataset.where(mask_x & mask_y, drop = True)
new_area_def = geometry.AreaDefinition(area_id, description, proj_id, projection, sliced_dataset['C'+str(channels[0]).zfill(2)].shape[0], sliced_dataset['C'+str(channels[0]).zfill(2)].shape[1], new_area_extent)
sliced_dataset.attrs['area'] = str(new_area_def)

# Testing write and read
#print(sliced_dataset)

filename = path_to_store_processed_data+'/datasetS'+str(gpm_file_t_0).replace(" ", "")+'E'+str(gpm_file_t_1).replace(" ", "")+'.nc'
sliced_dataset.to_netcdf(filename)

#dataset_1 = xr.open_dataset(filename)
#print(dataset_1)


#region_plot(new_area_def, sliced_dataset, 'gpm_precipitation')





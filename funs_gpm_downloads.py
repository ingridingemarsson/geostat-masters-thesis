import datetime
import re
import pyproj
import numpy as np
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr
from pansat.products.satellite.gpm import l2b_gpm_cmb
from pyresample import kd_tree, geometry

# Extracts date from resulting link in the earth data search
def gpm_extract_datetime(mystr):
	datematch = re.findall(r'(?:\.\d{8})', mystr)[0]
	startmatch = re.findall(r'(?:S\d{6})', mystr)[0]
	endmatch = re.findall(r'(?:E\d{6})', mystr)[0]
	
	start = datetime.datetime.strptime(datematch[1:]+startmatch[1:],"%Y%m%d%H%M%S")
	end = datetime.datetime.strptime(datematch[1:]+endmatch[1:],"%Y%m%d%H%M%S")
	
	if(start.hour > end.hour):
		end += datetime.timedelta(days=1)

	return([start, end])
	
def extract_latlon_limits_from_region(projection, corners):
	lower_left_x, lower_left_y, upper_right_x, upper_right_y = corners
	# Extracting approximate latitude region
	p = pyproj.Proj(projection)
	a_lon, a_lat = p(lower_left_x, lower_left_y, inverse=True)
	b_lon, b_lat = p(upper_right_x, lower_left_y, inverse=True)
	c_lon, c_lat = p(lower_left_x, upper_right_y, inverse=True)
	d_lon, d_lat = p(upper_right_x, upper_right_y, inverse=True)
	
	cS = np.mean([a_lat, b_lat]) # South border
	cN = np.mean([c_lat, d_lat]) # North border
	cW = np.mean([a_lon, c_lon]) # West border
	cE = np.mean([b_lon, d_lon]) # East border
	
	return([cS, cN, cW, cE])

def extract_time_in_brazil(latlon_limits, dataset_gpm):
	cS, cN, cW, cE = latlon_limits

	# Create mask to remove data outside of Brazil rectangle
	lats_mean = np.mean(dataset_gpm["latitude"], axis=1)
	lons_mean = np.mean(dataset_gpm["longitude"], axis=1)
	mask_gpm = (lats_mean > cS)*(lats_mean < cN)*(lons_mean > cW)*(lons_mean < cE)

	# Extract timestamp for arrival and departure in region
	time = dataset_gpm['scan_time']
	in_brazil_t_0 = datetime.datetime.strptime((min(time.values[mask_gpm]).astype(str).split(".")[0]),"%Y-%m-%dT%H:%M:%S")
	in_brazil_t_1 = datetime.datetime.strptime((max(time.values[mask_gpm]).astype(str).split(".")[0]),"%Y-%m-%dT%H:%M:%S")
	return([in_brazil_t_0, in_brazil_t_1])


def region_plot(area_def, dataset, feature):
	crs = area_def.to_cartopy_crs()
	ax = plt.axes(projection=crs)
	ax.coastlines()
	ax.gridlines()
	ax.add_feature(cartopy.feature.BORDERS)
	plt.imshow(dataset[feature], transform=crs, extent=crs.bounds, origin='upper')
	plt.show()
	
def create_box_dataset(box_num, gpm_transformed_data, region_center_y_idy, projvec, files_gpm, number_of_pixels):
		box_idy_up = region_center_y_idy+int((box_num+1)*number_of_pixels)
		box_idy_do = region_center_y_idy+int(box_num*number_of_pixels)

		swath_idx_up = np.where(np.isnan(gpm_transformed_data[box_idy_up,:,0]) == False)[0]
		swath_idx_do = np.where(np.isnan(gpm_transformed_data[box_idy_do,:,0]) == False)[0]

		box_middle_x = int(np.mean([min(min(swath_idx_up), min(swath_idx_do)), max(max(swath_idx_up), max(swath_idx_do))]))
		box_idx_le = box_middle_x-int(0.5*number_of_pixels)
		box_idx_ri = box_middle_x+int(0.5*number_of_pixels)


		#plt.imshow(gpm_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,0])
		#plt.show()
		box_data = gpm_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,0]
		box_t_0 = np.nanmin(gpm_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,1]).astype('datetime64[ns]')
		box_t_1 = np.nanmax(gpm_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,1]).astype('datetime64[ns]')

		projcoords_x, projcoords_y = projvec
		box_area_extent = [projcoords_x[box_idx_le], projcoords_y[box_idy_up], projcoords_x[box_idx_ri], projcoords_y[box_idy_do]]

		dataset = xr.Dataset(
				coords = dict(
					y = (["y"], projcoords_y[box_idy_do:box_idy_up]), 
					x = (["x"], projcoords_x[box_idx_le:box_idx_ri])),
				data_vars = dict(gpm_precipitation = (["y","x"], box_data)), 
				attrs = dict(
					area_extent = box_area_extent,
					shape = [number_of_pixels,number_of_pixels],
					time_in = box_t_0, time_out = box_t_1,
					filename = files_gpm[0]
						))
						
		return(dataset)
		
def gpm_data_processing(number_of_pixels, gpm_file_time, path_to_download_storage_gpm, area_def, region_corners):

	'''
	# DOWNLOAD gpm data
	files_gpm = l2b_gpm_cmb.download(gpm_file_time[0], gpm_file_time[1], path_to_download_storage_gpm)
	dataset_gpm = l2b_gpm_cmb.open(files_gpm[0])
	'''
	# TEST
	files_gpm = ["GPM/2B.GPM.DPRGMI.2HCSHv4-1.20180314-S194719-E211954.022965.V06A.HDF5"]
	dataset_gpm = l2b_gpm_cmb.open(files_gpm[0]) #TEST

	# Resample coordinates for GPM data into GOES data format
	repeted_time = np.transpose(np.tile(dataset_gpm['scan_time'].astype(np.int),(49,1)))
	gpm_data = np.stack((dataset_gpm['surface_precipitation'].values, repeted_time), axis = 2)
	swath_def = geometry.SwathDefinition(lons=dataset_gpm['longitude'], lats=dataset_gpm['latitude'])
	gpm_transformed_data = kd_tree.resample_nearest(swath_def, gpm_data, 
	area_def, radius_of_influence=3600, epsilon=0, fill_value=np.nan)


	region_center_y = np.mean([region_corners[1],region_corners[3]])
	projvec = area_def.get_proj_vectors() #tuple (X,Y)
	projcoords_y = projvec[1]
	projcoords_x = projvec[0]
	projcoords_y_diff = np.abs(projcoords_y - region_center_y)
	region_center_y_idy = np.argmin(projcoords_y_diff)
	
	#TODO #box_num = -2 #-2, -1, 0, 1
	dataset1 = create_box_dataset(1, gpm_transformed_data, region_center_y_idy, projvec, files_gpm, number_of_pixels)
	datasetm2 = create_box_dataset(-2, gpm_transformed_data, region_center_y_idy, projvec, files_gpm, number_of_pixels)

		
	return([dataset1,datasetm2])
	
	
	

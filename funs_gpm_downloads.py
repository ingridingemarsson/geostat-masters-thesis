from goes_downloads_with_cache import *
from pansat.products.satellite.goes import GOES16L1BRadiances

import datetime
import re
import pyproj
import numpy as np
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import xarray as xr
from pansat.products.satellite.gpm import l2b_gpm_cmb
from pyresample import kd_tree, geometry
from satpy import Scene
import warnings




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



def region_plot(area_def, dataset, feature, filename):
	crs = area_def.to_cartopy_crs()
	ax = plt.axes(projection=crs)
	ax.coastlines()
	ax.gridlines(draw_labels=True)
	ax.add_feature(cartopy.feature.BORDERS)
	plt.imshow(dataset[feature], transform=crs, extent=crs.bounds, origin='upper')
	plt.savefig(filename)
	
	
	
def region_plot2(area_def, datasets, feature, num_of_pixels, filename):
	ext0 = datasets[0].area_extent
	ext1 = datasets[-1].area_extent
	low_left_x = min(ext0[0],ext1[0])
	low_left_y = min(ext0[1],ext1[1])
	high_right_x = max(ext0[2],ext1[2])
	high_right_y = max(ext0[3],ext1[3])
	new_area_ext = [low_left_x, low_left_y, high_right_x, high_right_y]
	new_height = len(datasets)*num_of_pixels
	new_width = int(new_height/np.abs(new_area_ext[3]-new_area_ext[1])*np.abs(new_area_ext[2]-new_area_ext[0]))
	area_def_region = area_def.copy(area_extent = new_area_ext, height = new_height, width = new_width)
	crs = area_def_region.to_cartopy_crs()
	fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize = (10,10))
	#ax = plt.axes(projection=crs)
	for i in range(len(datasets)):
		area_def_box = area_def_region.copy(area_extent = datasets[i].area_extent, height = datasets[i].shape[0], width = datasets[i].shape[1])
		crs2 = area_def_box.to_cartopy_crs()
		rect = Rectangle((datasets[i].area_extent[0], datasets[i].area_extent[1]), datasets[i].area_extent[2]-datasets[i].area_extent[0], datasets[i].area_extent[3]-datasets[i].area_extent[1],
			linewidth=0.5, edgecolor='gray', facecolor='none')
		ax.add_patch(rect)
		plt.imshow(datasets[i][feature], transform=crs2, extent=crs2.bounds, origin='upper')
	ax.coastlines()
	ax.gridlines(draw_labels=True)
	ax.add_feature(cartopy.feature.BORDERS)
	ax.set_xlim(new_area_ext[0], new_area_ext[2])
	ax.set_ylim(new_area_ext[1], new_area_ext[3])

	plt.savefig(filename)
	
	
	
def create_box_dataset(box_num, gpm_transformed_data, area_def, region_corners, files_gpm, number_of_pixels):

	box_idx_le, box_idy_do, box_idx_ri, box_idy_up = get_box_ind(box_num, region_corners, area_def, gpm_transformed_data, number_of_pixels)

	box_data = gpm_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,0]
	box_t_0 = np.nanmin(gpm_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,1]).astype('datetime64[ns]')
	box_t_1 = np.nanmax(gpm_transformed_data[box_idy_do:box_idy_up, box_idx_le:box_idx_ri,1]).astype('datetime64[ns]')

	projvec = area_def.get_proj_vectors()
	projcoords_x, projcoords_y = projvec
	box_ind_extent = [box_idx_le, box_idy_up, box_idx_ri, box_idy_do]
	box_area_extent = [projcoords_x[box_idx_le], projcoords_y[box_idy_up], projcoords_x[box_idx_ri], projcoords_y[box_idy_do]]

	dataset = xr.Dataset(
			coords = dict(
				y = (["y"], projcoords_y[box_idy_do:box_idy_up]), 
				x = (["x"], projcoords_x[box_idx_le:box_idx_ri])),
			data_vars = dict(gpm_precipitation = (["y","x"], box_data)), 
			attrs = dict(
				ind_extent = box_ind_extent,
				area_extent = box_area_extent,
				shape = [number_of_pixels,number_of_pixels],
				gpm_time_in = str(box_t_0), 
				gpm_time_out = str(box_t_1),
				filename_gpm = str(files_gpm[0])
					))
					
	return(dataset)
		
		
		
def gpm_data_processing(number_of_pixels, gpm_file_time, area_def, region_corners):

	
	# DOWNLOAD gpm data
	files_gpm = l2b_gpm_cmb.download(gpm_file_time[0], gpm_file_time[1])
	dataset_gpm = l2b_gpm_cmb.open(files_gpm[0])
	'''
	# TEST
	files_gpm = ["GPM/2B.GPM.DPRGMI.2HCSHv4-1.20180314-S194719-E211954.022965.V06A.HDF5"]
	dataset_gpm = l2b_gpm_cmb.open(files_gpm[0]) #TEST
	'''
	
	# Resample coordinates for GPM data into GOES data format
	repeted_time = np.transpose(np.tile(dataset_gpm['scan_time'].astype(np.int),(49,1)))
	gpm_data = np.stack((dataset_gpm['surface_precipitation'].values, repeted_time), axis = 2)
	swath_def = geometry.SwathDefinition(lons=dataset_gpm['longitude'], lats=dataset_gpm['latitude'])
	gpm_transformed_data = kd_tree.resample_nearest(swath_def, gpm_data, 
	area_def, radius_of_influence=3700, epsilon=0, fill_value=np.nan) #3600 --> missing pixels

	
	region_center_y = np.mean([region_corners[1],region_corners[3]])
	projvec = area_def.get_proj_vectors() #tuple (X,Y)
	projcoords_y = projvec[1]
	projcoords_x = projvec[0]
	projcoords_y_diff = np.abs(projcoords_y - region_center_y)
	region_center_y_idy = np.argmin(projcoords_y_diff)
	
	box_nums = get_box_num(area_def, number_of_pixels,  region_corners)
	
	datasets = []
	for box_num in box_nums:
		dataset_box = create_box_dataset(box_num, gpm_transformed_data, area_def, region_corners, files_gpm, number_of_pixels)
		datasets.append(dataset_box)
	return(datasets)
	
	
	
def get_box_ind(box_num, region_corners, area_def, gpm_transformed_data, number_of_pixels):
	region_center_y = np.mean([region_corners[1],region_corners[3]])
	projvec = area_def.get_proj_vectors() #tuple (X,Y)
	projcoords_y = projvec[1]
	projcoords_x = projvec[0]
	projcoords_y_diff = np.abs(projcoords_y - region_center_y)
	region_center_y_idy = np.argmin(projcoords_y_diff)
	
	box_idy_up = region_center_y_idy+int((box_num+1)*number_of_pixels)
	box_idy_do = region_center_y_idy+int(box_num*number_of_pixels)

	swath_idx_up = np.where(np.isnan(gpm_transformed_data[box_idy_up,:,0]) == False)[0]
	swath_idx_do = np.where(np.isnan(gpm_transformed_data[box_idy_do,:,0]) == False)[0]

	box_middle_x = int(np.mean([min(min(swath_idx_up), min(swath_idx_do)), max(max(swath_idx_up), max(swath_idx_do))]))
	box_idx_le = box_middle_x-int(0.5*number_of_pixels)
	box_idx_ri = box_middle_x+int(0.5*number_of_pixels)
	
	return([box_idx_le, box_idy_do, box_idx_ri, box_idy_up])
	
	
	
def get_box_num(area_def, number_of_pixels, region_corners):
	projvec = area_def.get_proj_vectors() #tuple (X,Y)
	projcoords_y = projvec[1]
	h0 = np.argmin(np.abs(projcoords_y-region_corners[1]))
	h1 = np.argmin(np.abs(projcoords_y-region_corners[3])) 
	box_height = np.abs(h1-h0)
	n = box_height/(2*number_of_pixels)
	box_num_ind = 2*np.int(np.round(n))
	box_nums = np.array(range(box_num_ind))-int(box_num_ind/2)
	
	return(box_nums)
	
	
def goes_data_processing(time, channels, ind_extent, area_def, shape_full_disk):

	box_idx_le, box_idy_up, box_idx_ri, box_idy_do = ind_extent
	height, width = shape_full_disk
	filenames_goes = download_cached(time[0], time[1], channels)
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
		
		projvec = area_def.get_proj_vectors()
		projcoords_x, projcoords_y = projvec


		dataset = xr.Dataset(
					coords = dict(
						y = (["y"], projcoords_y[box_idy_do:box_idy_up]), 
						x = (["x"], projcoords_x[box_idx_le:box_idx_ri])),
					data_vars = data_vars_dict, 
					attrs = dict(
						goes_time_in = str(goes_scn[av_dat_names[0]].attrs['start_time']), 
						goes_time_out = str(goes_scn[av_dat_names[0]].attrs['end_time']),
						filenames_goes = [str(filename_goes) for filename_goes in filenames_goes]
							))

	return(dataset)
	
	

import numpy as np
import re
import datetime
import yaml
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import xarray as xr
import cartopy
import cartopy.crs as ccrs
from pyresample import geometry, load_area


def pars_dataset_filename(mystr):
	#GPMGOES-oS2017-12-25T19:29:08-c[1,6,8,10,13]-p256-b1.nc
	
	d_match = re.findall(r'(oS\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})', mystr)[0]
	filestart = datetime.datetime.strptime(d_match[2:],"%Y-%m-%dT%H:%M:%S")
		
	c_match = re.findall(r"c\[([^)]+)\]", mystr)[0].split(",")
	channels = [int(item) for item in c_match]
	
	p_match = re.findall(r'(p\d+)', mystr)[0]
	number_of_pixels = int(p_match[1:])
	
	b_match = re.findall(r'(b-?\d+)', mystr)[0]
	box_number = int(b_match[1:])
	
	return [filestart, channels, number_of_pixels, box_number]
	
	

def initial_load():
	'''
	Loading projection information from file an passing
	to global variables.
	'''
	areas_filepath =  '../downloads/areas.yaml'
	
	global area_def 
	global projection 
	global region_corners
	global shape_full_disk
	global region_width
	global region_height
	area_def = load_area(areas_filepath, 'full_disk')

	area_file = open(areas_filepath)
	parsed_area_file = yaml.load(area_file, Loader=yaml.FullLoader)
	area_dict_full_disk = parsed_area_file['full_disk']
	area_dict_region = parsed_area_file['region']
	area_file.close()

	projection = area_dict_full_disk['projection']
	region_corners = area_dict_region['area_extent']
	region_width = area_dict_region['width']
	region_height = area_dict_region['height']
	shape_full_disk = area_dict_full_disk['shape']
	
	
def region_plot(dataset, feature, filename):
	'''
	TODO
	'''
	ext0 = dataset.area_extent #area extent of first box
	low_left_x = min(ext0[0], region_corners[0])
	low_left_y = min(ext0[1], region_corners[1])
	high_right_x = region_corners[2]
	high_right_y = region_corners[3]
	new_area_ext = [low_left_x, low_left_y, high_right_x, high_right_y] #area extent of region containing all boxes
	new_width = int(region_width*np.abs(new_area_ext[2]-new_area_ext[0]))
	
	area_def_region = area_def.copy(area_extent = new_area_ext, height = region_height, width = new_width)
	crs = area_def_region.to_cartopy_crs()
	
	fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize = (10,10))
	ax.coastlines()
	ax.gridlines(draw_labels=True)
	ax.add_feature(cartopy.feature.BORDERS)
	rect = Rectangle((region_corners[0], region_corners[1]), region_corners[2]-region_corners[0], region_corners[3]-region_corners[1], linewidth=0.7, edgecolor='red', facecolor='none')
	ax.add_patch(rect)
	
	#Plot box
	area_def_box = area_def_region.copy(area_extent = dataset.area_extent, height = dataset.shape[0], width = dataset.shape[1])
	crs2 = area_def_box.to_cartopy_crs()
	rect = Rectangle((dataset.area_extent[0], dataset.area_extent[1]), dataset.area_extent[2]-dataset.area_extent[0], dataset.area_extent[3]-dataset.area_extent[1],
		linewidth=0.5, edgecolor='orange', facecolor='none')
	ax.add_patch(rect)
	plt.imshow(dataset[feature], transform=crs2, extent=crs2.bounds, origin='upper')
	ax.set_global()
	plt.savefig(filename)
	

initial_load()

parentdir = '../downloads/Dataset/linkfile2017-12/'
filename = 'GPMGOES-oS2017-12-26T07:49:29-c[1,6,8,10,13]-p256-b-1.nc' # 'GPMGOES-oS2017-12-26T18:37:18-c[1,6,8,10,13]-p256-b1.nc'
filestart, channels, number_of_pixels, box_number = pars_dataset_filename(filename)
dataset = xr.open_dataset(parentdir+filename)
dataset.close()
print(dataset['C01'])
region_plot(dataset, 'C'+str(channels[0]).zfill(2), 'hej.png')




	

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
	
	

def initial_load(areas_filepath):
	'''
	Loading projection information from file an passing
	to global variables.
	'''
	
	global area_def 
	#global projection 
	global region_corners
	#global shape_full_disk
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
	





	

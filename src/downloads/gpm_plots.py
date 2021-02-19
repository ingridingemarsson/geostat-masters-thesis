import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cartopy
import cartopy.crs as ccrs
from pyresample import geometry, load_area

import settings

	
def region_plot(datasets, feature, filename):
	'''
	TODO
	'''
	ext0 = datasets[0].area_extent #area extent of first box
	ext1 = datasets[-1].area_extent #area extent of last box
	low_left_x = min(ext0[0],ext1[0], settings.region_corners[0])
	low_left_y = min(ext0[1],ext1[1], settings.region_corners[1])
	high_right_x = max(ext0[2],ext1[2], settings.region_corners[2])
	high_right_y = max(ext0[3],ext1[3], settings.region_corners[3])
	new_area_ext = [low_left_x, low_left_y, high_right_x, high_right_y] #area extent of region containing all boxes
	new_height = len(datasets)*settings.number_of_pixels
	new_width = int(new_height/np.abs(new_area_ext[3]-new_area_ext[1])*np.abs(new_area_ext[2]-new_area_ext[0]))
	
	area_def_region = settings.area_def.copy(area_extent = new_area_ext, height = new_height, width = new_width)
	crs = area_def_region.to_cartopy_crs()
	
	fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize = (10,10))
	ax.coastlines()
	ax.gridlines(draw_labels=True)
	ax.add_feature(cartopy.feature.BORDERS)
	rect = Rectangle((settings.region_corners[0], settings.region_corners[1]), settings.region_corners[2]-settings.region_corners[0], settings.region_corners[3]-settings.region_corners[1], 		linewidth=0.7, edgecolor='red', facecolor='none')
	ax.add_patch(rect)
	
	#Plot boxes
	for i in range(len(datasets)):
		area_def_box = area_def_region.copy(area_extent = datasets[i].area_extent, height = datasets[i].shape[0], width = datasets[i].shape[1])
		crs2 = area_def_box.to_cartopy_crs()
		rect = Rectangle((datasets[i].area_extent[0], datasets[i].area_extent[1]), datasets[i].area_extent[2]-datasets[i].area_extent[0], datasets[i].area_extent[3]-datasets[i].area_extent[1],
			linewidth=0.5, edgecolor='orange', facecolor='none')
		ax.add_patch(rect)
		plt.imshow(datasets[i][feature], transform=crs2, extent=crs2.bounds, origin='upper')
	ax.set_global()
	plt.savefig(filename)
	
	
	
	
	

	

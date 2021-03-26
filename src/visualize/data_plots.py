import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm

import cartopy
import cartopy.crs as ccrs
from pyresample import geometry, load_area


	
def region_plot(datasets, feature, filename, region_corners, number_of_pixels, area_def):
	'''
	TODO
	'''
	precip_norm = LogNorm(1e-2, 1e2)
	ext0 = datasets[0].area_extent #area extent of first box
	ext1 = datasets[-1].area_extent #area extent of last box
	low_left_x = min(ext0[0],ext1[0], region_corners[0])
	low_left_y = min(ext0[1],ext1[1], region_corners[1])
	high_right_x = max(ext0[2],ext1[2], region_corners[2])
	high_right_y = max(ext0[3],ext1[3], region_corners[3])
	new_area_ext = [low_left_x, low_left_y, high_right_x, high_right_y] #area extent of region containing all boxes
	new_height = len(datasets)*number_of_pixels
	new_width = int(new_height/np.abs(new_area_ext[3]-new_area_ext[1])*np.abs(new_area_ext[2]-new_area_ext[0]))
	
	area_def_region = area_def.copy(area_extent = new_area_ext, height = new_height, width = new_width)
	crs = area_def_region.to_cartopy_crs()
	
	fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize = (10,11))
	ax.coastlines()
	ax.gridlines(draw_labels=True)
	ax.add_feature(cartopy.feature.BORDERS)
	rect = Rectangle((region_corners[0], region_corners[1]), region_corners[2]-region_corners[0], region_corners[3]-region_corners[1], linewidth=0.7, edgecolor='red', facecolor='none')
	ax.add_patch(rect)
	
	#Plot boxes
	plot_title = 'Feature: ' + str(feature) + ',\n Center of box:\n' 
	for i in range(len(datasets)):
		area_def_box = area_def_region.copy(area_extent = datasets[i].area_extent, height = datasets[i].shape[0], width = datasets[i].shape[1])
		crs2 = area_def_box.to_cartopy_crs()
		rect = Rectangle((datasets[i].area_extent[0], datasets[i].area_extent[1]), datasets[i].area_extent[2]-datasets[i].area_extent[0], datasets[i].area_extent[3]-datasets[i].area_extent[1],
			linewidth=0.5, edgecolor='orange', facecolor='none')
		ax.add_patch(rect)
		if (feature == 'gpm_precipitation'):
			plt.imshow(np.isnan(datasets[i][feature]), transform=crs2, extent=crs2.bounds, origin='upper', cmap=plt.get_cmap('binary'), alpha = 0.1)
			plt.imshow(datasets[i][feature], transform=crs2, extent=crs2.bounds, origin='upper', norm=precip_norm, cmap=plt.get_cmap('GnBu'))
		else:
			plt.imshow(datasets[i][feature], transform=crs2, extent=crs2.bounds, origin='upper', cmap=plt.get_cmap('inferno'))
		plot_title += ' (' + str(i) + ') x=' + str(round((datasets[i].area_extent[2]+datasets[i].area_extent[0])/2, 2)) + ', y=' + str(round((datasets[i].area_extent[3]+datasets[i].area_extent[1])/2, 2)) +'\n'
	ax.set_global()
	ax.title.set_text(plot_title)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	plt.savefig(filename)
	fig.clear()
	plt.close(fig)
	
	

	
def region_plot_overlay(datasets, feature, filename, region_corners, number_of_pixels, area_def):
	'''
	TODO
	'''
	low = 1e-2
	precip_norm = LogNorm(low, 1e2)
	ext0 = datasets[0].area_extent #area extent of first box
	ext1 = datasets[-1].area_extent #area extent of last box
	low_left_x = min(ext0[0],ext1[0], region_corners[0])
	low_left_y = min(ext0[1],ext1[1], region_corners[1])
	high_right_x = max(ext0[2],ext1[2], region_corners[2])
	high_right_y = max(ext0[3],ext1[3], region_corners[3])
	new_area_ext = [low_left_x, low_left_y, high_right_x, high_right_y] #area extent of region containing all boxes
	new_height = len(datasets)*number_of_pixels
	new_width = int(new_height/np.abs(new_area_ext[3]-new_area_ext[1])*np.abs(new_area_ext[2]-new_area_ext[0]))
	
	area_def_region = area_def.copy(area_extent = new_area_ext, height = new_height, width = new_width)
	crs = area_def_region.to_cartopy_crs()
	
	fig, ax = plt.subplots(subplot_kw={'projection': crs}, figsize = (10,11))
	ax.coastlines()
	ax.gridlines(draw_labels=True)
	ax.add_feature(cartopy.feature.BORDERS)
	rect = Rectangle((region_corners[0], region_corners[1]), region_corners[2]-region_corners[0], region_corners[3]-region_corners[1], linewidth=0.7, edgecolor='red', facecolor='none')
	ax.add_patch(rect)
	
	#Plot boxes
	plot_title = 'Feature: ' + str(feature) + ',\n Center of box:\n' 
	for i in range(len(datasets)):
		area_def_box = area_def_region.copy(area_extent = datasets[i].area_extent, height = datasets[i].shape[0], width = datasets[i].shape[1])
		crs2 = area_def_box.to_cartopy_crs()
		rect = Rectangle((datasets[i].area_extent[0], datasets[i].area_extent[1]), datasets[i].area_extent[2]-datasets[i].area_extent[0], datasets[i].area_extent[3]-datasets[i].area_extent[1],
			linewidth=0.5, edgecolor='orange', facecolor='none')
		ax.add_patch(rect)
		if(np.sum(datasets[i]['gpm_precipitation']>low)==0):
			pass
		else:
			plt.contour(datasets[i]['gpm_precipitation'], transform=crs2, extent=crs2.bounds, origin='upper', norm=precip_norm, cmap=plt.get_cmap('GnBu'))
		plt.imshow(datasets[i][feature], transform=crs2, extent=crs2.bounds, origin='upper', cmap=plt.get_cmap('inferno'))
		plot_title += ' (' + str(i) + ') x=' + str(round((datasets[i].area_extent[2]+datasets[i].area_extent[0])/2, 2)) + ', y=' + str(round((datasets[i].area_extent[3]+datasets[i].area_extent[1])/2, 2)) +'\n'
	ax.set_global()
	ax.title.set_text(plot_title)
	ax.set_xlabel('x')
	ax.set_ylabel('y')
	plt.savefig(filename)
	fig.clear()
	plt.close(fig)
	
	
	
	
	

	

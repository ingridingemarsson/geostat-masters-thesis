
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import cartopy
import cartopy.crs as ccrs
from pyresample import geometry, load_area

import settings


def region_plot(dataset, feature, filename):
	'''
	Plots TODO 
	'''
	crs = settings.area_def.to_cartopy_crs()
	ax = plt.axes(projection=crs)
	ax.coastlines()
	ax.gridlines(draw_labels=True)
	ax.add_feature(cartopy.feature.BORDERS)
	plt.imshow(dataset[feature], transform=crs, extent=crs.bounds, origin='upper')
	plt.savefig(filename)
	
	
	
def region_plot2(datasets, feature, filename):
	'''
	TODO
	'''
	ext0 = datasets[0].area_extent
	ext1 = datasets[-1].area_extent
	low_left_x = min(ext0[0],ext1[0])
	low_left_y = min(ext0[1],ext1[1])
	high_right_x = max(ext0[2],ext1[2])
	high_right_y = max(ext0[3],ext1[3])
	new_area_ext = [low_left_x, low_left_y, high_right_x, high_right_y]
	new_height = len(datasets)*settings.number_of_pixels
	new_width = int(new_height/np.abs(new_area_ext[3]-new_area_ext[1])*np.abs(new_area_ext[2]-new_area_ext[0]))
	area_def_region = settings.area_def.copy(area_extent = new_area_ext, height = new_height, width = new_width)
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
	
	
	
	
	
def region_plot3(dataset):
	'''
	Plots TODO 
	'''
	re_area_def = load_area('areas.yaml', 'region')
	crs = re_area_def.to_cartopy_crs()
	ax = plt.axes(projection=crs)
	ax.coastlines()
	ax.gridlines(draw_labels=True)
	ax.add_feature(cartopy.feature.BORDERS)
	plt.imshow(dataset, transform=crs, extent=crs.bounds, origin='upper')
	plt.show()
	#plt.savefig(filename)
	

import numpy as np
import re
import os
import datetime
import yaml
import warnings
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import xarray as xr
import cartopy
import cartopy.crs as ccrs
from pyresample import geometry, load_area

import plot_dataset_funs
from plot_dataset_funs import initial_load, pars_dataset_filename, region_plot, region_plot_overlay



parser = argparse.ArgumentParser()
parser.add_argument(
		"-s",
		"--singles",
		help="Plot single boxes",
		type=bool,
		default=False,
	)
parser.add_argument(
		"-f",
		"--filepath",
		help="Path to dataset",
		type=str,
		default='../dataset/origin/',
	)
args = parser.parse_args()


areas_filepath =  '../dataset/downloads/files/areas.yaml'
initial_load(areas_filepath)

linkfile = 'linkfileYYYY-MM/'
parentdir = args.filepath + linkfile
storeagedir = 'images/'+linkfile

if not Path(storeagedir).exists():
	os.mkdir(storeagedir)

for subdir in [f.name for f in os.scandir(parentdir) if f.is_dir()]:
	if not Path(storeagedir+subdir).exists():
		os.mkdir(storeagedir+subdir)
	datasets = []
	for filename in os.listdir(Path(parentdir+subdir)):
		print(filename)
	
		if filename.endswith(".nc"):
			filestart, channels, number_of_pixels, box_number = pars_dataset_filename(filename)
			dataset = xr.open_dataset(parentdir+subdir+'/'+filename)
			dataset.close()
			datasets.append(dataset)

			if(args.singles == True):
				for channel in channels:
					# This is a warning regarding loss of projection information when converting to a PROJ string
					with warnings.catch_warnings():
						warnings.simplefilter('ignore')
						region_plot([dataset], 'C'+str(channel).zfill(2), storeagedir + subdir +'/'+ filename[:-3] + str('C'+str(channel).zfill(2))+'af.png',
							plot_dataset_funs.region_corners, number_of_pixels, plot_dataset_funs.area_def)
						
				with warnings.catch_warnings():
					warnings.simplefilter('ignore')
					region_plot([dataset], 'gpm_precipitation', storeagedir + subdir +'/'+ filename[:-3] + 'gpm'+'af.png', 
						plot_dataset_funs.region_corners, number_of_pixels, plot_dataset_funs.area_def)

			for channel in channels:
				print(channel)
				# This is a warning regarding loss of projection information when converting to a PROJ string
				with warnings.catch_warnings():
					warnings.simplefilter('ignore')
					region_plot(datasets, 'C'+str(channel).zfill(2), storeagedir + subdir +'/'+ str('C'+str(channel).zfill(2))+'af.png',
						plot_dataset_funs.region_corners, number_of_pixels, plot_dataset_funs.area_def)
					region_plot_overlay(datasets, 'C'+str(channel).zfill(2), storeagedir + subdir +'/'+ str('C'+str(channel).zfill(2))+'overlay_af.png',
						plot_dataset_funs.region_corners, number_of_pixels, plot_dataset_funs.area_def)                    
					
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				region_plot(datasets, 'gpm_precipitation', storeagedir + subdir +'/'+ 'gpm'+'af.png', 
					plot_dataset_funs.region_corners, number_of_pixels, plot_dataset_funs.area_def)




		

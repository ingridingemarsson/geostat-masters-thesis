import os
import warnings
from pathlib import Path

import xarray as xr

import plot_dataset_funs
from plot_dataset_funs import initial_load, pars_dataset_filename, region_plot, region_plot_overlay


number_of_pixels = 128
channels = list(range(8,17))
channels.remove(12)

areas_filepath =  '../dataset/downloads/files/areas.yaml'
initial_load(areas_filepath)

parentdir = '../dataset/origin/independent_goes/'
storeagedir = 'images/independent_goes/'

if not Path(storeagedir).exists():
	os.mkdir(storeagedir)

for subdir in [f.name for f in os.scandir(parentdir) if f.is_dir()]:
	if not Path(storeagedir+subdir).exists():
		os.mkdir(storeagedir+subdir)
	datasets = []
	for filename in os.listdir(Path(parentdir+subdir)):
		print(filename)
	
		if filename.endswith(".nc"):
			dataset = xr.open_dataset(parentdir+subdir+'/'+filename)
			dataset.close()
			datasets.append(dataset)

			for channel in channels:
				print(channel)
				# This is a warning regarding loss of projection information when converting to a PROJ string
				with warnings.catch_warnings():
					warnings.simplefilter('ignore')
					region_plot(datasets, 'C'+str(channel).zfill(2), storeagedir + subdir +'/'+ str('C'+str(channel).zfill(2))+'af.png',
						plot_dataset_funs.region_corners, number_of_pixels, plot_dataset_funs.area_def)




		

import numpy as np 
import os

import xarray as xr

linkdir = "linkfile454004"
rootdir = "downloads/Dataset/" + linkdir
channels = [8,13]
number_of_pixels = 256

filename_list = []
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        filepath = subdir + os.sep + file

        if filepath.endswith(".nc"):
            filename_list.append(filepath)
            
X = np.zeros((len(filename_list), len(channels), number_of_pixels, number_of_pixels))
y = np.zeros((len(filename_list), number_of_pixels, number_of_pixels))

for i in range(len(filename_list)):
	dataset_tmp = xr.open_dataset(filename_list[i])
	dataset_tmp.close()
	for j in range(len(channels)):
		X[i][j] = dataset_tmp['C'+str(channels[j]).zfill(2)].values
	y[i] = dataset_tmp['gpm_precipitation'].values



#np.save(linkdir+'X'+'.npy', X)
#np.save(linkdir+'y'+'.npy', y)
filename_file = 'dataset' + 'filenames' + '.txt'
with open(filename_file, 'w') as f:
	for item in filename_list:
		f.write("%s\n" % item)
np.save('dataset'+'X'+'.npy', X)
np.save('dataset'+'y'+'.npy', y)

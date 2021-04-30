import numpy as np
import os
from pathlib import Path

import xarray as xr


def ExtractSingles(channels, list_of_data_files, path_to_storage, N=86040000):

	if not Path.exists(Path(path_to_storage)):
		os.makedirs(path_to_storage)

	vals_tot = np.zeros((N, len(channels)), dtype=np.float32)
	labs_tot = np.zeros((N), dtype=np.float32)

	num_files = 0
	current_row = 0					
	for f in list_of_data_files:
		box_dataset = xr.open_dataset(f)
		box_dataset.close()

		label_exists = np.logical_and((np.isnan(box_dataset['gpm_precipitation'].values)==False),(box_dataset['gpm_precipitation'].values >= 0.0))
		idy, idx = np.where(label_exists)
		labs = box_dataset['gpm_precipitation'].values[idy,idx]
		vals = np.zeros((np.sum(label_exists),len(channels)))
		for idc in range(len(channels)):
			vals[:,idc] = box_dataset['C'+str(channels[idc]).zfill(2)].values[idy,idx]

		vals_exists = (np.isnan(vals).any(axis=1)==False)
		if not ((vals_exists==False).all()):
			vals = vals[vals_exists, :]
			labs = labs[vals_exists]

		num_new_rows = len(labs)
		vals_tot[current_row:current_row+num_new_rows, :] = vals
		labs_tot[current_row:current_row+num_new_rows] = labs
		current_row += num_new_rows

		print(num_files)
		num_files += 1
	
    
	vals_tot = vals_tot[:current_row, :]
	labs_tot = labs_tot[:current_row]
	print(len(labs_tot))

	np.save(os.path.join(path_to_storage,'X_singles_dataset.npy'), vals_tot)
	np.save(os.path.join(path_to_storage,'y_singles_dataset.npy'), labs_tot)
	
					

def ConvertToSingles(path_to_save_data, path_to_list, dat_type, split_frac, seed=0):
			
	list_of_data_files = np.loadtxt(path_to_list, dtype=str)	
	
	np.random.seed(seed) 
	index = np.arange(len(list_of_data_files))
	np.random.shuffle(index) 
	num_of_train_inds = int(split_frac*len(list_of_data_files))

	ExtractSingles(channels, list_of_data_files[index[:num_of_train_inds]], os.path.join(path_to_save_data, dat_type[0]))
	if split_frac < 1.0:
		ExtractSingles(channels, list_of_data_files[index[num_of_train_inds:]], os.path.join(path_to_save_data, dat_type[1]))
		


def ComputeStats(path_to_storage):

	vals_tot = np.load(os.path.join(path_to_storage,'X_singles_dataset.npy'))
	
	X_mean = np.mean(vals_tot, axis=0)
	X_std = np.zeros((vals_tot.shape[1]))
	for i in range(vals_tot.shape[1]):
		X_std[i] = np.std(vals_tot[:,i])
		
	stats = np.stack([X_mean, X_std])
	print(stats)
	
	np.save(os.path.join(path_to_storage, 'X_singles_stats.npy'), stats)



path_to_storage = 'dataset-singles/'  
channels = list(range(8,17))
channels.remove(12)

print('train and val')
path_to_list = 'lists/trainvallist.txt' 
dat_type = ['train', 'validation']
ConvertToSingles(path_to_storage, path_to_list, dat_type, 0.8) 
ComputeStats(os.path.join(path_to_storage, dat_type[0]))
print('stats created')

print('test')
path_to_list = 'lists/testlist.txt' 
dat_type = ['test']
ConvertToSingles(path_to_storage, path_to_list, dat_type[0], 1) 






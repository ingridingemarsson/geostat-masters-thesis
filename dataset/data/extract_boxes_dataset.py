import numpy as np
import os
from pathlib import Path

import xarray as xr


def SaveAsNumpy(list_of_data_files, path_to_save_data):
	
	if not Path.exists(Path(path_to_save_data)):
		os.makedirs(path_to_save_data)
	i=0								
	for f in list_of_data_files: 
		print(i)
		i+=1   
		new_file = Path(os.path.join(path_to_save_data,Path(os.path.basename(f)[:-3]+'.npy')))
		if not Path.exists(new_file):						
			dataset = xr.open_dataset(f)
			dataset.close()
			data = np.stack([dataset[var].values for var in dataset.data_vars])
			np.save(new_file, data)
			
					

def CreateSubdataset(path_to_save_data, path_to_list, dat_type, split_frac, subfolder, seed=0):
			
	list_of_data_files = np.loadtxt(path_to_list, dtype=str)	
	
	np.random.seed(seed) 
	index = np.arange(len(list_of_data_files))
	np.random.shuffle(index) 
	num_of_train_inds = int(split_frac*len(list_of_data_files))

	SaveAsNumpy(list_of_data_files[index[:num_of_train_inds]], os.path.join(path_to_save_data, dat_type[0], subfolder))
	if split_frac < 1.0:
		SaveAsNumpy(list_of_data_files[index[num_of_train_inds:]], os.path.join(path_to_save_data, dat_type[1], subfolder))
		
		


					
def ComputeStatsFromNumpyFiles(path_to_stats, channels, path_to_data):
    
    i = 1
    glob_mean = np.zeros(len(channels))
    glob_std = np.zeros(len(channels))

    for f in os.listdir(path_to_data):
        if f.endswith('.npy'):
            data = np.load(os.path.join(path_to_data,f))
            box = data[:-1]
            tmp_mean = np.array([np.nanmean(box[c]) for c in range(box.shape[0])])
            tmp_std = np.array([np.nanstd(box[c]) for c in range(box.shape[0])])
            glob_mean = (glob_mean*(i-1)+tmp_mean)/i
            glob_std = (glob_std*(i-1)+tmp_std)/i
            i+=1
            
    stats = np.stack([glob_mean, glob_std])
    print(stats)

    if not np.isnan(np.sum(stats)):
        np.save(os.path.join(path_to_stats,'stats.npy'), stats)
    else:
        print('Array contains NaN')

    return(stats)



path_to_save_data = 'dataset-boxes/' 
channels = list(range(8,17))
channels.remove(12)
subfolder = 'npy_files'

print('train and val')
path_to_list = 'lists/trainvallist.txt' 
dat_type = ['train', 'validation']
CreateSubdataset(path_to_save_data, path_to_list, dat_type, 0.8, subfolder)
ComputeStatsFromNumpyFiles(os.path.join(path_to_save_data, dat_type[0]), channels, os.path.join(path_to_save_data, dat_type[0], subfolder)) 

print('test')
path_to_list = 'lists/testlist.txt' 
dat_type = ['test']
CreateSubdataset(path_to_save_data, path_to_list, dat_type, 1, subfolder)



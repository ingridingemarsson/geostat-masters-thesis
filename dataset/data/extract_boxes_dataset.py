import numpy as np
import os
from pathlib import Path

import xarray as xr

from common import CreateListOfLinkfilesInSpan


def ConvertDatasetToNumpy(path_to_load_data, path_to_save_data):

	if not Path.exists(Path(path_to_save_data)):
		try:
			os.makedirs(path_to_save_data)
		except OSError:
			print ("Creation of the directory %s failed" % path_to_save_data)
		else:
			print ("Successfully created the directory %s " % path_to_save_data)		
	
					
	list_of_data_files = []				
	for path, subdir, files in os.walk(path_to_load_data):
		for file in files:    
			if file.endswith('.nc'):	
				new_file = Path(os.path.join(path_to_save_data,Path(os.path.basename(file)[:-3]+'.npy')))
				list_of_data_files.append(new_file)
				if not Path.exists(new_file):						
					dataset = xr.open_dataset(os.path.join(path,file))
					dataset.close()
					data = np.stack([dataset[var].values for var in dataset.data_vars])
					np.save(new_file, data)
		
		
					
def ComputeStatsFromNumpyFiles(path_to_stats, channels, path_to_data):
    
    i = 1
    glob_mean = np.zeros(len(channels))
    glob_std = np.zeros(len(channels))

    for file in os.listdir(path_to_data):
        if file.endswith('.npy'):
            data = np.load(os.path.join(path_to_data,file))
            box = data[:-1]
            tmp_mean = np.array([np.nanmean(box[c]) for c in range(box.shape[0])])
            tmp_std = np.array([np.nanstd(box[c]) for c in range(box.shape[0])])
            glob_mean = (glob_mean*(i-1)+tmp_mean)/i
            glob_std = (glob_std*(i-1)+tmp_std)/i
            i+=1
            
    stats = np.stack([glob_mean, glob_std])
    print(stats)

    if not np.isnan(np.sum(stats)):
        np.save(path_to_stats+'stats.npy', stats)
    else:
        print('Array contains NaN')

    return(stats)




channels = list(range(8,17))
channels.remove(12)


path_to_data = '../origin/'
path_to_storage = 'dataset-boxes/' 


traindirlist = CreateListOfLinkfilesInSpan(17, 12, 19, 12)
traindirlist = [path_to_data+elem for elem in traindirlist]
dattype = 'train/'
for elem in traindirlist:
    ConvertDatasetToNumpy(path_to_load_data = elem, path_to_save_data = path_to_storage+dattype+'npy_files/')
    print('elem', elem, 'in train done')

ComputeStatsFromNumpyFiles(path_to_stats = path_to_storage+dattype, channels=channels, 
                           path_to_data=path_to_storage+dattype+'npy_files/')

valdirlist = CreateListOfLinkfilesInSpan(20, 1, 20, 7)
valdirlist = [path_to_data+elem for elem in valdirlist]
dattype = 'validation/'
for elem in valdirlist:
    ConvertDatasetToNumpy(path_to_load_data = elem, path_to_save_data = path_to_storage+dattype+'npy_files/')
    print('elem', elem, 'in val done')


testdirlist = CreateListOfLinkfilesInSpan(20, 8, 21, 3)
testdirlist = [path_to_data+elem for elem in testdirlist]
dattype = 'test/'
for elem in testdirlist:
    ConvertDatasetToNumpy(path_to_load_data = elem, path_to_save_data = path_to_storage+dattype+'npy_files/')
    print('elem', elem, 'in test done')


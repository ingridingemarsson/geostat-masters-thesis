import numpy as np
import os

import xarray as xr

from common import CreateListOfLinkfilesInSpan

def ConvertDatasetToSingles(channels, path_to_data_list, path_to_storage, max_num_files=np.inf):
    vals_tot = []
    labs_tot = []	
    num_files = 0					
    for path_to_data in path_to_data_list:
        for path, subdir, files in os.walk(path_to_data):
            for file in files:    
                if file.endswith('.nc'):
                    if (num_files > max_num_files):
                        break
                    box_dataset = xr.open_dataset(os.path.join(path,file))
                    box_dataset.close()

                    label_exists = (np.isnan(box_dataset['gpm_precipitation'].values)==False)
                    idy, idx = np.where(label_exists)
                    labs = box_dataset['gpm_precipitation'].values[idy,idx]
                    vals = np.zeros((np.sum(label_exists),len(channels)))
                    for idc in range(len(channels)):
                        vals[:,idc] = box_dataset['C'+str(channels[idc]).zfill(2)].values[idy,idx]


                    vals_exists = (np.isnan(vals).any(axis=1)==False)

                    vals = vals[vals_exists, :]
                    labs = labs[vals_exists]

                    vals_tot.append(vals)
                    labs_tot.append(labs)
                    print(num_files)
                    num_files += 1

    vals_tot = np.concatenate(vals_tot)
    labs_tot = np.concatenate(labs_tot)


    np.save(path_to_storage + 'X_singles_dataset.npy', vals_tot)
    np.save(path_to_storage + 'y_singles_dataset.npy', labs_tot)

    X_mean = np.mean(vals_tot, axis=0)
    X_std = np.std(vals_tot, axis=0)
    stats = np.stack([X_mean, X_std])

    np.save(path_to_storage + 'X_singles_stats.npy', stats)


channels = list(range(8,17))
channels.remove(12)

path_to_data = '../origin/'
path_to_storage = 'dataset-singles/' 


traindirlist = CreateListOfLinkfilesInSpan(17, 12, 18, 3)
traindirlist = [path_to_data+elem for elem in traindirlist]
dattype = 'train/'
ConvertDatasetToSingles(channels, traindirlist, path_to_storage+dattype)

valdirlist = CreateListOfLinkfilesInSpan(18, 4, 18, 4)
valdirlist = [path_to_data+elem for elem in valdirlist]
dattype = 'validation/'
ConvertDatasetToSingles(channels, valdirlist, path_to_storage+dattype)

'''
testdirlist = CreateListOfLinkfilesInSpan(19, 6, 19, 8)
testdirlist = [path_to_data+elem for elem in testdirlist]
dattype = 'test/'
ConvertDatasetToSingles(channels, testdirlist, path_to_storage+dattype)
'''

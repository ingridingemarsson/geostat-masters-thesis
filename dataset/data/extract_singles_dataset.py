import numpy as np
import os

import xarray as xr

from common import CreateListOfLinkfilesInSpan

def ConvertDatasetToSingles(channels, path_to_data_list, path_to_storage, max_num_files=np.inf):

    N = 86040000 #86034714
    print(N)
    vals_tot = np.zeros((N, len(channels)), dtype=np.float32)
    labs_tot = np.zeros((N), dtype=np.float32)
    
    num_files = 0
    current_row = 0					
    for path_to_data in path_to_data_list:
        for path, subdir, files in os.walk(path_to_data):
            for file in files:    
                if file.endswith('.nc'):
                    if (num_files > max_num_files):
                        break
                    box_dataset = xr.open_dataset(os.path.join(path,file))
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

                        #vals_tot.append(vals)
                        #labs_tot.append(labs)

                        num_new_rows = len(labs)
                        vals_tot[current_row:current_row+num_new_rows, :] = vals
                        labs_tot[current_row:current_row+num_new_rows] = labs
                        current_row += num_new_rows
                        
                        #print('num_new_rows:', num_new_rows)
                        #print('current_row:', current_row)
                            
                    print(num_files)
                    num_files += 1

    print('done with extracting all data')
    
    #vals_tot = np.concatenate(vals_tot)
    #labs_tot = np.concatenate(labs_tot)
    vals_tot = vals_tot[:current_row+1, :]
    labs_tot = labs_tot[:current_row+1]
    

    print('size of dataset:', len(labs_tot))

    np.save(path_to_storage + 'X_singles_dataset.npy', vals_tot)
    np.save(path_to_storage + 'y_singles_dataset.npy', labs_tot)

    #N = 20
    #di = int(len(vals_tot)/N) 
    #X_mean_sub = []   
    #X_std_sub = []
    #for i in range(N-1):
    #	vals_tot_sub = vals_tot[i:i+di]
    #	X_mean_sub.append(np.mean(vals_tot_sub, axis=0))
    #	X_std_sub.append(np.std(vals_tot_sub, axis=0))
    #	print('len vals_tot_sub', len(vals_tot_sub))
    #	
    #vals_tot_sub = vals_tot[i+1:]
    #X_mean_sub.append(np.mean(vals_tot_sub, axis=0))
    #X_std_sub.append(np.std(vals_tot_sub, axis=0))
    #print('len vals_tot_sub', len(vals_tot_sub))
    
    #X_mean = np.mean(X_mean_sub, axis=0)
    #X_mean = np.mean(X_mean_sub, axis=0)
    X_mean = np.mean(vals_tot, axis=0)
    X_std = np.mean(vals_tot, axis=0)
    stats = np.stack([X_mean, X_std])

    np.save(path_to_storage + 'X_singles_stats.npy', stats)


channels = list(range(8,17))
channels.remove(12)
#channels = [8, 9, 10, 11, 13]


path_to_data = '../origin/'
path_to_storage = 'dataset-singles/' #'dataset-test/' 


traindirlist = CreateListOfLinkfilesInSpan(17, 12, 19, 12) # (17, 12, 18, 2) 
traindirlist = [path_to_data+elem for elem in traindirlist]
dattype = 'train/'
ConvertDatasetToSingles(channels, traindirlist, path_to_storage+dattype)

valdirlist = CreateListOfLinkfilesInSpan(20, 1, 20, 7) # (18, 3, 18, 3) 
valdirlist = [path_to_data+elem for elem in valdirlist]
dattype = 'validation/'
ConvertDatasetToSingles(channels, valdirlist, path_to_storage+dattype)

testdirlist = CreateListOfLinkfilesInSpan(20, 8, 21, 3)
testdirlist = [path_to_data+elem for elem in testdirlist]
dattype = 'test/'
ConvertDatasetToSingles(channels, testdirlist, path_to_storage+dattype)


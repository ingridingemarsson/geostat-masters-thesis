import numpy as np
import os
from pathlib import Path



def listData(path_to_data_list, filename):

    list_of_data_files = []
    for path_to_data in path_to_data_list:
        for path, subdir, files in os.walk(path_to_data):
            for f in files:    
                if f.endswith('.nc'):	
                    list_of_data_files.append(os.path.join(path, f))
    np.savetxt(filename, list_of_data_files, fmt='%s')
    
    
    
def createSplit(path_to_lists, split_frac, filenames, seed=0):
    list_of_data_files = np.loadtxt(os.path.join(path_to_lists, 'trainvallist.txt'), dtype=str)
    
    np.random.seed(seed) 
    index = np.arange(len(list_of_data_files))
    np.random.shuffle(index) 
    num_of_train_inds = int(split_frac*len(list_of_data_files))
    
    trainfiles = list_of_data_files[index[:num_of_train_inds]]
    np.savetxt(os.path.join(path_to_lists, filenames[0]), trainfiles, fmt='%s')
    valfiles = list_of_data_files[index[num_of_train_inds:]]
    np.savetxt(os.path.join(path_to_lists, filenames[1]), valfiles, fmt='%s')
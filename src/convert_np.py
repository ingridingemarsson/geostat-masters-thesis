import numpy as np
import os
from pathlib import Path

import xarray as xr



def ConvertDatasetToNumpy(path_to_load_data, path_to_save_data):

	if not Path.exists(Path(path_to_save_data)):
		try:
			os.mkdir(path_to_save_data)
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
    
    if not Path.exists(Path(path_to_stats)):
        i = 1
        glob_mean = np.zeros(len(channels))
        glob_std = np.zeros(len(channels))

        for file in os.listdir(path_to_data):
            if file.endswith('.npy'):
                data = np.load(os.path.join(path_to_data,file))
                box = data['box']
                tmp_mean = np.array([np.nanmean(box[c]) for c in range(box.shape[0])])
                tmp_std = np.array([np.nanstd(box[c]) for c in range(box.shape[0])])
                glob_mean = (glob_mean*(i-1)+tmp_mean)/i
                glob_std = (glob_std*(i-1)+tmp_std)/i
                i+=1
            
        stats = np.stack([glob_mean, glob_std])

        if not np.isnan(np.sum(stats)):
            np.save(path_to_stats, stats)
        else:
            print('Array contains NaN')
    else:
        stats = np.load(path_to_stats)

    return(stats)




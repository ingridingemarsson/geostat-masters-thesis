import numpy as np
import os
import time

import xarray as xr

start_timing = time.time()

channels = [8,13]
dattype = 'train/' #'train/'
max_num_files = np.inf #64
path_to_data = '/home/ingrid/Documents/Git/gpm/src/data/first_net_dataset/origin/'+dattype
path_to_storage = '/home/ingrid/Documents/Git/gpm/src/data/first_net_dataset/singles/'+dattype

vals_tot = []
labs_tot = []	
num_files = 0					
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

	
end_timing = time.time()
total_timing = end_timing-start_timing
print('total time: ' + str(total_timing)+ ' seconds')		


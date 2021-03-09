import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import xarray as xr

class ToTensor(object):
    '''
	TODO
    '''

    def __call__(self, sample):
        box, label = sample['box'], sample['label']

        # torch image: C X H X W
        return {'box': torch.from_numpy(box).float(),
                'label': torch.from_numpy(label).float()}


class Standardize(object):
	'''
	TODO
	'''
	
	def __init__(self, path_to_data, path_to_stats, channels):
			
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
				print(stats)
			else:
				print('Array contains NaN')
		else:
			stats = np.load(path_to_stats)
			print(stats)

		self.stats = stats
		
	    
	def __call__(self, sample):
		box, label = sample['box'], sample['label']
		
		standardized_box = np.stack([(box[i]- self.stats[0, i])/self.stats[1, i] for i in range(self.stats.shape[0])])
		
		#label = label*0.01
		filled_label = np.where(np.isnan(label), -1, label)

		return {'box': standardized_box, 'label': filled_label}


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


class GOESRETRIEVALSDataset(Dataset):
	'''
	TODO
	'''
	def __init__(self, path_to_data, channels, transform=None, singles=True):
		self.path_to_data = path_to_data
		self.channels = ['C'+str(channel).zfill(2) for channel in channels]
		self.transform=transform
		self.singles = singles

	
		list_of_data_files = []				
		for file in os.listdir(self.path_to_data):
			if file.endswith('.npy'):	
				list_of_data_files.append(os.path.join(path_to_data,file))
			 		
		self.list_of_data_files = list_of_data_files

				
	
	def __len__(self):
		return len(self.list_of_data_files)
		
	
	def __getitem__(self, idx):
		data_file_path = self.list_of_data_files[idx]
		data = np.load(data_file_path)
		
		if (self.singles == False):
			box = np.asarray(data[:-1]) #Unit K
			label = np.asarray(data[-1]) #Unit mm/h
		else:
			center_y, center_x = [int(c/2) for c in data[-1].shape]
			box = np.asarray(data[:-1, center_y,center_x])
			label = np.asarray(data[-1, center_y,center_x])
		
		sample = {'box': box, 'label': label}
		
		if (self.transform != None):
			sample = self.transform(sample)

		return(sample)
		
	def getfilename(self, idx):
		return(self.list_of_data_files[idx])

'''		
dataset = GOESImageRETRIEVALSDataset(path_to_data  = '/home/ingrid/Documents/Exjobb/Dendrite/UserAreas/Ingrid/Dataset/linkfile2017-12', channels = channels, transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)


for data in dataloader:
	print('Data: ', data)
	print('Box: ', data['box'])
'''



          
          
          

import numpy as np
import os
import torch
import matplotlib.pyplot as plt
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
        return {'box': torch.from_numpy(box),
                'label': torch.from_numpy(label)}


class GOESRETRIEVALSDataset(Dataset):
	def __init__(self, path_to_data, channels, transform=None):
		self.path_to_data = path_to_data
		self.channels = ['C'+str(channel).zfill(2) for channel in channels]
		self.transform=transform
		
						
		list_of_data_files = []				
		for path, subdir, files in os.walk(self.path_to_data):
			for file in files:    
				if file.endswith('.nc'):
					list_of_data_files.append(os.path.join(path,file)) 			
		
		self.list_of_data_files = list_of_data_files
		
	
	def __len__(self):
		return len(self.list_of_data_files)
		
	
	def __getitem__(self, idx):
		box_dataset_file_path = self.list_of_data_files[idx]
		box_dataset = xr.open_dataset(box_dataset_file_path)
		box_dataset.close()

		box = np.stack([box_dataset[channel].values for channel in self.channels], axis = 0)
		label = box_dataset['gpm_precipitation'].values
		
		sample = {'box': box, 'label': label}
		
		if self.transform:
			sample = self.transform(sample)

		
		return(sample)
		

channels = [8,13]		
dataset = GOESRETRIEVALSDataset(path_to_data  = 'downloads/Dataset/linkfile2017-12', channels = channels)#, transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

'''
for data in dataloader:
	print('Data: ', data)
	print('Box: ', data['box'])
'''		



          
          
          

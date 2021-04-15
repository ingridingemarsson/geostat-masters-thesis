import numpy as np
import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class RandomLog(object):
	'''
	Log of labels, where zero vals are replaced by small random values.
	
	'''
	
	def __init__(self):
		pass
	
	def __call__(self, sample):
		box, label = sample['box'], sample['label']
		
		num_vals_to_replace = len(label == 0.0)
		new_rand_vals = np.random.uniform(1e-4, 1e-3, num_vals_to_replace)
		replaced_label = np.where(label == 0.0, new_rand_vals, label)
		
		logged_label = np.where(replaced_label > 0.0, np.log(replaced_label), replaced_label)

		return {'box': box, 'label': logged_label}



class Mask(object):
	'''
	TODO
	'''
	
	def __init__(self, fillvalue = -1):
		self.fillvalue = fillvalue
		
		
	def __call__(self, sample):
		box, label = sample['box'], sample['label']
		
		filled_box = np.stack([np.where(np.isnan(box[i]), self.fillvalue, box[i]) for i in range(box.shape[0])])
		
		filled_label = np.where(np.isnan(label), self.fillvalue, label)

		return {'box': filled_box, 'label': filled_label}
		


class RandomCrop(object):
	'''
	Random crop of the sample.
	
	Args:
		output_size (int): Desired output size. 
	'''
	
	def __init__(self, output_size):
		self.output_size = output_size
	
	def __call__(self, sample):
		box, label = sample['box'], sample['label']
		
		h, w = label.shape
		new_d = self.output_size
		
		top = np.random.randint(0, h - new_d)
		left = np.random.randint(0, w - new_d)
		
		cropped_box = np.stack([box[i, top: top + new_d, left: left + new_d] for i in range(box.shape[0])])
		cropped_label = label[top: top + new_d, left: left + new_d]
		
		return {'box': cropped_box, 'label': cropped_label}


class Standardize(object):
	'''
	Computes or loads dataset mean and std from data files and standardizes sample.
	
	Args:
		path_to_data: Path to directory where numpy data files are stored
		path_to_stats: Path to file to save to/load from
		channels: List of channels in data
		
	Returns:
		New sample dicitionary with standardized sample
	'''
	
	def __init__(self, path_to_data, path_to_stats, channels):
			
		if not Path.exists(Path(path_to_stats)):
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

			if not np.isnan(np.sum(stats)):
				np.save(path_to_stats, stats)
				print('mean, std:')
				print(stats)
			else:
				print('Array contains NaN')
		else:
			stats = np.load(path_to_stats)

		self.stats = stats
		
	    
	def __call__(self, sample):
		box, label = sample['box'], sample['label']
		standardized_box = np.stack([(box[i]- self.stats[0, i])/self.stats[1, i] for i in range(self.stats.shape[1])])
		
		return {'box': standardized_box, 'label': label}



class ToTensor(object):
    '''
	Converts numpy sample to tensor.
	
	Args:
		sample: Dictionary containing training and reference data
		
	Returns:
		new sample dictionary with tensors
	
    '''

    def __call__(self, sample):
        box, label = sample['box'], sample['label']

        # torch image: C X H X W
        return {'box': torch.from_numpy(box).float(),
                'label': torch.from_numpy(label).float()}






class GOESRETRIEVALSDataset(Dataset):
	'''
	TODO
	'''
	def __init__(self, path_to_data, channels, transform=None):
		self.path_to_data = path_to_data
		self.channels = ['C'+str(channel).zfill(2) for channel in channels]
		self.transform=transform

	
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
		
		box = np.asarray(data[:-1]) #Unit K
		label = np.asarray(data[-1]) #Unit mm/h
		
		sample = {'box': box, 'label': label}
		
		
		if (self.transform != None):
			sample = self.transform(sample)

		return(sample)
		
	def getfilename(self, idx):
		return(self.list_of_data_files[idx])




          
          
          

import numpy as np
import pandas as pd
import yaml
import os
import re
import warnings
import datetime
import shutil
import torch
from pathlib import Path

import xarray as xr
from pyresample import kd_tree, geometry, load_area
from satpy import Scene

from pansat.download.providers.goes_aws import GOESAWSProvider
from pansat.products.satellite.goes import GOES16L1BRadiances
from quantnn.qrnn import QRNN


# GLOBAL VARIABLES ################################################################################################################################################################

global channels
channels = list(range(8,17))
channels.remove(12)

global num_quantiles
num_quantiles = 99

path_to_rain_gauge_data = '.'

global storage_path_temp
storage_path_temp = '../dataset/temp'
if not Path(storage_path_temp).exists():
	os.mkdir(storage_path_temp)

global storage_path_final
storage_path_final='rain_gauge_preds/'
if not Path(storage_path_final).exists():
	os.mkdir(storage_path_final)
	
global number_of_pixels
number_of_pixels = 128

area_path='../dataset/downloads/files/areas.yaml'

global area_def
area_def = load_area(area_path, 'full_disk')

area_file = open(area_path)
parsed_area_file = yaml.load(area_file, Loader=yaml.FullLoader)
area_dict_full_disk = parsed_area_file['full_disk']
area_dict_region = parsed_area_file['region']
area_file.close()

global region_corners
region_corners = area_dict_region['area_extent']

global shape_full_disk
shape_full_disk = area_dict_full_disk['shape']

global projection
projection = area_dict_full_disk['projection']

global qrnn
qrnn = QRNN.load('../results/qrnn_model.pckl')
global stats
stats = np.load('../dataset/data/dataset-boxes/train/stats.npy')


# FRAMEWORK SETUP #################################################################################################################################################################

def region_setup():
	'''
	From the information on the region and the number of pixels compute the y (North-South) placement of the cropped data.
	
	Returns:
		region_ind_extent: Pixel index extent (in reference to full disk) in order West, South, East, North
	'''

	projcoords_x, projcoords_y = area_def.get_proj_vectors()

	region_corners_idx_low = np.argmin(np.abs(projcoords_x-region_corners[0]))
	region_corners_idx_high = np.argmin(np.abs(projcoords_x-region_corners[2]))
	region_corners_idy_high = np.argmin(np.abs(projcoords_y-region_corners[1]))
	region_corners_idy_low = np.argmin(np.abs(projcoords_y-region_corners[3]))

	region_width = region_corners_idx_high-region_corners_idx_low
	region_height = region_corners_idy_high-region_corners_idy_low

	if(region_width % number_of_pixels > 0):
		new_region_width = int(np.ceil(region_width/number_of_pixels)*number_of_pixels)
		region_corners_idx_high += int(np.floor((new_region_width - region_width)/2))
		region_corners_idx_low += -int(np.ceil((new_region_width - region_width)/2))
		
	if(region_height % number_of_pixels > 0):
		new_region_height = int(np.ceil(region_height/number_of_pixels)*number_of_pixels)
		region_corners_idy_high += int(np.floor((new_region_height - region_height)/2))
		region_corners_idy_low += -int(np.ceil((new_region_height - region_height)/2))
		
	region_ind_extent = region_corners_idx_low, region_corners_idy_high, region_corners_idx_high, region_corners_idy_low
	
	return(region_ind_extent)
	
	
	
def get_gauge_locations(path_to_rain_gauge_data, region_ind_extent):
	region_corners_idx_low, __, __, region_corners_idy_low = region_ind_extent
	lonlats = pd.read_pickle(os.path.join(path_to_rain_gauge_data,'rain_gauge_locs.pckl'))
	colrows = []
	for lon, lat in zip(lonlats['lon'], lonlats['lat']):
		col, row = area_def.lonlat2colrow(lon, lat)
		colrows.append((col-region_corners_idx_low, row-region_corners_idy_low))

	colrows = pd.DataFrame(colrows, columns = ['cols', 'rows'])
	
	return(colrows)
	


def getHoursList(start, end):
	delta = end - start

	datehours = []
	for i in range(delta.days + 1):
		for h in range(0,24):
			datehours.append(start + datetime.timedelta(days=i, hours=h))
		
	datehours = datehours[:-int(24-delta.seconds/3600)+1]
	
	return(datehours)



class RetrieveHour():
		"""
		Class to handle processing of hourly prediction.
		"""
	
		def __init__(self, hour_start, hour_end):
			"""
			Args:

			"""
			self.hour_start = hour_start
			self.hour_end = hour_end
			
			
		def goes_filename_extract_datetime(self, mystr):
			'''
			Extracting start and end datetime from GOES combined product filename.
			
			Args:
				mystr: filename for GOES combined product
				
			Returns:
				start: datetime for measurement start
				end: datetime for measurement end
			'''
			
			
			startmatch = re.findall(r'(?:s\d{14})', mystr)[0]
			endmatch = re.findall(r'(?:e\d{14})', mystr)[0]
			
			start = datetime.datetime.strptime(startmatch[1:-1],"%Y%j%H%M%S")
			end = datetime.datetime.strptime(endmatch[1:-1],"%Y%j%H%M%S")
			
			if(start.hour > end.hour):
				end += datetime.timedelta(days=1)

			return([start, end])			
		
		def get_datetimes_in_range(self, padtime=5):
			channel = channels[0]
				
			p = GOES16L1BRadiances("F", channel)
			dest = Path(storage_path_temp)
			dest.mkdir(parents=True, exist_ok=True)

			provider = GOESAWSProvider(p)
			pad = datetime.timedelta(minutes=padtime)
			filenames0 = provider.get_files_in_range(self.hour_start-pad, self.hour_end, start_inclusive=False)	
			datetimes = [self.goes_filename_extract_datetime(filename) for filename in filenames0]	
			
			return(datetimes)	
		
		def make_retrievals(self, datetimes):
			extracted_predictions_agg = np.zeros((len(colrows['cols']), num_quantiles))
			retrievals = [self.Retrieve(datetime[0], datetime[1]) for datetime in datetimes]
			for retrieval in retrievals:	
				retrieval.download()
				retrieval.crop()
				retrieval.make_predicion()
				extracted_predictions = retrieval.extract_relevant_predictions()
				#retrieval.remove_files(files)
				extracted_predictions_agg = np.sum(np.stack([extracted_predictions_agg, extracted_predictions]), axis=0)
				del retrieval		
			extracted_predictions_agg = extracted_predictions_agg/len(retrievals)
			return(extracted_predictions_agg)
			
		def save(self, filename, aggregated_predictions):
			np.save(os.path.join(storage_path_final,filename), aggregated_predictions)
			
		class Retrieve():
		
			def __init__(self, start, end):
				self.start = start
				self.end = end			
						
			def download(self, no_cache=False):
				files = []
				for channel in channels:

					p = GOES16L1BRadiances("F", channel)
					provider = GOESAWSProvider(p)
					filenames = provider.get_files_in_range(self.start, self.end, start_inclusive=True)
					if(len(filenames)==0):
						files.append(None)
					else:
						f = filenames[0]
						path = os.path.join(storage_path_temp, f)
						
						if not Path(path).exists() or no_cache:
							data = provider.download_file(f, path)
						files.append(path)
				self.files = files							
							

		
			def crop(self): 
				filenames = map(str, self.files)
				scn = Scene(filenames=filenames, reader='abi_l1b')
				av_dat_names = scn.available_dataset_names()
				# This is a warning regarding loss of projection information when converting to a PROJ string
				with warnings.catch_warnings():
					warnings.simplefilter('ignore')
					scn.load((av_dat_names))				
					scn = scn.resample(scn.min_area(), resampler = 'native')
					height, width = shape_full_disk
					ref_height = scn[av_dat_names[0]].y.shape[0]
					ref_width = scn[av_dat_names[0]].x.shape[0]
					scn = scn.aggregate(x=np.int(ref_width/width), y=np.int(ref_height/height), func='mean')
					region_corners_idx_low, region_corners_idy_high, region_corners_idx_high, region_corners_idy_low = region_ind_extent
					self.values = np.stack([np.array(scn[av_dat_name].values[region_corners_idy_low:region_corners_idy_high, region_corners_idx_low:region_corners_idx_high]) for av_dat_name in av_dat_names])
				
				
			def make_predicion(self, split_nums=8):
				predictions = np.zeros((num_quantiles, self.values.shape[1], self.values.shape[2]))
				for u in range(int(split_nums)):
					for v in range(int(split_nums)):
						indsx = [u*int(self.values.shape[1]/split_nums), (u+1)*int(self.values.shape[1]/split_nums)]
						indsy = [v*int(self.values.shape[2]/split_nums), (v+1)*int(self.values.shape[2]/split_nums)]
						print(indsx, indsy)
						subdata = np.stack([self.values[i, indsx[0]:indsx[1],indsy[0]:indsy[1]] for i in range(len(channels))])
						subdata = np.stack([(subdata[i]- stats[0, i])/stats[1, i] for i in range(stats.shape[1])])
						subdata = torch.from_numpy(subdata).float()
						predictions[:, indsx[0]:indsx[1], indsy[0]:indsy[1]] = qrnn.predict(subdata.unsqueeze(0)).squeeze().detach().numpy()		
				self.predictions = predictions
			
			def extract_relevant_predictions(self):
				extracted_predictions = np.zeros((len(colrows['cols']), num_quantiles))
				i=0
				for col, row in zip(colrows['cols'], colrows['rows']):
					extracted_predictions[i, :] = self.predictions[:, col, row]
					i+=1
				del self.predictions
				return(extracted_predictions)
				
			def remove_files(self):
				
				for f in self.files:
					if Path(f).exists():
						os.remove(f)


global region_ind_extent
region_ind_extent = region_setup()
global colrows
colrows = get_gauge_locations(path_to_rain_gauge_data, region_ind_extent)


period_start = datetime.datetime(2020,3,2,22) 
period_end = datetime.datetime(2020,3,3,1) 


hourslist = getHoursList(period_start,period_end)
retrieve_hours = [RetrieveHour(hourslist[h_ind],hourslist[h_ind+1]) for h_ind in range(len(hourslist)-1)]

for retrieve_hour in retrieve_hours: 
	datetimes = retrieve_hour.get_datetimes_in_range()
	for d in datetimes:
		print(d)	
	aggregated_predictions = retrieve_hour.make_retrievals(datetimes)
	print(retrieve_hour.hour_start.strftime('%Y%m%d%H')+'.npy')
	retrieve_hour.save(retrieve_hour.hour_start.strftime('%Y%m%d%H')+'.npy', aggregated_predictions)
	del retrieve_hour

	
	
	
	
	
	
	
	
	


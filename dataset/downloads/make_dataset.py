

import numpy as np
import yaml
import os
import re
import warnings
import datetime
import argparse
import shutil
from pathlib import Path

import xarray as xr
from pyresample import kd_tree, geometry, load_area
from satpy import Scene

from pansat.products.satellite.gpm import l2b_gpm_cmb
from pansat.download.providers.goes_aws import GOESAWSProvider
from pansat.products.satellite.goes import GOES16L1BRadiances


# GLOBAL VARIABLES ################################################################################################################################################################

parser = argparse.ArgumentParser()
parser.add_argument(
	"-lf",
	"--linkfile",
	help="Path to Earth data search link file.",
	type=str,
	default="files/links/linkfiles/linkfileYYYY-MM.txt")
args = parser.parse_args()

global link_file_path
link_file_path = args.linkfile

global channels
channels = list(range(8,17))
channels.remove(12)

global used_remove
used_remove = True

global check_for_nans
check_for_nans = False

global storage_path_temp
storage_path_temp = '../temp'
if not Path(storage_path_temp).exists():
	os.mkdir(storage_path_temp)

global storage_path_final
storage_path_final='../origin'
if not Path(storage_path_final).exists():
	os.mkdir(storage_path_final)

global number_of_pixels
number_of_pixels = 256



area_path='files/areas.yaml'

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


# FRAMEWORK SETUP #################################################################################################################################################################

def box_setup():
	'''
	From the information on the region and the number of pixels compute the y (North-South) placement of the cropped boxes (before random shift).
	
	Returns:
		offset_range: region height minus boxes height, i.e. the remaining part of the region in which the boxes can be shifted in the y direction
		box_idy_low_center: The lowest y-index border of the 'middle' box (Northmost border)
		box_numbers: array of box indices
	'''

	# Create image boxes
	__, projcoords_y = area_def.get_proj_vectors()
       
	region_center_y = np.mean([region_corners[1], region_corners[3]]) # lower left y coordinate (SW), upper right y coordinate (NE)
	projcoords_y_diff = np.abs(projcoords_y - region_center_y)
	region_center_y_idy = np.argmin(projcoords_y_diff)

	region_min_y = np.argmin(np.abs(projcoords_y-region_corners[1])) #lowest y-index in region (Northmost)
	region_max_y = np.argmin(np.abs(projcoords_y-region_corners[3])) #highest y-index in region (Southmost)
	region_height = np.abs(region_max_y-region_min_y) #y extent of region in pixels

	number_of_boxes = np.int(np.round(region_height/number_of_pixels-1))
	offset_range = region_height-number_of_boxes*number_of_pixels #how much extra y extent is there to play with
	box_numbers = np.array(range(number_of_boxes))-int(number_of_boxes/2) #for iteration

	box_shift = 0
	if ((number_of_boxes % 2) != 0):
		box_shift = np.int(number_of_pixels/2) #correction for region center y index for odd number of boxes

	box_idy_low_center = region_center_y_idy-box_shift #The lowest y-index border of the 'middle' box (Northmost border)
	
	return(offset_range, box_idy_low_center, box_numbers)



# CLASS FOR DOWNLOADING AND PREPROCESSING #########################################################################################################################################

class MakeOverpass():
	"""
	Class handling matching of one gpm combined product file passage over region, to the closest goes data files.

	"""
	
	def __init__(self, link, offset_range, box_idy_low_center, box_numbers):
		"""
		Args:
			link: Link from Earth data search specifying the gpm combined product file.
		"""
		self.link = link
		self.offset_range = offset_range
		self.box_idy_low_center = box_idy_low_center
		self.box_numbers = box_numbers
		
	def gpm_link_extract_datetime(self):
		'''
		Extracting start and end datetime from Earth data search 2BCMB downloading link.
		'''
		
		datematch = re.findall(r'(?:\.\d{8})', self.link)[0]
		startmatch = re.findall(r'(?:S\d{6})', self.link)[0]
		endmatch = re.findall(r'(?:E\d{6})', self.link)[0]
		
		start = datetime.datetime.strptime(datematch[1:]+startmatch[1:],"%Y%m%d%H%M%S")
		end = datetime.datetime.strptime(datematch[1:]+endmatch[1:],"%Y%m%d%H%M%S")
		
		if(start.hour > end.hour):
			end += datetime.timedelta(days=1)

		self.gpm_file_start = start
		self.gpm_file_end = end
		
		
	def gpm_download(self):
		"""
		Uses pansat to download gpm combined product file.
		"""
		files = l2b_gpm_cmb.download(self.gpm_file_start, self.gpm_file_end, destination=storage_path_temp)
		self.gpm_file = files[0]
		self.gpm_data = l2b_gpm_cmb.open(self.gpm_file)
		
		
	def gpm_transform(self):
		'''
		Resamples gpm data from swath coordinates to goes data coordinates.
		'''
		
		rep_num = self.gpm_data['matched_pixels'].shape[0]
		times = self.gpm_data['scan_time'].values.astype(np.int)
		repeted_times = np.transpose(np.tile(times,(rep_num,1)))

		precip = self.gpm_data['surface_precipitation'].values
		label_data = np.stack((precip, repeted_times), axis = 2)

		swath_def = geometry.SwathDefinition(lons=self.gpm_data['longitude'], lats=self.gpm_data['latitude'])
		gpm_transformed_data = kd_tree.resample_nearest(swath_def, label_data, area_def, radius_of_influence=3700, epsilon=0, fill_value=np.nan)
					                        
		self.gpm_transformed_data = gpm_transformed_data
		
		
	def randomize_boxes_offset(self):
		"""
		Adds random offset in offset_range to box_idy_low_center.
		
		Args:
			offset_range: Range for possible y-shift of boxes
			box_idy_low_center: The lowest y-index border of the 'middle' box (Northmost border)
		"""
		offset = np.random.randint(self.offset_range)-np.int(self.offset_range/2) #how much to shift the region center y index
		self.box_idy_low_center = box_idy_low_center + offset
		
		
	def process_boxes(self):
		"""
		For each box in the overpass, create an instance of the MakeBox class, download goes data, process and store.
		
		Args:
			box_numbers: indices to iterate over the boxes.
		"""
		
		boxes = [self.MakeBox(box_number, self.gpm_transformed_data, self.gpm_file, self.gpm_file_start) for box_number in self.box_numbers]
		for box in boxes:
			box.gpm_data_crop(self.box_idy_low_center)
			if not (box.box_area_extent[0] < region_corners[0] or box.box_area_extent[2] > region_corners[2]): # Check that current box lies well inside region border
				files = box.download_cached()
				if not files==None: # Check that there exists matching goes files
					cont = box.goes_data_process()
					if cont:
						box.get_dataset_filename()
						box.save_combined_dataset()
			
	
	
	class MakeBox():
		"""
		Class to handle processing of each data box.
		"""
	
		def __init__(self, 
			box_number,
			gpm_transformed_data,
			gpm_file,
			gpm_file_start):
			"""
			Args:
				box_number: index of current box
				box_idy_low_center_shifted: The lowest y-index border of the 'middle' box (Northmost border) 
				gpm_transformed_data: the transformed gpm combined product data
				gpm_file: filename of gpm data file
				gpm_file_start: start datetime for whole gpm file 
			"""
			self.box_number = box_number
			self.gpm_transformed_data = gpm_transformed_data
			self.gpm_file = gpm_file
			self.gpm_file_start = gpm_file_start
			
		def gpm_data_crop(self, box_idy_low_center):
			'''
			Crop gpm data to current box shape.
			box_area_extent: list of projection coordinates specifying box borders (lower left x, lower left y, upper right x, upper right y)
			
			'''

			box_idy_low = box_idy_low_center + self.box_number*number_of_pixels #The lowest y-index border of the current box (Northmost border)
			box_idy_high = box_idy_low + number_of_pixels #The highest y-index border of the current box (Southmost border)

			#Check which x-indices is in the swath at the North and South box border
			swath_idx_low = np.where(np.isnan(self.gpm_transformed_data[box_idy_low,:,0]) == False)[0]
			swath_idx_high = np.where(np.isnan(self.gpm_transformed_data[box_idy_high,:,0]) == False)[0]

			box_idx_low_swath = min(min(swath_idx_high), min(swath_idx_low)) #The lowest x-index in the swath (Westmost)
			box_idx_high_swath = max(max(swath_idx_high), max(swath_idx_low)) #The highest x-index in the swath (Eastmost)
			box_middle_x = int(np.mean([box_idx_low_swath, box_idx_high_swath])) # Center of swath in x direction
			
			box_idx_low = box_middle_x-int(0.5*number_of_pixels) #The lowest x-index of the current box (Westmost border)
			box_idx_high = box_middle_x+int(0.5*number_of_pixels) #The highest x-index of the current box (Eastmost border)
			
			self.gpm_box_data = self.gpm_transformed_data[box_idy_low:box_idy_high, box_idx_low:box_idx_high,0] #Cropping the whole data to the current box
			
			gpm_time_in = np.nanmin(self.gpm_transformed_data[box_idy_low:box_idy_high, box_idx_low:box_idx_high,1]).astype('datetime64[ns]')
			self.gpm_time_in = datetime.datetime.strptime(str(gpm_time_in)[:-3],"%Y-%m-%dT%H:%M:%S.%f")
			
			gpm_time_out = np.nanmax(self.gpm_transformed_data[box_idy_low:box_idy_high, box_idx_low:box_idx_high,1]).astype('datetime64[ns]')
			self.gpm_time_out = datetime.datetime.strptime(str(gpm_time_out)[:-3],"%Y-%m-%dT%H:%M:%S.%f")

			self.box_ind_extent = [box_idx_low, box_idy_high, box_idx_high, box_idy_low] # In form of area extent: lower left x, lower left y, upper right x, upper right y --> West, South, East, North
			projcoords_x, projcoords_y = area_def.get_proj_vectors()
			self.box_area_extent = [projcoords_x[self.box_ind_extent[0]], projcoords_y[self.box_ind_extent[1]], projcoords_x[self.box_ind_extent[2]], projcoords_y[self.box_ind_extent[3]]]				

		

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
									
						
		def download_cached(self, no_cache=False, time_tol = 480):
			"""
			Download all files from the GOES satellite combined product with
			channels in channels in a given time range but avoid redownloading
			files that are already present.
			
			Does two checks: i) If there are two availible files for download, choose the file with smallest time
				difference compared to gpm time between either start or end timestamps.
				ii) If the time difference between center timestamps are greater than tolerance, return no files.

			Args:
				no_cache: If this is set to True, it forces a re-download of files even 
				if they are already present.
				time_tol: Allowed time difference between mid time in gpm and goes data files

			Returns:
				List of pathlib.Path object pointing to the available data files
				in the request time range.
			"""


			files = []
			
			#
			channel = channels[0]

			p = GOES16L1BRadiances("F", channel)
			dest = Path(storage_path_temp)
			dest.mkdir(parents=True, exist_ok=True)

			provider = GOESAWSProvider(p)
			filenames = provider.get_files_in_range(self.gpm_time_in, self.gpm_time_out, start_inclusive=True)
			if (len(filenames)>0):
				f_ind = 0
				if (len(filenames) == 2):
					timediff = []
					for filename in filenames:
						goes_start, goes_end = self.goes_filename_extract_datetime(filename)
						timediff.append(min(np.abs((self.gpm_time_in-goes_start).total_seconds()), np.abs((self.gpm_time_out-goes_end).total_seconds())))

					if (timediff[0] > timediff[1]):
						f_ind = 1

				goes_start, goes_end = self.goes_filename_extract_datetime(filenames[f_ind])
				timesmid_goes = goes_start + datetime.timedelta(seconds=int((goes_end-goes_start).total_seconds()/2)) 
				timesmid_gpm = self.gpm_time_in + datetime.timedelta(seconds=int((self.gpm_time_out-self.gpm_time_in).total_seconds()/2))            
					   
				if(np.abs((timesmid_goes-timesmid_gpm).total_seconds()) > time_tol):
				    return None
						       
				f = filenames[f_ind]
				path = dest / f
				
				if not path.exists() or no_cache:
					data = provider.download_file(f, path)
				files.append(path)
				
				for channel in channels[1:]:
					p = GOES16L1BRadiances("F", channel)
					dest = Path(storage_path_temp)
					dest.mkdir(parents=True, exist_ok=True)

					provider = GOESAWSProvider(p)
					filenames = provider.get_files_in_range(goes_start, goes_end, start_inclusive=True)
					
					goes_start_new, goes_end_new = self.goes_filename_extract_datetime(filenames[0])	
					if(np.abs((goes_start_new - goes_start).total_seconds())>60 or np.abs((goes_end_new - goes_end).total_seconds())>60):
						return None	
						
					f = filenames[0]
					path = dest / f
					
					if not path.exists() or no_cache:
						data = provider.download_file(f, path)
					files.append(path)				
						
				self.filenames_goes = files	
				
			else:
				return None			
			return(files)			
			
		
		def goes_data_process(self):
			'''
			Read and resample goes data.
			'''
			
			
			files_goes = map(str, self.filenames_goes)
			goes_scn = Scene(reader='abi_l1b', filenames=files_goes)
			av_dat_names = goes_scn.available_dataset_names()

			# This is a warning regarding loss of projection information when converting to a PROJ string
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				goes_scn.load((av_dat_names))

				
			self.goes_time_in = [str(goes_scn[av_dat_name].attrs['start_time']) for av_dat_name in av_dat_names]
			self.goes_time_out = [str(goes_scn[av_dat_name].attrs['end_time']) for av_dat_name in av_dat_names]


			# This is a warning regarding loss of projection information when converting to a PROJ string
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')	
				goes_scn = goes_scn.resample(goes_scn.min_area(), resampler = 'native')
				
			height, width = shape_full_disk
			ref_height = goes_scn[av_dat_names[0]].y.shape[0]
			ref_width = goes_scn[av_dat_names[0]].x.shape[0]

			goes_scn = goes_scn.aggregate(x=np.int(ref_width/width), y=np.int(ref_height/height), func='mean')

			keys = av_dat_names
			box_idx_low, box_idy_high, box_idx_high, box_idy_low = self.box_ind_extent
			
			# RuntimeWarning: invalid value encountered in true_divide x = np.divide(x1, x2, out)
			# Caused by doing mean of nan-values in aggregate
			with warnings.catch_warnings():
				warnings.simplefilter('ignore')
				values = []
				for av_dat_name in av_dat_names:
					av_dat_vals = goes_scn[av_dat_name].values[box_idy_low:box_idy_high, box_idx_low:box_idx_high]
					if(np.isnan(np.sum(av_dat_vals)) and check_for_nans):
						return False
					values.append((["y", "x"], av_dat_vals)) 

				
			self.keys = keys
			self.values = values
			
			return True
			
			
		def get_dataset_filename(self):
			'''
			Name combined gpm/goes box dataset files.
			'''
			parent_dir = storage_path_final + os.sep + os.path.basename(link_file_path).replace(".txt", "")
			
			if not Path(storage_path_final).exists():
				os.mkdir(storage_path_final)
				
			if not Path(parent_dir).exists():
				os.mkdir(parent_dir)
			
			common_filename = '/GPMGOES-' 
			common_filename += 'oS' + str(self.gpm_file_start).replace(" ", "T") 
			common_filename += '-c' + str(channels).replace(" ", "") 
			common_filename += '-p' + str(number_of_pixels) 

			common_dir = parent_dir + common_filename

			if not Path(common_dir).exists():
				os.mkdir(common_dir)
			
			combined_dataset_filename = common_dir + common_filename + '-b' + str(self.box_number) + '.nc'
			self.combined_dataset_filename = combined_dataset_filename				
						
		def save_combined_dataset(self):
			"""
			Save combined gpm/goes box dataset as xarray dataset in netCDF file.
			"""
			self.keys.append('gpm_precipitation')
			self.values.append((["y","x"], self.gpm_box_data))
			data_vars_dict = dict(zip(self.keys, self.values))
			
			box_dataset = xr.Dataset(
				data_vars = data_vars_dict, 
					attrs = dict(
						ind_extent = self.box_ind_extent,
						area_extent = self.box_area_extent,
						shape = [number_of_pixels, number_of_pixels],
						gpm_time_in = str(self.gpm_time_in), 
						gpm_time_out = str(self.gpm_time_out),
						goes_time_in = self.goes_time_in,
						goes_time_out = self.goes_time_out,
						filename_gpm = str(os.path.basename(self.gpm_file)),
						filenames_goes = [str(os.path.basename(filename_goes)) for filename_goes in self.filenames_goes]))
			box_dataset = box_dataset.astype(np.float32)
			
			
			box_dataset_filename = self.combined_dataset_filename
			box_dataset.to_netcdf(box_dataset_filename)
			box_dataset.close()
			
			

# EXECUTE #########################################################################################################################################################################

# Calculate box info
offset_range, box_idy_low_center, box_numbers = box_setup()

# Load list of links
link_file = open(link_file_path, "r") 
link_list = link_file.readlines()
link_file.close()

# Download and process the data
i = 0
for link in link_list:
	print(i)
	i+=1
	overpass = MakeOverpass(link, offset_range, box_idy_low_center, box_numbers)
	overpass.gpm_link_extract_datetime()
	overpass.gpm_download()
	overpass.gpm_transform()
	overpass.randomize_boxes_offset()
	overpass.process_boxes()

	# Remove temporary files
	if(i%10==0):
		if (used_remove == True):			
			for f in os.listdir(storage_path_temp):
			    os.remove(os.path.join(storage_path_temp, f))			
							
if (used_remove == True):			
	for f in os.listdir(storage_path_temp):
	    os.remove(os.path.join(storage_path_temp, f))	









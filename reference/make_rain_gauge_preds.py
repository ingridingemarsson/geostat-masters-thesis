import numpy as np
import pandas as pd
import yaml
import os
import re
import warnings
import datetime
import shutil
import torch
import time
from pathlib import Path
from scipy import ndimage
import argparse

import xarray as xr
from pyresample import kd_tree, geometry, load_area
from satpy import Scene

from pansat.download.providers.goes_aws import GOESAWSProvider
from pansat.products.satellite.goes import GOES16L1BRadiances
from quantnn.qrnn import QRNN
import sys
sys.path.append('../src') #to be able to import mlp.pckl

# GLOBAL VARIABLES ################################################################################################################################################################

# ARGUMENTS
parser = argparse.ArgumentParser(description='Train fully-connected QRNN')
parser.add_argument(
    "--day",
    help="Day in month.",
    type=int,
    default=1
    )
parser.add_argument(
    "--save_path",
    help="Path to store predictions.",
    type=str,
    default='rain_gauge_preds/'
    )
parser.add_argument(
    "--temp_path",
    help="Path to store temp files.",
    type=str,
    default='temp_gauge/'
    )
args = parser.parse_args()

global channels
channels = list(range(8,17))
channels.remove(12)

global quantiles
quantiles = np.linspace(0.01, 0.99, 99)
global q_ind_to_save
q_ind_to_save = [94, 98]

path_to_rain_gauge_data = '.'

global storage_path_temp
storage_path_temp = args.temp_path #'temp_gauge'+str(args.day)
if not Path(storage_path_temp).exists():
    os.makedirs(storage_path_temp)
    
print(storage_path_temp)

global storage_path_final
storage_path_final = args.save_path #'rain_gauge_preds/'+str(args.day)
if not Path(storage_path_final).exists():
    os.makedirs(storage_path_final)
print(storage_path_final)

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

global xception
xception = QRNN.load('../results/models/xception64_[100]_0.01__boxes_100_0.01_0_t5412_v1354[0, 1, 2, 3, 4, 5, 6, 7]_Adam_1622288705.386947.pckl') #xception.pckl')
global mlp
mlp = QRNN.load('../results/models/singles_fc32786_[100]_0.001__singles_100_0.001_0_t83360758_v20805499[0, 1, 2, 3, 4, 5, 6, 7]_Adam_1622293711.867882.pckl') #mlp.pckl')

global stats
f = open('../path_to_data.txt', "r")
path_to_dataset = f.readline().rstrip('\n')
f.close() 
print(path_to_dataset)
stats = np.load(os.path.join(path_to_dataset,'data','stats.npy')) #np.load('../dataset/data/stats.npy')
print(stats)

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


'''
def get_gauge_locations(path_to_rain_gauge_data, region_ind_extent):
    region_corners_idx_low, __, __, region_corners_idy_low = region_ind_extent
    lonlats = pd.read_pickle(os.path.join(path_to_rain_gauge_data,'rain_gauge_locs.pckl'))
    colrows = []
    for lon, lat in zip(lonlats['lon'], lonlats['lat']):
        with warnings.catch_warnings():    
            warnings.simplefilter('ignore')
            col, row = area_def.lonlat2colrow(lon, lat)
            colrows.append((col-region_corners_idx_low, row-region_corners_idy_low))

    colrows = pd.DataFrame(colrows, columns = ['cols', 'rows'])

    return(colrows)
'''


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

        def get_datetimes_in_range(self, padtime=0):
            channel = channels[0]

            p = GOES16L1BRadiances("F", channel)
            dest = Path(storage_path_temp)
            dest.mkdir(parents=True, exist_ok=True)

            provider = GOESAWSProvider(p)
            pad = datetime.timedelta(minutes=padtime)
            filenames0 = provider.get_files_in_range(self.hour_start-pad, self.hour_end, start_inclusive=False)	
            datetimes = [self.goes_filename_extract_datetime(filename) for filename in filenames0]	
            self.datetimes = datetimes

        def make_retrievals(self):
            retrievals = [self.Retrieve(datetime[0], datetime[1]) for datetime in self.datetimes]
            main_preds_list = []
            nans_at_loc_list = []
            for retrieval in retrievals:
                retrieval.download()
                retrieval.crop()
                nans_at_loc = retrieval.handleNaNs(stats)
                nans_at_loc_list.append(nans_at_loc)
                #retrieval.save_raw_data()
                main_preds_xception = retrieval.make_prediction(xception, stats, 'boxes', split_nums=1)
                main_preds_mlp = retrieval.make_prediction(mlp, stats, 'singles', split_nums=1)
                main_preds_list.append(np.concatenate([main_preds_xception, main_preds_mlp]))     
                retrieval.remove_files()
                del retrieval		
            print(nans_at_loc_list)
            extract_nans_at_loc = np.stack(nans_at_loc_list).any(axis=0)
            print(extract_nans_at_loc)
            print(extract_nans_at_loc.shape)
            extract_main_predictions_agg = np.mean(main_preds_list, axis=0)
            extract_main_predictions_agg = np.where(np.tile(extract_nans_at_loc, (extract_main_predictions_agg.shape[0],1,1)), np.nan, extract_main_predictions_agg)
            print(extract_main_predictions_agg)
            self.extract_main_predictions_agg = extract_main_predictions_agg

        #def save(self, filename, aggregated_predictions):
        #    np.save(os.path.join(storage_path_final,filename), aggregated_predictions)

        def save_preds(self, filename):
            keys = ['posterior_mean']
            keys.extend(['Q'+"{:0.2f}".format(quantiles[i]) for i in q_ind_to_save])   
            mods = ['xception', 'mlp']
            keys = [s+'_'+v for s in mods for v in keys]
            values_list = []
            for i in range(self.extract_main_predictions_agg.shape[0]):
                values_list.append((["y", "x"], self.extract_main_predictions_agg[i])) 

            projcoords_x, projcoords_y  = area_def.get_proj_vectors()
            area_extent = [projcoords_x[region_ind_extent[0]], projcoords_y[region_ind_extent[1]],
                        projcoords_x[region_ind_extent[2]], projcoords_y[region_ind_extent[3]]]

            data_vars_dict = dict(zip(keys, values_list))
            dataset = xr.Dataset(
                data_vars = data_vars_dict, 
                    attrs = dict(
                        ind_extent = region_ind_extent,
                        area_extent = area_extent,
                        shape = [region_ind_extent[2]-region_ind_extent[0], region_ind_extent[1]-region_ind_extent[3]],
                        start = str(self.hour_start),
                        end = str(self.hour_end), 
                        datetimes = [str(datetime) for datetime in self.datetimes]))
            dataset = dataset.astype(np.float32)
            print(dataset)
            dataset.to_netcdf(os.path.join(storage_path_final,filename+'.nc'))
            dataset.close()            


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
                self.keys = av_dat_names
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

            def handleNaNs(self, stats):
                mask = (np.isnan(self.values).any(axis=0))    
                #print(mask)
                #print(mask.shape)
                #print(self.values.shape)
                #print(np.tile(mask,(self.values.shape[0],1,1)).shape)
                #print(np.tile(stats[0], (self.values.shape[1]*self.values.shape[2],1)).T.reshape(self.values.shape).shape)
                self.values = np.where(np.tile(mask,(self.values.shape[0],1,1)), np.tile(stats[0], (self.values.shape[1]*self.values.shape[2],1)).T.reshape(self.values.shape), self.values)
                
                print(np.sum(mask))
                
                mask = ndimage.binary_dilation(mask, structure=np.ones((16,16))).astype(mask.dtype)
                print(np.sum(mask))
                return(mask)


            def save_raw_data(self):
                values_list = []
                for i in range(len(self.keys)):
                    vals = self.values[i]
                    values_list.append((["y", "x"], vals)) 

                projcoords_x, projcoords_y  = area_def.get_proj_vectors()
                area_extent = [projcoords_x[region_ind_extent[0]], projcoords_y[region_ind_extent[1]], projcoords_x[region_ind_extent[2]], projcoords_y[region_ind_extent[3]]]		

                data_vars_dict = dict(zip(self.keys, values_list))
                dataset = xr.Dataset(
                    data_vars = data_vars_dict, 
                        attrs = dict(
                            ind_extent = region_ind_extent,
                            area_extent = area_extent,
                            shape = [region_ind_extent[2]-region_ind_extent[0], region_ind_extent[1]-region_ind_extent[3]],
                            goes_time_in = str(self.start),
                            goes_time_out = str(self.end), 
                            filenames_goes = [str(os.path.basename(filename_goes)) for filename_goes in self.files]))
                dataset = dataset.astype(np.float32)

                dataset_filename = 'GOES_s'+ self.start.strftime('%Y%m%d%H%M%S')+'e'+self.end.strftime('%Y%m%d%H%M%S')+'.nc'
                dataset.to_netcdf(os.path.join(storage_path_final,dataset_filename))
                dataset.close()	

            def make_prediction(self, model, stats, data_type, split_nums=8):
                predictions = np.zeros((len(quantiles), self.values.shape[1],self.values.shape[1]))
                y_mean = np.zeros((self.values.shape[1], self.values.shape[2]))
                for u in range(int(split_nums)):
                    for v in range(int(split_nums)):
                        indsx = [u*int(self.values.shape[1]/split_nums), (u+1)*int(self.values.shape[1]/split_nums)]
                        indsy = [v*int(self.values.shape[2]/split_nums), (v+1)*int(self.values.shape[2]/split_nums)]
                        print(indsx, indsy)
                        subdata = np.stack([self.values[i, indsx[0]:indsx[1],indsy[0]:indsy[1]] for i in range(len(channels))])
                        subdata = np.stack([(subdata[i]-stats[0, i])/stats[1, i] for i in range(stats.shape[1])])
                        subdata = torch.from_numpy(subdata).float()
                        if data_type=='boxes': 
                            print('boxes')
                            #UserWarning: Default upsampling behavior when mode=bilinear is changed to align_corners=False since 0.4.0. Please specify align_corners=True if the old behavior is desired. See the documentation of nn.Upsample for details. warnings.warn("Default upsampling behavior when mode={} is changed "
                            with warnings.catch_warnings():
                                warnings.simplefilter('ignore')
                                
                                #print(subdata.shape)
                                inp = subdata.unsqueeze(0)
                                #print(inp.shape)
                                with torch.no_grad():
                                    preds = model.predict(inp)
                                #print(preds.shape)
                                preds = preds.squeeze().detach().numpy()
                                #print(preds.shape)
                                
                                predictions[:, indsx[0]:indsx[1], indsy[0]:indsy[1]] = preds
                                
                                with torch.no_grad():
                                    y_mean[indsx[0]:indsx[1], indsy[0]:indsy[1]] = model.posterior_mean(
                                    inp).squeeze().detach().numpy()     
                                
                        elif data_type=='singles':
                            print('singles')
                            #print(subdata.shape)
                            inp = subdata
                            #inp = inp.unsqueeze(0)
                            #print(inp.shape)
                            inp = torch.flatten(inp, start_dim=1)
                            #print(inp.shape)
                            #inp = torch.squeeze(inp)
                            #print(inp.shape)
                            inp = torch.transpose(inp, 0, 1)
                            #print(inp.shape)
                            with torch.no_grad():
                                preds = model.predict(inp)
                            #print(preds.shape)
                            #preds = preds.squeeze().detach()
                            preds = preds.detach()
                            preds = torch.transpose(preds, 0, 1)
                            #print(preds.shape)
                            preds = torch.reshape(preds,
                                (len(quantiles),
                                 int(self.values.shape[1]/split_nums),
                                 int(self.values.shape[2]/split_nums))).numpy()
                            #print(preds.shape)
                            
                            predictions[:, indsx[0]:indsx[1], indsy[0]:indsy[1]] = preds
                            
                            with torch.no_grad():
                                y_mean[indsx[0]:indsx[1], indsy[0]:indsy[1]] = torch.reshape(
                                model.posterior_mean(inp).detach(),
                                (int(self.values.shape[1]/split_nums),
                                 int(self.values.shape[2]/split_nums))).numpy() 
                res = [y_mean]              
                res.extend([predictions[i] for i in q_ind_to_save]) 
                return(np.stack(res))
                #return(np.stack([y_mean, predictions[94], predictions[98]]))

            #def extract_rain_gauge_predictions(self):
            #	extracted_predictions = np.zeros((len(colrows['cols']), len(quantiles)))
            #	i=0
            #	for col, row in zip(colrows['cols'], colrows['rows']):
            #		extracted_predictions[i, :] = self.predictions[:, col, row]
            #		i+=1
            #	del self.predictions
            #	return(extracted_predictions)

            def remove_files(self):

                for f in self.files:
                    if Path(f).exists():
                        os.remove(f)


global region_ind_extent
region_ind_extent = region_setup()
#global colrows
#colrows = get_gauge_locations(path_to_rain_gauge_data, region_ind_extent)


#period_start = datetime.datetime(2020,3,3,5) 
#period_end = datetime.datetime(2020,3,3,6) 

#period_start = datetime.datetime(2020,12,args.day,18) 
#period_end = period_start+datetime.timedelta(hours=6)

period_start = datetime.datetime(2020,12,args.day,0) 
period_end = period_start+datetime.timedelta(hours=24)

hourslist = getHoursList(period_start,period_end)
retrieve_hours = [RetrieveHour(hourslist[h_ind],hourslist[h_ind+1]) for h_ind in range(len(hourslist)-1)]

for retrieve_hour in retrieve_hours: 
    start = time.time()
    retrieve_hour.get_datetimes_in_range()
    retrieve_hour.make_retrievals()
    print(retrieve_hour.hour_end.strftime('%Y%m%d%H')+'.npy')
    retrieve_hour.save_preds(retrieve_hour.hour_end.strftime('%Y%m%d%H'))
    del retrieve_hour
    end = time.time()
    print(f"Retrieve hour time {end - start}")












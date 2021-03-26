# From Simons example
import numpy as np
from pathlib import Path
import os
import re
import datetime

from pansat.download.providers.goes_aws import GOESAWSProvider
from pansat.products.satellite.goes import GOES16L1BRadiances
import downloads.settings as st

#CHANNELS = [8, 13]


def download_cached(start_time, end_time, CHANNELS, no_cache=False, time_tol = 480):
	"""
	Download all files from the GOES satellite combined product with
	channels in CHANNELS in a given time range but avoid redownloading
	files that are already present.

	Args:
		start_time: datetime object specifying the start of the time
		range for which to download files.
		end_time: datetime object specifying the end of the time range.
		no_cache: If this is set to True, it forces a re-download of files even 
		if they are already present.

	Returns:
		List of pathlib.Path object pointing to the available data files
		in the request time range.
	"""


	files = []
	for channel in CHANNELS:

		p = GOES16L1BRadiances("F", channel)
		dest = Path(st.path_to_store_goes_data)
		dest.mkdir(parents=True, exist_ok=True)

		provider = GOESAWSProvider(p)
		filenames = provider.get_files_in_range(start_time, end_time, start_inclusive=True)
		print('start_time', start_time)
		print('end_time', end_time)
		f_ind = 0
		if (len(filenames) == 2):
		
			times0start, times0end = goes_filename_extract_datetime(filenames[0])
			timediff0 = min(np.abs((start_time-times0start).total_seconds()), np.abs((end_time-times0end).total_seconds()))
			print(timediff0)
			times1start, times1end = goes_filename_extract_datetime(filenames[1])
			timediff1 = min(np.abs((start_time-times1start).total_seconds()), np.abs((end_time-times1end).total_seconds()))
			print(timediff1)
			if (timediff0 > timediff1):
				f_ind = 1
		print(f_ind)
		timesstart, timesend = goes_filename_extract_datetime(filenames[f_ind])
		print('timesstart', timesstart)
		print('timesend', timesend)
		timesmid_goes = timesstart + datetime.timedelta(seconds=int((timesend-timesstart).total_seconds()/2)) 
		print('timesmid_goes', timesmid_goes)
		timesmid_label = start_time + datetime.timedelta(seconds=int((end_time-start_time).total_seconds()/2))                       
		if(np.abs((timesmid_goes-timesmid_label).total_seconds()) > time_tol):
		    return None
		                       
		f = filenames[f_ind]
		parent_dir = dest / Path(st.linkfile.replace(".txt", ""))
		path = parent_dir / f
		
			
		if not parent_dir.exists():
			os.mkdir(parent_dir)
		
		if not path.exists() or no_cache:
			data = provider.download_file(f, path)
		files.append(path)
		
	return files



def goes_filename_extract_datetime(mystr):
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
	



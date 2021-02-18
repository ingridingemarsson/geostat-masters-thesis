# From Simons example
import numpy as np
from pathlib import Path
import os
import re
import datetime

from pansat.download.providers.goes_aws import GOESAWSProvider
from pansat.products.satellite.goes import GOES16L1BRadiances
import settings

#CHANNELS = [8, 13]


def download_cached(start_time, end_time, CHANNELS, no_cache=False):
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
		dest = Path(settings.path_to_store_goes_data)
		dest.mkdir(parents=True, exist_ok=True)

		provider = GOESAWSProvider(p)
		filenames = provider.get_files_in_range(start_time, end_time, start_inclusive=True)
		
		f_ind = 0
		if (len(filenames) == 2):
		
			__, times0end = goes_filename_extract_datetime(filenames[0])
			timediff0 = np.abs((start_time-times0end).total_seconds())
			
			times1start, __ = goes_filename_extract_datetime(filenames[1])
			timediff1 = np.abs((times1start-end_time).total_seconds())
			
			if (timediff0 < timediff1):
				f_ind = 1
				
		
		
		f = filenames[f_ind]
		parent_dir = dest / Path(settings.linkfile.replace(".txt", ""))
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
	



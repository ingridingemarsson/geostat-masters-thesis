# From Simons example
import numpy as np
from pathlib import Path
import os

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
	start_time: datetime object specifying the start of the time range
	    for which to download files.
	end_time: datetime object specifying the end of the time range.
	no_cache: If this is set to True, it forces a re-download of files even
	     if they are already present.

	Returns:
	List of pathlib.Path object pointing to the available data files
	in the request time range.
	"""
	#global CACHE

	files = []
	for channel in CHANNELS:

		p = GOES16L1BRadiances("F", channel)
		dest = Path(settings.path_to_store_goes_data)
		dest.mkdir(parents=True, exist_ok=True)

		provider = GOESAWSProvider(p)
		filenames = provider.get_files_in_range(start_time, end_time, start_inclusive=True)
		#filename_to_date
		
		time_diffs = []
		for f in filenames:
			time_diffs.append(np.abs((p.filename_to_date(f)-start_time).total_seconds()))
			
			
		f_ind = np.argmin(time_diffs)
		
		
		
		'''
		for f in filenames:
			path = dest / f
			if not path.exists() or no_cache:
				data = provider.download_file(f, path)
			files.append(path)
		'''
		
		#TODO
		#f_ind = 0
		f = filenames[f_ind]
		parent_dir = dest / Path(settings.linkfile.replace(".txt", ""))
		path = parent_dir / f
		
			
		if not parent_dir.exists():
			os.mkdir(parent_dir)
		
		if not path.exists() or no_cache:
			data = provider.download_file(f, path)
		files.append(path)
		
	return files
    


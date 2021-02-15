# From Simons example
from pathlib import Path
from pansat.download.providers.goes_aws import GOESAWSProvider
from pansat.products.satellite.goes import GOES16L1BRadiances

#CHANNELS = [8, 13]


def download_cached(start_time, end_time, CHANNELS, no_cache=False):
	"""
	Download all files from the GOES satellite products in PRODUCTS in a given
	time range but avoid redownloading files that are already present.

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
	global CACHE

	files = []
	for channel in CHANNELS:

		p = GOES16L1BRadiances("F", channel)
		dest = Path(p.default_destination)
		dest.mkdir(parents=True, exist_ok=True)

		provider = GOESAWSProvider(p)
		filenames = provider.get_files_in_range(start_time, end_time, start_inclusive=True)
		
		'''
		for f in filenames:
			path = dest / f
			if not path.exists() or no_cache:
				data = provider.download_file(f, path)
			files.append(path)
		'''
		
		f = filenames[0]
		path = dest / f
		if not path.exists() or no_cache:
			data = provider.download_file(f, path)
		files.append(path)
		
	return files
    
'''
from datetime import datetime
t_0 = datetime(2021, 2, 11, 10, 0)    
t_1 = datetime(2021, 2, 11, 10, 5)

files = download_cached(t_0, t_1)
print(files)
'''

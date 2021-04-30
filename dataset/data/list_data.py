import numpy as np
import os
from pathlib import Path

import xarray as xr

from common import CreateListOfLinkfilesInSpan

def listData(path_to_data_list, filename):
	
	list_of_data_files = []
	for path_to_data in path_to_data_list:								
		for path, subdir, files in os.walk(path_to_data):
			for f in files:    
				if f.endswith('.nc'):	
					list_of_data_files.append(os.path.join(path, f))
	np.savetxt(filename, list_of_data_files, fmt='%s')
					
				
path_to_data = '../origin/'
path_to_storage = 'lists/' 

if not Path.exists(Path(path_to_storage)):
	os.makedirs(path_to_storage)

trainvaldirlist = CreateListOfLinkfilesInSpan(17, 12, 20, 3)
trainvaldirlist = [path_to_data+elem for elem in trainvaldirlist]
listData(trainvaldirlist, os.path.join(path_to_storage, 'trainvallist.txt'))

testdirlist = CreateListOfLinkfilesInSpan(20, 4, 21, 3)
testdirlist = [path_to_data+elem for elem in testdirlist]
listData(testdirlist, os.path.join(path_to_storage, 'testlist.txt'))

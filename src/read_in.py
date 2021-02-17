import numpy as np
import matplotlib.pyplot as plt
import yaml
from datetime import datetime

import xarray as xr
from pyresample import geometry
from pyresample import load_area

'''
from downloads.funs_gpm_downloads import extract_latlon_limits_from_region
from downloads.gpm_plots import region_plot, region_plot2
import downloads.settings

settings.parse_arguments()
settings.initial_load()


dataset_0 = xr.open_dataset('Dataset/linkfile454004/GPMGOES-oS2018-11-02T23:55:03-c[8,13]-p256/GPMGOES-oS2018-11-02T23:55:03-c[8,13]-p256-nS2018-11-03T00:17:05.692999936-b0.nc')
dataset_0.close()
dataset_1 = xr.open_dataset('Dataset/linkfile454004/GPMGOES-oS2018-11-02T23:55:03-c[8,13]-p256/GPMGOES-oS2018-11-02T23:55:03-c[8,13]-p256-nS2018-11-03T00:14:21.892999936-b1.nc')
dataset_1.close()
dataset_2 = xr.open_dataset('Dataset/linkfile454004/GPMGOES-oS2018-11-02T23:55:03-c[8,13]-p256/GPMGOES-oS2018-11-02T23:55:03-c[8,13]-p256-nS2018-11-03T00:11:28.292000000-b2.nc')
dataset_2.close()
dataset_3 = xr.open_dataset('Dataset/linkfile454004/GPMGOES-oS2018-11-02T23:55:03-c[8,13]-p256/GPMGOES-oS2018-11-02T23:55:03-c[8,13]-p256-nS2018-11-03T00:08:08.792000000-b3.nc')
dataset_3.close()
print(dataset_1)



region_plot2([dataset_1],'gpm_precipitation', 'exregionplotp'+'.png')

region_plot2([dataset_0, dataset_1, dataset_2, dataset_3], 'gpm_precipitation',  'wholeexregionplotp'+'.png')
'''


dataset = xr.open_dataset('downloads/Dataset/linkfile464113/GPMGOES-oS2018-10-31T13:17:53-c[8,13]-p256/GPMGOES-oS2018-10-31T13:17:53-c[8,13]-p256-nS2018-10-31T14:25:30.382000128-b0.nc')
dataset.close()

plt.imshow(dataset['gpm_precipitation'])
plt.imshow(dataset['C08'])
plt.imshow(dataset['C13'])
plt.show()

print(dataset)


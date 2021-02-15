import xarray as xr
from funs_gpm_downloads import region_plot, extract_latlon_limits_from_region, region_plot2
from pyresample import geometry
from pyresample import load_area
import yaml
from datetime import datetime
import matplotlib.pyplot as plt

area_def = load_area('areas.yaml', 'region')

dataset_1 = xr.open_dataset('Dataset/GPMGOES-oS2018-03-14T08:59:12-c[8,13]-p256/GPMGOES-oS2018-03-14T08:59:12-c[8,13]-p256-nS2018-03-14T10:15:04.235000064-b3.nc')
dataset_1.close()
print(dataset_1)

area_def = load_area('areas.yaml', 'full_disk')
region_plot2(area_def, [dataset_1],'C08', 256, 'exregionplotC08'+'.pdf')
region_plot2(area_def, [dataset_1],'C13', 256, 'exregionplotC13'+'.pdf')
region_plot2(area_def, [dataset_1],'gpm_precipitation', 256, 'exregionplotp'+'.pdf')


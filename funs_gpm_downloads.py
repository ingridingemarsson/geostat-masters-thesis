import datetime
import re
import pyproj
import numpy as np
import cartopy
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# Extracts date from resulting link in the earth data search
def gpm_extract_datetime(mystr):
	datematch = re.findall(r'(?:\.\d{8})', mystr)[0]
	startmatch = re.findall(r'(?:S\d{6})', mystr)[0]
	endmatch = re.findall(r'(?:E\d{6})', mystr)[0]
	
	start = datetime.datetime.strptime(datematch[1:]+startmatch[1:],"%Y%m%d%H%M%S")
	end = datetime.datetime.strptime(datematch[1:]+endmatch[1:],"%Y%m%d%H%M%S")
	
	if(start.hour > end.hour):
		end += datetime.timedelta(days=1)

	return([start, end])
	
def extract_latlon_limits_from_region(projection, corners):
	lower_left_x, lower_left_y, upper_right_x, upper_right_y = corners
	# Extracting approximate latitude region
	p = pyproj.Proj(projection)
	a_lon, a_lat = p(lower_left_x, lower_left_y, inverse=True)
	b_lon, b_lat = p(upper_right_x, lower_left_y, inverse=True)
	c_lon, c_lat = p(lower_left_x, upper_right_y, inverse=True)
	d_lon, d_lat = p(upper_right_x, upper_right_y, inverse=True)
	
	cS = np.mean([a_lat, b_lat]) # South border
	cN = np.mean([c_lat, d_lat]) # North border
	cW = np.mean([a_lon, c_lon]) # West border
	cE = np.mean([b_lon, d_lon]) # East border
	
	return([cS, cN, cW, cE])

def extract_time_in_brazil(latlon_limits, dataset_gpm):
	cS, cN, cW, cE = latlon_limits

	# Create mask to remove data outside of Brazil rectangle
	lats_mean = np.mean(dataset_gpm["latitude"], axis=1)
	lons_mean = np.mean(dataset_gpm["longitude"], axis=1)
	mask_gpm = (lats_mean > cS)*(lats_mean < cN)*(lons_mean > cW)*(lons_mean < cE)

	# Extract timestamp for arrival and departure in region
	time = dataset_gpm['scan_time']
	in_brazil_t_0 = datetime.datetime.strptime((min(time.values[mask_gpm]).astype(str).split(".")[0]),"%Y-%m-%dT%H:%M:%S")
	in_brazil_t_1 = datetime.datetime.strptime((max(time.values[mask_gpm]).astype(str).split(".")[0]),"%Y-%m-%dT%H:%M:%S")
	return([in_brazil_t_0, in_brazil_t_1])


def region_plot(area_def, dataset, feature):
	crs = area_def.to_cartopy_crs()
	ax = plt.axes(projection=crs)
	ax.coastlines()
	ax.gridlines()
	ax.add_feature(cartopy.feature.BORDERS)
	plt.imshow(dataset[feature], transform=crs, extent=crs.bounds, origin='upper')
	plt.show()
	
	
	

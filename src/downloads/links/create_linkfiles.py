import datetime
import re
import numpy as np

link_file = open("../searchfiles/download20210216.txt", "r") 
link_list = link_file.readlines()
link_file.close()


def gpm_extract_startdate(mystr):
	'''
	Extracting startdate from Earth data search GPM downloading link.
	
	Args:
		mystr: link for GPM product download
		
	Returns:
		startdate
	'''
	datematch = re.findall(r'(?:\.\d{8})', mystr)[0]
	
	startdate = datetime.datetime.strptime(datematch[1:],"%Y%m%d")


	return(startdate)

yearmonth = []
years = []
months = []
for i in range(len(link_list)):
	year_tmp = gpm_extract_startdate(link_list[i]).year
	month_tmp = gpm_extract_startdate(link_list[i]).month
	years.append(year_tmp)
	months.append(month_tmp)
	yearmonth.append(year_tmp*month_tmp)
	
yearmonth_arr = np.array(yearmonth)	
unique_years = np.unique(years)
unique_months = np.unique(months)

filenames = []
directory = 'linkfiles/'
for year in unique_years:
	for month in unique_months:
		inds_tmp = np.where(yearmonth_arr == year*month)
		link_list_uniquemonth = [link_list[ind] for ind in inds_tmp[0]]
		
		if(len(link_list_uniquemonth) > 0):
			filename_tmp = directory + 'linkfile' + str(year) + '-' + str(month).zfill(2) + '.txt'
			filenames.append(filename_tmp)
			with open(filename_tmp, 'w') as f:
				for item in link_list_uniquemonth:
					f.write("%s" % item)
					
					
with open("linkfilenames.txt", 'w') as f:
	for item in filenames:
		f.write("%s\n" % item)


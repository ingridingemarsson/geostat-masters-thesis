import datetime
import re

link_file = open("linkfiles/download20210216.txt", "r") 
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

start_inds = []
month_tmp_old = 0
for i in range(len(link_list)):
	year_tmp = gpm_extract_startdate(link_list[i]).year
	month_tmp_new = gpm_extract_startdate(link_list[i]).month
	
	if(month_tmp_new != month_tmp_old):
		start_inds.append(i)
	
	month_tmp_old = month_tmp_new
	
start_inds.append(len(link_list)+1)

link_list_months = [link_list[start_inds[i]:start_inds[i+1]] for i in range(len(start_inds)-1)]

directory = 'linkfiles/'
for i in range(len(link_list_months)):
	filename_tmp = directory + 'linkfile' + str(i).zfill(2) + str(start_inds[i]).zfill(3) + '.txt'
	with open(filename_tmp, 'w') as f:
		for item in link_list_months[i]:
			f.write("%s" % item)
	print(i)


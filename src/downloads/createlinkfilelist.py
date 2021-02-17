import os

linkdir = "linkfiles/"

linkfilename_list = []
for subdir, dirs, files in os.walk(linkdir):
    for file in files:
        linkfilename_list.append(file)
        
filename_file = 'link' + 'filenames' + '.txt'
with open(filename_file, 'w') as f:
	for item in linkfilename_list:
		f.write("%s\n" % item)


#linkfilename_list = open('linkfilenames.txt', "r") 

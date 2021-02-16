from pyresample import load_area
import yaml
import argparse


def parse_arguments():
	'''
	Parsing arguments from command line for the downloading of
	GPM and GOES data, and passes argument to global variables.
	'''
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"-c",
		"--channels",
		nargs="*",
		help="Choose channels in range 1-16",
		type=int,
		default=[8,13],
	)
	parser.add_argument(
		"-p",
		"--pixels",
		help="Number of pixel height and width of image",
		type=int,
		default=256,
	)
	parser.add_argument(
		"-lf",
		"--linkfile",
		help="Path to Earth data search link file.",
		type=str,
		default="link_file_test.txt",
	)
	parser.add_argument(
		"-s",
		"--storage_path",
		help="Path to folder for final storage of dataset",
		type=str,
		default="Dataset",
	)
	args = parser.parse_args()
	
	global channels
	channels = args.channels
	global number_of_pixels
	number_of_pixels = args.pixels
	global linkfile
	linkfile = args.linkfile
	global path_to_store_processed_data
	path_to_store_processed_data = args.storage_path
	

def initial_load():
	'''
	Loading projection information from file an passing
	to global variables.
	'''
	global area_def 
	global projection 
	global region_corners
	global shape_full_disk
	area_def = load_area('areas.yaml', 'full_disk')

	area_file = open('areas.yaml')
	parsed_area_file = yaml.load(area_file, Loader=yaml.FullLoader)
	area_dict_full_disk = parsed_area_file['full_disk']
	area_dict_region = parsed_area_file['region']
	area_file.close()

	projection = area_dict_full_disk['projection']
	region_corners = area_dict_region['area_extent']
	shape_full_disk = area_dict_full_disk['shape']
	
	


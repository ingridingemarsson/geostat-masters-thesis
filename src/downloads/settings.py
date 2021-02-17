import yaml
import argparse

from pyresample import load_area


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
		default="linkfile464113.txt",
	)
	parser.add_argument(
		"-s",
		"--storage_path",
		help="Path to folder for final storage of dataset",
		type=str,
		default="Dataset",
	)
	parser.add_argument(
	"--plot",
	help="Make plot of box datasets",
	type=bool,
	default=False,
	)
	parser.add_argument(
	"--rem",
	help="Remove used GPM and GOES files",
	type=bool,
	default=False,
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
	global path_to_store_goes_data
	path_to_store_goes_data = "GOES-16"
	global make_box_plot
	make_box_plot = args.plot
	global used_remove
	used_remove = args.rem

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

	
	


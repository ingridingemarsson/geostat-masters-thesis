import numpy as np
import argparse
import os

from load_data import GOESRETRIEVALSDataset, RandomCrop, Mask, Standardize, ToTensor

parser = argparse.ArgumentParser(description='Train fully-connected QRNN')
parser.add_argument(
	"-p",
	"--path_to_data",
	help="Path to data.",
	type=str,
	default="../dataset/data/dataset-singles/"
	)
parser.add_argument(
	"-s",
	"--path_to_storage",
	help="Path to store results.",
	type=str,
	default="../results/"
	)
args = parser.parse_args()

path_to_data = args.path_to_data
path_to_storage = args.path_to_storage


print('hej')

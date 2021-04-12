import numpy as np
import argparse
import os

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


X_train = np.load(os.path.join(path_to_data,'train/X_singles_dataset.npy'))

res = X_train[0]

np.save(os.path.join(path_to_storage, 'test.npy'), res)

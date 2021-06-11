import numpy as np
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from quantnn.qrnn import QRNN
from quantnn.models.pytorch.xception import XceptionFpn

from load_data import GOESRETRIEVALSDataset, Mask, RandomSmallVals, RandomCrop, Standardize, ToTensor


# ARGUMENTS
parser = argparse.ArgumentParser(description='Train fully-connected QRNN')
parser.add_argument(
	"-p",
	"--path_to_data",
	help="Path to data.",
	type=str,
	default="../dataset/data/dataset-test/test/"
	)
parser.add_argument(
	"-st",
	"--path_to_stats",
	help="Path to data.",
	type=str,
	default="../dataset/data/dataset-test/train/stats.npy"
	)
parser.add_argument(
	"-s",
	"--path_to_storage",
	help="Path to store results.",
	type=str,
	default="../results/"
	)
parser.add_argument(
	"-M",
	"--path_to_model",
	help="Path to stored model.",
	type=str,
	default="../results/models/xception64_[100]_0.01__boxes_100_0.01_0_t5412_v1354[0, 1, 2, 3, 4, 5, 6, 7]_Adam_1622288705.386947.pckl"
	)
parser.add_argument(
	"-b",
	"--BATCH_SIZE",
	help="Batch size.",
	type=int,
	default=64
	)
parser.add_argument(
	"--channel_inds", 
	help="Subset of avalible channel indices", 
	nargs="+", 
	type=int,
	default=list(range(0,8)))
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)


BATCH_SIZE = args.BATCH_SIZE

# SETUP
channel_inds = args.channel_inds
num_channels = len(channel_inds)

quantiles = np.linspace(0.01, 0.99, 99)


path_to_test_data = args.path_to_data
path_to_storage = args.path_to_storage
path_to_stats = args.path_to_stats

xception = QRNN.load(args.path_to_model) #xception.pckl')



def importData(BATCH_SIZE, path_to_data, path_to_stats, channel_inds, isTrain=False):

    transforms_list = [Mask(), RandomSmallVals()]
    if isTrain:
        transforms_list.append(RandomCrop(128))
    transforms_list.extend([Standardize(path_to_stats, channel_inds), ToTensor()])

    dataset = GOESRETRIEVALSDataset(
        path_to_data=path_to_data, 
        channel_inds=channel_inds,
        transform=transforms.Compose(transforms_list))
    print('number of samples:', len(dataset))

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    return(dataset, dataloader)



keys=("box", "label")

path_to_test_data_files = os.path.join(path_to_test_data,'npy_files') # os.path.join(path_to_data,'dataset-boxes', 'test', 'npy_files')

test_dataset, test_data = importData(BATCH_SIZE, path_to_test_data_files, path_to_stats, channel_inds)



y_true_tot = []
y_pred_tot = []
#crps_tot = []

with torch.no_grad():
    for batch_index, batch_data in enumerate(test_data):
        print(batch_index)
        
        boxes = batch_data['box'].to(device)
        y_true = batch_data['label']
        
        mask = (torch.less(y_true, 0))
        
        y_pred = xception.posterior_mean(boxes)
        #crps = xception.crps(x=boxes, y_true=y_true)
        
        y_true_tot += [y_true[~mask].detach().cpu().numpy()]
        y_pred_tot += [y_pred[~mask].detach().cpu().numpy()]
        #crps_tot += [crps[~mask].detach().numpy()]

        
y_true_tot_c = np.concatenate(y_true_tot, axis=0)
y_pred_tot_c = np.concatenate(y_pred_tot, axis=0)        

MSE = np.mean(np.square(np.subtract(y_true_tot_c, y_pred_tot_c)))
bias = np.mean(np.subtract(y_true_tot_c, y_pred_tot_c))
MAE = np.mean(np.abs(np.subtract(y_true_tot_c, y_pred_tot_c)))

print(MSE)
print(bias)
print(MAE)

### SETTINGS PLOTS

from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from matplotlib import cm
big = cm.get_cmap('autumn_r', 512)
newcmp = ListedColormap(big(np.linspace(0.2, 0.9, 256)))

import matplotlib
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

SMALL_SIZE = 12 #8
MEDIUM_SIZE = 14 #10
BIGGER_SIZE = 16 #12
matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    
###

def Hist2D(y_true, y_pred, filename):

    norm = Normalize(0, 100)
    bins = np.logspace(-2, 2, 81)
    freqs, _, _ = np.histogram2d(y_true, y_pred, bins=bins)
    
    f, ax = plt.subplots(figsize=(8, 8))

    m = ax.pcolormesh(bins, bins, freqs.T, cmap=newcmp, norm=norm)
    ax.set_xlim([1e-2, 1e2])
    ax.set_ylim([1e-2, 1e2])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reference rain rate [mm / h]")
    ax.set_ylabel("Predicted rain rate [mm / h]")
    ax.plot(bins, bins, c="grey", ls="--")
    ax.set_aspect(1.0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    plt.colorbar(m, cax=cax)

    plt.tight_layout()
    plt.savefig(filename)


Hist2D(y_true_tot_c, y_pred_tot_c, os.path.join(path_to_storage, '2Dhist.png'))


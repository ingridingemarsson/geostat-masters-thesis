import numpy as np
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from quantnn.qrnn import QRNN
from quantnn.models.pytorch.xception import XceptionFpn
import quantnn.quantiles as qq

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
	help="Path to stored models.",
	type=str,
	nargs='+',
	default=["../results/models/xception64_[100]_0.01__boxes_100_0.01_0_t5412_v1354[0, 1, 2, 3, 4, 5, 6, 7]_Adam_1622288705.386947.pckl", "../results/models/singles_fc32786_[100]_0.001__singles_100_0.001_0_t83360758_v20805499[0, 1, 2, 3, 4, 5, 6, 7]_Adam_1622293711.867882.pckl"]
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

xception = QRNN.load(args.path_to_model[0]) 
mlp = QRNN.load(args.path_to_model[1])



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


### SETTINGS PLOTS

from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import LogNorm
big = cm.get_cmap('magma', 512)
newcmp = ListedColormap(big(np.linspace(0.05, 0.95, 256)))

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
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

#colors
color_HE_corr = '#ec6726'
color_HE ='#d1b126'
color_cnn ='#72196d'
color_mlp = '#327a4f'
color_gauges = '#64a6a1'
color_neutral = '#990909'

###

def Hist2D(y_true, y_pred, filename):

    #norm = Normalize(0, 100)
    bins = np.logspace(-4, 3, 80)
    freqs, _, _ = np.histogram2d(y_true, y_pred, bins=bins)
    
    freqs[freqs==0.0] = np.nan
    
    freqs_normed = freqs
    #print( np.nansum(freqs, axis=1))
    for col_ind in range(freqs.shape[0]):
        if np.isnan(freqs[col_ind, :]).all():
            freqs_normed[col_ind, :] = np.array([np.nan] * freqs.shape[1])
        else:
            freqs_normed[col_ind, :] = freqs[col_ind, :] / np.nansum(freqs[col_ind, :])    
            
    freqs = freqs_normed
    
    #freqs = freqs / np.nansum(freqs, axis=0)
    #norm = LogNorm(vmin=np.nanmin(freqs), vmax=np.nanmax(freqs))
    
    f, ax = plt.subplots(figsize=(8, 8))

    m = ax.pcolormesh(bins, bins, freqs.T, cmap=newcmp)#, norm=norm)
    ax.set_xlim([1e-4, 1e3])
    ax.set_ylim([1e-4, 1e3])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reference rain rate [mm / h]")
    ax.set_ylabel("Predicted rain rate [mm / h]")
    ax.plot(bins, bins, c="grey", ls="--")
    ax.grid(True,which="both",ls="--",c='lightgray')  
    #ax.set_aspect(1.0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    plt.colorbar(m, cax=cax)

    plt.tight_layout()
    plt.savefig(filename)
    
    
    
def calibrationPlot(y_true, y_pred, filename):
    cal = np.zeros(len(quantiles))
    N = len(y_true)

    for i in range(len(quantiles)):
        cal[i] = (y_true < y_pred[:,i]).sum() / N
        
    f, ax = plt.subplots(figsize=(8, 8))
    ax.plot(quantiles, cal, color=color_neutral)
    ax.plot(quantiles, quantiles, c="grey", ls="--")
    ax.set_xlabel("True quantiles")
    ax.set_ylabel("Observed quantiles")
    plt.savefig(filename)
    
    
def pdf(y_true, y_b, y_s, q_b, q_s, filename):
    end = np.max([np.max(y_true), np.max(y_b), np.max(y_s)])
    bins = np.linspace(0,end,101)
    f, ax = plt.subplots(figsize=(8, 8))
    ax.hist(y_b, label='CNN posterior mean', bins=bins, histtype='step', color=color_cnn) 
    ax.hist(q_b, label='CNN 95th quantile', bins=bins, histtype='step', color=color_cnn, linestyle='dotted') 
    ax.hist(y_s, label='MLP posterior mean', bins=bins, histtype='step', color=color_mlp) 
    ax.hist(q_s, label='MLP 95th quantile', bins=bins, histtype='step', color=color_mlp, linestyle='dotted') 
    ax.hist(y_true, label='Reference', bins=bins, alpha=0.4, color=color_gauges)
    ax.set_ylabel("Log scaled frequency")
    ax.set_xlabel("Rain rate (mm/h)")
    ax.set_yscale("log")
    ax.legend()
    plt.savefig(filename)
    
    
def diff(y_true, y_b, y_s, filename):
    start = np.min([np.min(y_true-y_b), np.min(y_true-y_s)])
    end = np.max([np.max(y_true-y_b), np.max(y_true-y_s)])
    bins = np.linspace(start,end,101)
    print(bins)
    f, ax = plt.subplots(figsize=(12,8))
    ax.set_yscale("log")
    ax.hist(np.subtract(y_true, y_b), alpha=0.6, bins=bins, color=color_cnn, label='CNN')
    ax.hist(np.subtract(y_true, y_s), alpha=1, bins=bins, color=color_mlp, label='MLP', histtype='step')
    ax.set_ylabel('Logged counts')
    ax.set_xlabel('Rain difference (mm)')
    ax.axvline(x=0.0, color='grey', alpha=0.5, linestyle='dashed')
    ax.grid(True,which="both",ls="--",c='lightgray')  
    ax.legend()
    plt.savefig(filename)
    
###

def pred(model_boxes, model_singles):

    num = 44869385
    y_true_tot = np.empty(num)
    y_pred_boxes_tot = np.empty((num,len(quantiles)))
    y_pred_singles_tot = np.empty((num,len(quantiles)))

    i=0
    with torch.no_grad():
        for batch_index, batch_data in enumerate(test_data):
            print(batch_index)

            boxes = batch_data['box'].to(device)
            y_true = batch_data['label']

            mask = (torch.less(y_true, 0))
            
            #Boxes
            y_pred_boxes = model_boxes.predict(boxes).detach().cpu().numpy()
            y_pred_boxes = np.concatenate([y_pred_boxes[i, :, mask[i].detach().cpu().numpy()==0] 
                                            for i in range(y_pred_boxes.shape[0])], axis=0)

            increase = len(y_true[~mask].detach().cpu().numpy())
            y_true_tot[i:i+increase] = y_true[~mask].detach().cpu().numpy()
            y_pred_boxes_tot[i:i+increase, :] = y_pred_boxes
            
            #Singles
            boxes = torch.transpose(torch.flatten(torch.transpose(boxes, 0, 1), start_dim=1), 0, 1)
            mask = torch.flatten(mask)

            y_pred_singles = model_singles.predict(boxes)
            y_pred_singles_tot[i:i+increase, :] = y_pred_singles[~mask].detach().cpu().numpy()
            i+=increase

    #print('concatenate')
    #y_true_tot_c = np.concatenate(y_true_tot, axis=0)
    #del y_true_tot
    #y_pred_boxes_tot_c = np.concatenate(y_pred_boxes_tot, axis=0)
    #del y_pred_boxes_tot
    #y_pred_singles_tot_c = np.concatenate(y_pred_singles_tot, axis=0)
    #del y_pred_singles_tot

    #return(y_true_tot_c, y_pred_boxes_tot_c, y_pred_singles_tot_c)
    return(y_true_tot, y_pred_boxes_tot, y_pred_singles_tot)

def applyTreshold(y, th):
    y[y<th] = 0.0
    return(y)
    
def computeMetrics(y_true, y_pred, name):
    calibrationPlot(y_true, y_pred, os.path.join(path_to_storage, 'calibration'+name+'.png'))
    loss = qq.quantile_loss(y_pred, quantiles, y_true, quantile_axis=1)
    print('loss mean', loss.mean())
    crps = qq.crps(y_pred, quantiles, y_true, quantile_axis=1)
    print('crps mean', crps.mean())


def computeMeanMetrics(y_true, y_mean, name):
    Hist2D(y_true, y_mean, os.path.join(path_to_storage, '2Dhist_'+name+'.png'))
    bias = np.mean(np.subtract(y_true, y_mean))
    print('bias', bias)
    mae = np.mean(np.abs(np.subtract(y_true, y_mean)))
    print('MAE', mae)
    mse = np.mean(np.square(np.subtract(y_true, y_mean)))
    print('MSE', mse)
    
    


# COMPUTE
y_true, y_boxes, y_singles = pred(xception, mlp)

#Boxes
computeMetrics(y_true, y_boxes, 'boxes')
y_mean_boxes = qq.posterior_mean(y_boxes, quantiles, quantile_axis=1)
q95_boxes = y_boxes[:,94]
#q99_boxes = y_boxes[:,98]
del y_boxes

#Singles
computeMetrics(y_true, y_singles, 'singles')
y_mean_singles = qq.posterior_mean(y_singles, quantiles, quantile_axis=1)
q95_singles = y_singles[:,94]
#q99_singles = y_singles[:,98]
del y_singles

#Mean
y_true = applyTreshold(y_true, 1e-2)
y_mean_boxes = applyTreshold(y_mean_boxes, 1e-2)
y_mean_singles = applyTreshold(y_mean_singles, 1e-2)
computeMeanMetrics(y_true, y_mean_boxes, 'boxes')
computeMeanMetrics(y_true, y_mean_singles, 'singles')

#Common
pdf(y_true, y_mean_boxes, y_mean_singles, q95_boxes, q95_singles, os.path.join(path_to_storage, 'pdf.png'))
diff(y_true, y_mean_boxes, y_mean_singles, os.path.join(path_to_storage, 'diff.png'))

print('done')





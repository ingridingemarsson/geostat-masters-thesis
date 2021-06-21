import numpy as np
import os
import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib.colors import LogNorm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import csv

from quantnn.qrnn import QRNN
from quantnn.models.pytorch.xception import XceptionFpn
import quantnn.quantiles as qq

from load_data import GOESRETRIEVALSDataset, Mask, RandomSmallVals, RandomCrop, Standardize, ToTensor


# ARGUMENTS ####################################################################################################################
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

path_to_test_data = args.path_to_data
path_to_storage = args.path_to_storage
path_to_stats = args.path_to_stats

xception = QRNN.load(args.path_to_model[0]) 
mlp = QRNN.load(args.path_to_model[1])

channel_inds = args.channel_inds
num_channels = len(channel_inds)
path_to_test_data_files = os.path.join(path_to_test_data,'npy_files') 

# CONSTANTS ####################################################################################################################

keys=("box", "label")
quantiles = np.linspace(0.01, 0.99, 99)
threshold_val = 1e-1

#plot settings
big = cm.get_cmap('magma', 512)
newcmp = ListedColormap(big(np.linspace(0.05, 0.95, 256)))
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
color_cnn ='#72196d'
color_mlp = '#327a4f'
color_neutral = '#990909'
color_grid = "#e9e9e9"
color_reference = '#64a6a1'

#alpha
alpha_cnn_hist = 0.4
alpha_reference_hist = 0.3

# FUNCTIONS ####################################################################################################################

#Data
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

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, 
                            shuffle=False, #Try this to run for boxes and singles separately
                            num_workers=1)
    return(dataset, dataloader)


#Predict
def pred(model, mod_type, filename):

    y_true_tot = []
    y_mean_tot = []
    y_q95_tot = []
    cal = np.zeros(len(quantiles))
    loss = []
    crps = []

    with torch.no_grad():
        for batch_index, batch_data in enumerate(test_data):
            print(batch_index)

            boxes = batch_data['box'].to(device)
            y_true = batch_data['label']
            
            mask = (torch.less(y_true, 0))
            
            y_true = y_true[~mask].detach().cpu().numpy()
       
            if mod_type=='boxes':
                y_pred_boxes = model.predict(boxes).detach().cpu().numpy()
                y_pred = np.concatenate([y_pred_boxes[i, :, mask[i].detach().cpu().numpy()==0] 
                                                for i in range(y_pred_boxes.shape[0])], axis=0)

            elif mod_type=='singles':
                boxes = torch.transpose(torch.flatten(torch.transpose(boxes, 0, 1), start_dim=1), 0, 1)
                mask = torch.flatten(mask)

                y_pred_singles = model.predict(boxes)
                y_pred = y_pred_singles[~mask].detach().cpu().numpy()
                
            y_true_tot += [y_true]

            #Metrics
            for i in range(len(quantiles)):
                cal[i] += np.sum(y_true < y_pred[:,i])          
                
            loss += [qq.quantile_loss(y_pred, quantiles, y_true, quantile_axis=1).mean(axis=1)]
            crps += [qq.crps(y_pred, quantiles, y_true, quantile_axis=1)]
            
            y_mean_tot += [qq.posterior_mean(y_pred, quantiles, quantile_axis=1)]
            y_q95_tot += [y_pred[:,94]]
                         
    y_true_tot_c = np.concatenate(y_true_tot, axis=0)
    y_mean_tot_c = np.concatenate(y_mean_tot, axis=0)
    y_q95_tot_c = np.concatenate(y_q95_tot, axis=0)
    loss_c = np.concatenate(loss, axis=0)
    crps_c = np.concatenate(crps, axis=0)
    
    header = ['loss mean', 'crps mean', 'crps median']
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        # write the header
        writer.writerow(header)
        
        # write multiple rows
        writer.writerows([[loss_c.mean(),  crps_c.mean(),  np.median(crps_c)]])      
    
    return(y_true_tot_c, y_mean_tot_c, y_q95_tot_c, cal/len(y_true_tot_c))


#Visualization
def calibrationPlot(cal, filename):
        
    f, ax = plt.subplots(figsize=(8, 8))
    ax.plot(quantiles, cal, color=color_neutral)
    ax.plot(quantiles, quantiles, c="grey", ls="--")
    ax.grid(True,which="both",ls="--",c=color_grid)
    ax.set_xlabel("True quantiles")
    ax.set_ylabel("Observed quantiles")
    plt.savefig(filename)

    
def Hist2D(y_true, y_pred, filename, norm_type=None):
    bins = np.logspace(-1, 2, 80)# 81)
    
    #print('max y_true', np.max(y_true))
    
    freqs, _, _ = np.histogram2d(y_true, y_pred, bins=bins)
    freqs[freqs==0.0] = np.nan
    vmax = None
    extend = 'neither'

    if norm_type==None:
        freqs_normed = freqs
    elif norm_type=='colwise':
        freqs_normed = freqs
        for col_ind in range(freqs.shape[0]):
            if np.isnan(freqs[col_ind, :]).all():
                freqs_normed[col_ind, :] = np.array([np.nan] * freqs.shape[1])
            else:
                freqs_normed[col_ind, :] = freqs[col_ind, :] / np.nansum(freqs[col_ind, :])
        vmax=np.percentile(freqs_normed[np.isnan(freqs_normed)==False], 99)
        extend = 'max'
        #print(vmax)

    f, ax = plt.subplots(figsize=(8,8))
    m = ax.pcolormesh(bins, bins, freqs_normed.T, cmap=newcmp, vmax=vmax)
    ax.set_xlim([1e-1, 1e2])
    ax.set_ylim([1e-1, 1e2])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Reference rain rate (mm/h)")
    ax.set_ylabel("Predicted rain rate (mm/h)")
    ax.plot(bins, bins, c="grey", ls="--")
    ax.grid(True,which="both",ls="--",c=color_grid)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.07)
    plt.colorbar(m, cax=cax, extend=extend)
    
    plt.tight_layout()
    plt.savefig(filename)
    
    
def pdf(y_true, y_b, y_s, q_b, q_s, filename):
    end = np.max([np.max(y_true), np.max(y_b), np.max(y_s)])
    bins = np.linspace(0,end,100)
    f, ax = plt.subplots(figsize=(12, 8))
    ax.hist(y_b, label='CNN posterior mean', bins=bins, histtype='step', color=color_cnn) 
    ax.hist(q_b, label='CNN 95th quantile', bins=bins, histtype='step', color=color_cnn, linestyle='dotted') 
    ax.hist(y_s, label='MLP posterior mean', bins=bins, histtype='step', color=color_mlp) 
    ax.hist(q_s, label='MLP 95th quantile', bins=bins, histtype='step', color=color_mlp, linestyle='dotted') 
    ax.hist(y_true, label='Reference', bins=bins, alpha=alpha_reference_hist, color=color_reference)
    ax.set_ylabel("Frequency")
    ax.set_xlabel("Rain rate (mm/h)")
    ax.set_yscale("log")
    ax.grid(True,which="both",ls="--",c=color_grid)  
    ax.legend()
    plt.savefig(filename)
    
    
def diff(y_true, y_b, y_s, filename):
    start = np.min([np.min(y_true-y_b), np.min(y_true-y_s)])
    end = np.max([np.max(y_true-y_b), np.max(y_true-y_s)])
    bins = np.linspace(start,end,100)
    f, ax = plt.subplots(figsize=(12,8))
    ax.hist(np.subtract(y_true, y_b), alpha=alpha_cnn_hist, bins=bins, color=color_cnn, label='CNN')
    ax.hist(np.subtract(y_true, y_s), bins=bins, color=color_mlp, label='MLP', histtype='step')
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Difference rain rate (mm/h)')
    ax.axvline(x=0.0, color='grey', alpha=0.5, linestyle='dashed')
    ax.set_yscale("log")
    ax.grid(True,which="both",ls="--",c=color_grid)  
    ax.legend()
    plt.savefig(filename)
    
    
#def applyThreshold(y, th):
#    y[y<th] = 0.0
#    return(y)
  

def computeMeanMetrics(y_true, y_mean):
    bias = np.mean(np.subtract(y_true, y_mean))
    mae = np.mean(np.abs(np.subtract(y_true, y_mean)))
    mse = np.mean(np.square(np.subtract(y_true, y_mean)))
    
    return([bias, mae, mse])
    
    
def computeMeanMetricsIntervals(y_true, y_pred, filename):
    
    intervals = [0.0, 1e-2, 1e0, 1e1, 1e3]
    metrics = []
    
    metrics_row = []
    metrics_row.append(0.0)
    metrics_row.append(1e3)
    metrics_row.append(len(y_true))
    metrics_row.extend(computeMeanMetrics(y_true, y_pred))
    metrics.append(metrics_row)    
    
    for i in range(len(intervals)-1):
        interval_mask = (y_true >= intervals[i]) & (y_true < intervals[i+1])
        metrics_row = []
        metrics_row.append(intervals[i])
        metrics_row.append(intervals[i+1])
        
        metrics_row.append(len(y_true[interval_mask])/len(y_true))
        metrics_row.extend(computeMeanMetrics(y_true[interval_mask], y_pred[interval_mask]))
        metrics.append(metrics_row)          

    header = ['start', 'end', 'fraction', 'bias', 'mae', 'mse']
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        # write the header
        writer.writerow(header)
        
        # write multiple rows
        writer.writerows(metrics)    
    
    
def Classification(y, p, threshold, filename):

    TP = (p[y>threshold]>threshold).sum() #Is rain, predict rain
    TN = (p[y<=threshold]<=threshold).sum() #Is no rain, predict no rain
    FP = (p[y<=threshold]>threshold).sum() #Is no rain, predict rain
    FN = (p[y>threshold]<=threshold).sum() #Is rain, predict no rain
    
    #print('TP, Is rain, predict rain:', TP)
    #print('TN, Is no rain, predict no rain:', TN)
    #print('FP, Is no rain, predict rain:', FP)
    #print('FN, Is rain, predict no rain:', FN)
    
    FPR = FP/(FP+TN)
    FNR = FN/(FN+TP)
    
    header = ['TP', 'TN', 'FP', 'FN', 'FPR', 'FNR']
    with open(filename, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        
        # write the header
        writer.writerow(header)
        
        # write multiple rows
        writer.writerows([[TP, TN, FP, FN, FPR, FNR]]) 
    
    
def FalsePlots(y, p, r, threshold, filenames):
    bins=np.linspace(0,100,100)
    pp = p[y<=threshold]
    rr = r[y<=threshold]
    fig, ax = plt.subplots(figsize=(12,8))
    ax.hist(pp[pp>threshold], bins=bins, alpha=alpha_cnn_hist, color=color_cnn, label='CNN')
    ax.hist(rr[rr>threshold], bins=bins, color=color_mlp, histtype='step', label='MLP')
    ax.set_yscale("log")
    ax.grid(True,which="both",ls="--",c=color_grid) 
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Predicted rain rate (mm/h)')
    #ax.set_title('Distribution of non-zero predicted rain corresponding to no rain reference values')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filenames[0])

    yp = y[p<=threshold]
    yr = y[r<=threshold]
    fig, ax = plt.subplots(figsize=(12,8))
    ax.hist(yp[yp>threshold], bins=bins, alpha=alpha_cnn_hist, color=color_cnn, label='CNN')
    ax.hist(yr[yr>threshold], bins=bins, color=color_mlp, histtype='step', label='MLP')
    ax.set_yscale("log")
    ax.grid(True,which="both",ls="--",c=color_grid) 
    #ax.set_ylim([5e-1, 1e4])
    ax.set_ylabel('Frequency')
    ax.set_xlabel('Reference rain rate (mm/h)')
    #ax.set_title('Distribution of non-zero reference values corresponding to no rain predictions')
    ax.legend()
    plt.tight_layout()
    plt.savefig(filenames[1])    


    
# COMPUTATION ##################################################################################################################

#Dataset
test_dataset, test_data = importData(BATCH_SIZE, path_to_test_data_files, path_to_stats, channel_inds)

#Boxes
y_true, y_boxes, y_boxes_q95, cal = pred(xception, 'boxes', os.path.join(path_to_storage,'pred_metrics_boxes.csv'))
calibrationPlot(cal, 'calibration_boxes.png')

#Singles
y_true_s, y_singles, y_singles_q95, cal = pred(mlp, 'singles', os.path.join(path_to_storage,'pred_metrics_singles.csv'))
calibrationPlot(cal, 'calibration_singles.png')

same = (y_true == y_true_s).all()
assert same, "True values differ"
del y_true_s

#Threshold
#y_true = applyThreshold(y_true, threshold_val)
#y_boxes = applyThreshold(y_boxes, threshold_val)
#y_singles = applyThreshold(y_singles, threshold_val)

#Classification
Classification(y_true, y_boxes, threshold_val, os.path.join(path_to_storage,'classification_boxes.csv'))
Classification(y_true, y_singles, threshold_val, os.path.join(path_to_storage,'classification_singles.csv'))

#Hist
Hist2D(y_true, y_boxes, os.path.join(path_to_storage, '2Dhist_boxes_colwise.png'), norm_type='colwise')
Hist2D(y_true, y_singles, os.path.join(path_to_storage, '2Dhist_singles_colwise.png'),  norm_type='colwise')
Hist2D(y_true, y_boxes, os.path.join(path_to_storage, '2Dhist_boxes.png'))
Hist2D(y_true, y_singles, os.path.join(path_to_storage, '2Dhist_singles.png'))

#Common
pdf(y_true, y_boxes, y_singles, y_boxes_q95, y_singles_q95, os.path.join(path_to_storage, 'pdf.png'))
diff(y_true, y_boxes, y_singles, os.path.join(path_to_storage, 'diff.png'))
FalsePlots(y_true, y_boxes, y_singles, threshold_val, [os.path.join(path_to_storage, 'FalsePositives.png'),  os.path.join(path_to_storage, 'FalseNegatives.png')])

#Scalar metrics
computeMeanMetricsIntervals(y_true, y_boxes, os.path.join(path_to_storage,'metrics_boxes_intervals.csv'))
computeMeanMetricsIntervals(y_true, y_singles, os.path.join(path_to_storage,'metrics_singles_intervals.csv'))

print('done')





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

import plotTestSetup as setup
from plotTest import plotFalse, plotError, plotDistribution, hist2D
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
plot_type = '.pdf'

#plot settings
big = cm.get_cmap('magma', 512)
newcmp = ListedColormap(big(np.linspace(0.1, 0.9, 256)))
plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

SMALL_SIZE = 16 #8
MEDIUM_SIZE = 18 #10
BIGGER_SIZE = 20 #12
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
        
    f, ax = plt.subplots(figsize=(6,6))
    ax.plot(quantiles, cal, color=color_neutral)
    ax.plot(quantiles, quantiles, c="grey", ls="--")
    ax.grid(True,which="both",ls="--",c=color_grid)
    ax.set_xlabel("True quantiles")
    ax.set_ylabel("Observed quantiles")
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    

def computeMeanMetrics(y_true, y_mean):
    bias = np.mean(np.subtract(y_mean, y_true))
    mae = np.mean(np.abs(np.subtract(y_mean, y_true)))
    mse = np.mean(np.square(np.subtract(y_mean, y_true)))
    
    return([bias, mae, mse])
    
    
def computeMeanMetricsIntervals(y_true, y_pred, filename):
    
    intervals = [0.0, 1e-1, 1e0, 1e1, 1e3]
    metrics = []
    
    metrics_row = []
    metrics_row.append(0.0)
    metrics_row.append(1e3)
    metrics_row.append(1.0)
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
    
   
    
# COMPUTATION ##################################################################################################################

#Dataset
test_dataset, test_data = importData(BATCH_SIZE, path_to_test_data_files, path_to_stats, channel_inds)

#Boxes
y_true, y_boxes, y_boxes_q95, cal = pred(xception, 'boxes', os.path.join(path_to_storage,'pred_metrics_boxes.csv'))
calibrationPlot(cal, os.path.join(path_to_storage, 'calibration_boxes'+plot_type))

#Singles
y_true_s, y_singles, y_singles_q95, cal = pred(mlp, 'singles', os.path.join(path_to_storage,'pred_metrics_singles.csv'))
calibrationPlot(cal, os.path.join(path_to_storage,'calibration_singles'+plot_type))

same = (y_true == y_true_s).all()
assert same, "True values differ"
del y_true_s

quantity = 'precipitation rate (mm)'
data_dict = {}
data_dict['gpm'] = y_true
data_dict['xception_posterior_mean'] = y_boxes
data_dict['mlp_posterior_mean'] = y_singles
data_dict['mlp_Q0.95'] = y_singles_q95
data_dict['xception_Q0.95'] = y_boxes_q95


start = 0.0
end = np.around(np.max(y_true),1)
binsize = 0.1
num_of_bins = int(np.round((end-start)/binsize)+1)
bins = np.linspace(start,end,num_of_bins)

var_list = ['mlp_posterior_mean', 'xception_posterior_mean', 'mlp_Q0.95', 'xception_Q0.95']
plotDistribution(data_dict, bins, 'gpm', var_list, quantity=quantity,  linestyles=['solid', 'solid', 'dotted', 'dotted'], filename=os.path.join(path_to_storage,'gpm_pdf.pdf'))

var_list = ['xception_posterior_mean', 'mlp_posterior_mean']
start = 0.0
end = 100.0
binsize = 0.1
num_of_bins = int(np.round((end-start)/binsize)+1)
bins = np.linspace(start,end,num_of_bins)
plotFalse(data_dict, bins, 'gpm', var_list, ty='FP', threshold=1e-1, crop_at=10.1, filename=os.path.join(path_to_storage,'gpm_FP.pdf'), quantity=quantity)
plotFalse(data_dict, bins, 'gpm', var_list, ty='FN', threshold=1e-1, crop_at=10.1, filename=os.path.join(path_to_storage,'gpm_FN.pdf'), quantity=quantity)


start = -250.0
end = 60.0
binsize = 0.1
num_of_bins = int(np.round((end-start)/binsize)+1)
bins = np.linspace(start,end,num_of_bins)
plotError(data_dict, bins, 'gpm', var_list, quantity=quantity, filename=os.path.join(path_to_storage,'gpm_diff.pdf'))

hist2D(data_dict, 'gpm', var_list, norm_type=None, quantity=quantity, filename=os.path.join(path_to_storage,'gpm_2Dhist.pdf'))
hist2D(data_dict, 'gpm', var_list, norm_type='colwise', quantity=quantity, filename=os.path.join(path_to_storage,'gpm_2Dhist_colwise.pdf'))


#Scalar metrics
computeMeanMetricsIntervals(y_true, y_boxes, os.path.join(path_to_storage,'metrics_boxes_intervals.csv'))
computeMeanMetricsIntervals(y_true, y_singles, os.path.join(path_to_storage,'metrics_singles_intervals.csv'))

print('done')





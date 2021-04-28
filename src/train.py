import numpy as np
import os
from datetime import datetime
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from matplotlib.colors import Normalize
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
import warnings
warnings.filterwarnings("ignore")

from quantnn.qrnn import QRNN
from quantnn.models.pytorch.logging import TensorBoardLogger
from quantnn.metrics import ScatterPlot

from load_data import GOESRETRIEVALSDataset, Mask, RandomSmallVals, TakeLog, RandomCrop, Standardize, ToTensor

stamp = str(datetime.today().timestamp())

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

# ARGUMENTS
parser = argparse.ArgumentParser(description='Train fully-connected QRNN')
parser.add_argument(
	"-p",
	"--path_to_data",
	help="Path to data.",
	type=str,
	nargs='+',
	default=["../dataset/data/dataset-test/train/", "../dataset/data/dataset-test/validation/"]
	)
parser.add_argument(
	"-s",
	"--path_to_storage",
	help="Path to store results.",
	type=str,
	default="../results/"
	)
parser.add_argument(
	"-b",
	"--BATCH_SIZE",
	help="Batch size.",
	type=int,
	default=4
	)
parser.add_argument(
	"-l",
	"--log",
	help="Apply log transform to label data.",
	type=bool,
	default=False
	)
parser.add_argument(
	"-m",
	"--model",
	help="Specify which model to use",
	type=str,
	default='xception'
	)
parser.add_argument(
	"-e",
	"--epochs_list", 
	help="List of number of epochs",
	nargs="+", 
	type=int,
	default=[10])
parser.add_argument(
	"-lr",
	"--lr", 
	help="Learning rate",
	type=float,
	default=0.001)
parser.add_argument(
	"-D",
	"--data_type", 
	help="Type of dataset, boxes or singles", 
	type=str,
	default="boxes")
parser.add_argument(
	"-O",
	"--optimizer", 
	help="Optimizer", 
	type=str,
	default="Adam")
args = parser.parse_args()

BATCH_SIZE = args.BATCH_SIZE
apply_log = args.log
print(apply_log)
n_epochs_arr = args.epochs_list
lr = args.lr
data_type = args.data_type
net_name = args.model
optim = args.optimizer 

# SETUP
channels = list(range(8,17))
channels.remove(12)
fillvalue = -1
quantiles = np.linspace(0.01, 0.99, 99)

filename = (net_name + str(apply_log) + str(BATCH_SIZE) + '_' + str(n_epochs_arr) + '_' + str(lr) + '_' + '_' + data_type).replace(" ", "")

path_to_training_data = args.path_to_data[0]
path_to_validation_data = args.path_to_data[1]

path_to_storage = args.path_to_storage

global path_to_save_model
path_to_save_model = os.path.join(path_to_storage, filename, 'saved_models')
if not Path(path_to_save_model).exists():
	os.makedirs(path_to_save_model)
	
global log_directory
log_directory = os.path.join(path_to_storage, filename, 'runs', str(n_epochs_arr)+'_'+str(lr), stamp)
if not Path(log_directory).exists():
	os.makedirs(log_directory)


if (data_type == "singles"):

	if (net_name == 'singles_fc'):
		from models.singles_fc import Net
		net = Net(len(quantiles), len(channels))
	elif (net_name == 'singles_fc2'):
		from models.singles_fc2 import Net
		net = Net(len(quantiles), len(channels))
	elif (net_name == 'singles_fc3'):
		from models.singles_fc3 import Net
		net = Net(len(quantiles), len(channels))
	elif (net_name == 'singles_fc4'):
		from models.singles_fc4 import Net
		net = Net(len(quantiles), len(channels))	

	from quantnn.models.pytorch import BatchedDataset

	keys = None
	
	X_train = np.load(os.path.join(path_to_training_data, 'X_singles_dataset.npy'))
	y_train = np.load(os.path.join(path_to_training_data, 'y_singles_dataset.npy'))
	X_val = np.load(os.path.join(path_to_validation_data, 'X_singles_dataset.npy'))
	y_val = np.load(os.path.join(path_to_validation_data, 'y_singles_dataset.npy'))	
	
		
	#subs = 100000
	#X_train = X_train[:subs].astype(np.float32)
	#y_train = y_train[:subs].astype(np.float32)
	#X_val = X_val[:subs].astype(np.float32)
	#y_val = y_val[:subs].astype(np.float32)
	#print('size of training data: ', X_train.shape)
	
	
	def Standardize(X, path_to_training_data):
		stats = np.load(os.path.join(path_to_training_data, 'X_singles_stats.npy'))
		return ((X-stats[0,:])/stats[1,:]).astype(np.float32)
	
	def ZeroToRand(y):
		inds = np.where(y==0.0)
		print(len(y[inds]))
		y[inds] = np.random.uniform(1e-4, 1e-3, len(y[inds]))

		return y
	
	
	X_train = Standardize(X_train, path_to_training_data)
	X_val = Standardize(X_val, path_to_training_data)
	
	y_train = ZeroToRand(y_train)
	y_val = ZeroToRand(y_val)
	
	training_data = BatchedDataset((X_train, y_train), BATCH_SIZE)
	validation_data = BatchedDataset((X_val, y_val), BATCH_SIZE)
	
	logger = TensorBoardLogger(np.sum(n_epochs_arr), log_directory=log_directory)
	dat_size = str(len(y_train))+'_v'+str(len(y_val))
		
elif (data_type == "boxes"):
	print(quantiles.size)
	if (net_name == 'xception'):
		from quantnn.models.pytorch.xception import XceptionFpn
		net = XceptionFpn(len(channels), quantiles.size, n_features=128)
	elif (net_name == 'boxes_one'):
		from models.boxes_one import Net
		net = Net(quantiles.size, len(channels))
	
	keys=("box", "label")
	
	path_to_train_data_files = os.path.join(path_to_training_data, 'npy_files')
	path_to_stats = os.path.join(path_to_training_data, 'stats.npy')
	path_to_val_data_files = os.path.join(path_to_validation_data, 'npy_files')

	def importData(channels, BATCH_SIZE, path_to_data, path_to_stats, apply_log=False):
		transforms_list = [Mask(), RandomSmallVals()]
		if apply_log:
			transforms_list.append(TakeLog())
		transforms_list.extend([RandomCrop(128), Standardize(path_to_data, path_to_stats, channels), ToTensor()])
		dataset = GOESRETRIEVALSDataset(
			path_to_data = path_to_data,
			channels = channels, 
			transform = transforms.Compose(transforms_list))
		print('number of samples:', len(dataset))

		dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
		return(dataset, dataloader)

	training_dataset, training_data = importData(channels, BATCH_SIZE, path_to_train_data_files, path_to_stats, apply_log=apply_log)
	validation_dataset, validation_data  = importData(channels, BATCH_SIZE, path_to_val_data_files, path_to_stats, apply_log=apply_log)

	def make_prediction(writer, model, epoch_index):
	    """
	    Predicts the mean precipitation rate on a random sample
	    from the validation set.
	    
	    Args:
		writer: The SummaryWriter object that is used to log
		     to the tensbor board.
		model: The model attributed of the qrnn object being
		    trained.
		epoch_index: The index (zero-based) of the current
		    epoch.
	    """
	    
	    precip_norm = LogNorm(1e-2, 1e2)
	    
	    # Make prediction
	    y_mean = torch.squeeze(qrnn.posterior_mean(x)).cpu().detach().numpy()
	    
	    # Store output using add_figure function of SummaryWriter
	    fig_pred = plt.figure()
	    gs = GridSpec(2, 1, figure=fig_pred, height_ratios=[1.0, 0.1])
	    ax = plt.subplot(gs[0, 0])
	    m = ax.imshow(y_mean, norm=precip_norm, cmap=plt.get_cmap('GnBu'))
	    ax.imshow(y.squeeze(0)!=-1,  cmap=plt.get_cmap('binary_r'), alpha = 0.02)
	    ax.grid(False)
	    ax.set_title("(d) Model prediction rain rate", loc="left")
	    ax = plt.subplot(gs[1, 0])
	    plt.colorbar(m, cax=ax, orientation="horizontal", label=r"Rain rate [mm/h]")    
	    plt.tight_layout()
	    writer.add_figure("predicted_rain_rate", fig_pred, epoch_index)
	    
	    # Store reference rain using add_figure function of
	    # SummaryWriter. No need to store for every epoch.
	    fig_true = plt.figure()
	    gs = GridSpec(2, 1, figure=fig_true, height_ratios=[1.0, 0.1])
	    ax = plt.subplot(gs[0, 0])
	    m = ax.imshow(y.squeeze(0), norm=precip_norm, cmap=plt.get_cmap('GnBu'))
	    ax.imshow(y.squeeze(0)!=-1, cmap=plt.get_cmap('binary_r'), alpha = 0.02)
	    ax.grid(False)
	    ax.set_title("(d) Model prediction rain rate", loc="left")
	    ax = plt.subplot(gs[1, 0])
	    plt.colorbar(m, cax=ax, orientation="horizontal", label=r"Rain rate [mm/h]")    
	    plt.tight_layout()
	    writer.add_figure("reference_rain_rate", fig_true, 0)



	index = 0 #np.random.randint(len(validation_dataset))
	x = validation_dataset[index]['box'].unsqueeze(0).to(device)
	y = validation_dataset[index]['label']
	
	logger = TensorBoardLogger(np.sum(n_epochs_arr), log_directory=log_directory, epoch_begin_callback=make_prediction)
	#logger = TensorBoardLogger(np.sum(n_epochs_arr), log_directory=log_directory)
	dat_size = str(len(training_dataset))+'_v'+str(len(validation_dataset))

# TRAIN MODEL
qrnn = QRNN(quantiles=quantiles, model=net)
metrics = ["MeanSquaredError", "Bias", "CRPS", "CalibrationPlot"]
scatter_plot = ScatterPlot(bins=np.logspace(-2, 2, 100), log_scale=True)
metrics.append(scatter_plot)

logger.set_attributes({"optimizer": optim, "n_epochs": str(n_epochs_arr), "learning_rate": lr}) 
if (optim=="Adam"):
	optimizer = Adam(qrnn.model.parameters(), lr=lr)
elif (optim=="SGD"):
	optimizer = SGD(qrnn.model.parameters(), lr=lr, momentum=0.9)

for i in range(len(n_epochs_arr)):
	qrnn.train(training_data=training_data,
		      validation_data=validation_data,
		      keys=keys,
		      n_epochs=n_epochs_arr[i],
		      optimizer=optimizer,
		      scheduler = StepLR(optimizer, step_size=100, gamma=0.1),
		      mask=fillvalue,
		      device=device,
		      metrics=metrics,
		      logger=logger);
	filename_tmp = filename+'_'+str(n_epochs_arr[i])+'_'+str(lr)+'_'+str(i)+'_t'+dat_size+'_'+optim+'_'+stamp
	qrnn.save(os.path.join(path_to_save_model, filename_tmp+'.pckl'))


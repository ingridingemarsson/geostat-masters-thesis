import numpy as np
import os
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
from torch.optim.lr_scheduler import CosineAnnealingLR

from quantnn.qrnn import QRNN
from quantnn.models.pytorch.logging import TensorBoardLogger

from load_data import GOESRETRIEVALSDataset, Mask, RandomLog, RandomCrop, Standardize, ToTensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

# ARGUMENTS
parser = argparse.ArgumentParser(description='Train fully-connected QRNN')
parser.add_argument(
	"-p",
	"--path_to_data",
	help="Path to data.",
	type=str,
	default="../dataset/data/dataset-test/"
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
	default=128
	)
parser.add_argument(
	"-f",
	"--filename_ext",
	help="Naming of files, extra.",
	type=str,
	default=''
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
	default='boxes_one'
	)
parser.add_argument(
	"-e",
	"--epochs_list", 
	help="List of number of epochs",
	nargs="+", 
	type=int,
	default=[2, 3, 4])
parser.add_argument(
	"-lr",
	"--lr_list", 
	help="List of learning rates",
	nargs="+", 
	type=float,
	default=[0.01, 0.01, 0.001])
parser.add_argument(
	"-o",
	"--optimizer", 
	help="Optimizer", 
	type=str,
	default="Adam")
args = parser.parse_args()

BATCH_SIZE = args.BATCH_SIZE
filename_extension = args.filename_ext
apply_log = args.log
print(apply_log)
n_epochs_arr = args.epochs_list
lrs = args.lr_list
if (len(n_epochs_arr) != len(lrs)):
	raise ValueError("list epochs_list and list lr_list must have the same length")
	
opt = args.optimizer

# SETUP
channels = list(range(8,17))
channels.remove(12)
fillvalue = -1
quantiles = np.linspace(0.01, 0.99, 99)

net_name = args.model
if (net_name == 'boxes_one'):
	from models.boxes_one import Net
	net = Net(len(quantiles), len(channels))
elif (net_name == 'boxes_one_modified'):
	from models.boxes_one_modified import Net
	net = Net(len(quantiles), len(channels))
elif (net_name == 'xception'):
	from quantnn.models.pytorch.xception import XceptionFpn
	net = XceptionFpn(len(channels), quantiles.size, n_features=128)
elif (net_name == 'ResNet'):
	from quantnn.models.pytorch.resnet import ResNet
	net = ResNet(len(channels), quantiles.size)
elif (net_name == 'UNet'):
	from quantnn.models.pytorch.unet import UNet
	net = UNet(len(channels), quantiles.size)
elif (net_name == 'per'):
	from models.per import Net
	net = Net(len(quantiles), len(channels))	
	
filename = (net_name + str(apply_log) + str(BATCH_SIZE) + '_' + str(n_epochs_arr) + '_' + str(lrs) + '_' + opt + filename_extension).replace(" ", "")

path_to_data = args.path_to_data
path_to_storage = args.path_to_storage

global path_to_save_model
path_to_save_model = os.path.join(path_to_storage, filename, 'saved_models')
if not Path(path_to_save_model).exists():
	os.makedirs(path_to_save_model)
global path_to_save_y
path_to_save_y = os.path.join(path_to_storage, filename, 'preds')
if not Path(path_to_save_y).exists():
	os.makedirs(path_to_save_y)
global path_to_save_images
path_to_save_images = os.path.join(path_to_storage, filename, 'images')
if not Path(path_to_save_images).exists():
	os.makedirs(path_to_save_images)
	
global log_directory
log_directory = os.path.join(path_to_storage, filename, 'runs')
if not Path(log_directory).exists():
	os.makedirs(log_directory)


path_to_train_data = os.path.join(path_to_data, 'train/npy_files')
path_to_stats = os.path.join(Path(path_to_train_data).parent, Path('stats.npy'))
path_to_val_data = os.path.join(path_to_data, 'validation/npy_files')
path_to_test_data = os.path.join(path_to_data, 'test/npy_files')


# DATA
def importData(channels, BATCH_SIZE, path_to_data, path_to_stats, apply_log=False):
	transforms_list = [Mask()]
	if apply_log:
		transforms_list.append(RandomLog())
	transforms_list.extend([RandomCrop(128), Standardize(path_to_data, path_to_stats, channels), ToTensor()])
	dataset = GOESRETRIEVALSDataset(
		path_to_data = path_to_data,
		channels = channels, 
		transform = transforms.Compose(transforms_list))
	print('number of samples:', len(dataset))

	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
	return(dataset, dataloader)

training_dataset, training_data = importData(channels, BATCH_SIZE, path_to_train_data, path_to_stats, apply_log=apply_log)
validation_dataset, validation_data  = importData(channels, BATCH_SIZE, path_to_val_data, path_to_stats, apply_log=apply_log)



# PLOT PERFORMANCE
def performance(data, qrnn, filename, fillvalue, apply_log):

	y_true = []
	y_pred = []
	
	torch.cuda.empty_cache()
	with torch.no_grad():
		for batch_index, batch in enumerate(data):
			y_true += [batch['label'].detach().numpy()]
			X = batch['box'].to(device).detach()
			y_pred += [qrnn.posterior_mean(x=X).cpu().detach().numpy()] 
	y_true = np.concatenate(y_true, axis=0)
	y_pred = np.concatenate(y_pred, axis=0)
	indices = (y_true != fillvalue)
	np.savetxt(os.path.join(path_to_save_y, filename+'.txt'), np.transpose(np.stack((y_true[indices],  y_pred[indices]))))
	
	
	bins = np.logspace(-2, 2, 100)
	if apply_log:
		y_true = np.exp(y_true[indices])
		y_pred = np.exp(y_pred[indices])
	freqs, _, _ = np.histogram2d(y_true[indices], y_pred[indices], bins=bins)

	f, ax = plt.subplots(figsize=(8, 9))

	p = ax.pcolormesh(bins, bins, freqs.T)
	ax.set_xlim([1e-2, 1e2])
	ax.set_ylim([1e-2, 1e2])
	ax.set_xscale("log")
	ax.set_yscale("log")
	ax.set_xlabel("Reference rain rate [mm / h]")
	ax.set_ylabel("Predicted rain rate [mm / h]")
	ax.plot(bins, bins, c="grey", ls="--")
	f.colorbar(p, ax=ax, orientation="horizontal", label="Surface precipitation [mm / h]")
	ax.set_aspect(1.0)

	plt.tight_layout()
	plt.savefig(os.path.join(path_to_save_images, filename+'.png'))
	

# TRAIN MODEL
qrnn = QRNN(quantiles=quantiles, model=net)
metrics = ["MeanSquaredError", "Bias", "CRPS"]

log_sub_dir = os.path.join(log_directory,str(n_epochs_arr)+'_'+str(lrs))
if not Path(log_sub_dir).exists():
	os.makedirs(log_sub_dir)
logger = TensorBoardLogger(np.sum(n_epochs_arr), log_directory=log_sub_dir)

if opt == "Adam":
	logger.set_attributes({"optimizer": "Adam", "n_epochs": str(n_epochs_arr), "learning_rates": str(lrs)}) 
elif opt == "SGD":
	logger.set_attributes({"optimizer": "SGD", "scheduler": "CosineAnnealingLR", "n_epochs": str(n_epochs_arr), "learning_rates": str(lrs)}) 
	optimizer = SGD(qrnn.model.parameters(), lr=0.1, momentum=0.9)

for i in range(len(n_epochs_arr)):
	if opt == "Adam":
		optimizer = Adam(qrnn.model.parameters(), lr=lrs[i])
		scheduler = None
	elif opt == "SGD":
		scheduler = CosineAnnealingLR(optimizer, n_epochs_arr[i], lrs[i])
		
	qrnn.train(training_data=training_data,
		      validation_data=validation_data,
		      keys=("box", "label"),
		      n_epochs=n_epochs_arr[i],
		      optimizer=optimizer,
		      scheduler=scheduler,
		      mask=fillvalue,
		      device=device,
		      metrics=metrics,
		      logger=logger);
	filename_tmp = filename+'_'+str(n_epochs_arr[i])+'_'+str(lrs[i])+'_'+str(i)
	qrnn.save(os.path.join(path_to_save_model, filename_tmp+'.pckl'))
	performance(training_data, qrnn, filename_tmp+'train', fillvalue, apply_log)
	performance(validation_data, qrnn, filename_tmp+'val', fillvalue, apply_log)	



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
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

from quantnn.qrnn import QRNN

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
args = parser.parse_args()

BATCH_SIZE = args.BATCH_SIZE
filename_extension = args.filename_ext
apply_log = args.log
print(apply_log)

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
	net =  XceptionFpn(len(channels), quantiles.size, n_features=128)
	
filename = net_name + str(apply_log) + str(BATCH_SIZE) + filename_extension

path_to_data = args.path_to_data
path_to_storage = args.path_to_storage

global path_to_save_model
path_to_save_model = os.path.join(path_to_storage, filename, 'saved_models')
if not Path(path_to_save_model).exists():
	os.makedirs(path_to_save_model)
global path_to_save_errors
path_to_save_errors = os.path.join(path_to_storage, filename, 'errors')
if not Path(path_to_save_errors).exists():
	os.makedirs(path_to_save_errors)
global path_to_save_y
path_to_save_y = os.path.join(path_to_storage, filename, 'preds')
if not Path(path_to_save_y).exists():
	os.makedirs(path_to_save_y)
	

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
	return(dataloader)

training_data = importData(channels, BATCH_SIZE, path_to_train_data, path_to_stats, apply_log=apply_log)
validation_data  = importData(channels, BATCH_SIZE, path_to_val_data, path_to_stats, apply_log=apply_log)



# PLOT PERFORMANCE
def performance(validation_data, qrnn, filename, fillvalue):

	y_true = []
	y_pred = []
	
	torch.cuda.empty_cache()
	with torch.no_grad():
		for batch_index, batch in enumerate(validation_data):
			y_true += [batch['label'].detach().numpy()]
			X = batch['box'].to(device).detach()
			y_pred += [qrnn.posterior_mean(x=X).cpu().detach().numpy()] 
	y_true = np.concatenate(y_true, axis=0)
	y_pred = np.concatenate(y_pred, axis=0)
	indices = (y_true != fillvalue)
	np.savetxt(os.path.join(path_to_save_y, filename+'.txt'), np.transpose(np.stack((y_true[indices],  y_pred[indices]))))
	

# TRAIN MODEL
qrnn_model = QRNN(quantiles=quantiles, model=net)
optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9)



def runTrain(optimizer, qrnn_model, training_data, validation_data, filename, n_epochs=10, lr=0.01):
	scheduler = CosineAnnealingLR(optimizer, n_epochs, lr)
	loss = qrnn_model.train(training_data=training_data,
		      validation_data=validation_data,
		      keys=("box", "label"),
		      n_epochs=n_epochs,
		      optimizer=optimizer,
		      scheduler=scheduler,
		      mask=fillvalue,
		      device=device);

	qrnn_model.save(os.path.join(path_to_save_model, filename))
	np.savetxt(os.path.join(path_to_save_errors, filename+'.txt'), loss))
	performance(validation_data, qrnn_model, filename, fillvalue)


n_epochs_arr = [10, 20, 40]
lrs = [0.01, 0.001, 0.0001]
for i in range(len(n_epochs_arr)):
	runTrain(optimizer, qrnn_model, training_data, validation_data, filename+str(i), n_epochs=n_epochs_arr[i], lr=lrs[i])
	



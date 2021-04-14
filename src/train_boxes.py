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

from load_data import GOESRETRIEVALSDataset, RandomCrop, Mask, Standardize, ToTensor
from models.FirstGenericNet import boxes_one 
net_name = 'boxes_one' 

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
args = parser.parse_args()

path_to_data = args.path_to_data
path_to_storage = args.path_to_storage
path_to_save_model = os.path.join(path_to_storage, 'saved_models')

path_to_train_data = os.path.join(path_to_data, 'train/npy_files')
path_to_stats = os.path.join(Path(path_to_train_data).parent, Path('stats.npy'))
path_to_val_data = os.path.join(path_to_data, 'validation/npy_files')
path_to_test_data = os.path.join(path_to_data, 'test/npy_files')


# SETUP
channels = list(range(8,17))
channels.remove(12)

fillvalue = -1

BATCH_SIZE = 256

quantiles = np.linspace(0.01, 0.99, 99)

# DATA
def importData(channels, BATCH_SIZE, path_to_data, path_to_stats):
	dataset = GOESRETRIEVALSDataset(
		path_to_data = path_to_data,
		channels = channels, 
		transform = transforms.Compose([Mask(), RandomCrop(128),
				Standardize(path_to_data, path_to_stats, channels),
				ToTensor()])
	)

	dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
	return(dataloader)

training_data = importData(channels, BATCH_SIZE, path_to_train_data, path_to_stats)
validation_data  = importData(channels, BATCH_SIZE, path_to_val_data, path_to_stats)



# PLOT PERFORMANCE
def plotPerformance(validation_data, qrnn, filename):

	y_true = []
	y_pred = []
	with torch.no_grad():
		for batch_index, batch in enumerate(validation_data):
			y_true += [batch['label'].detach().numpy()]
			X = batch['box'].to(device).detach()
			y_pred += [qrnn.posterior_mean(x=X).cpu().detach().numpy()] 
	y_true = np.concatenate(y_true, axis=0)
	y_pred = np.concatenate(y_pred, axis=0)

	bins = np.logspace(-2, 2, 81)
	indices = y_true >= 0.0
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
	plt.savefig(os.path.join(path_to_storage, 'images', filename))
	plt.close(f)


# TRAIN MODEL
net = Net(len(quantiles), len(channels))
qrnn_model = QRNN(quantiles=quantiles, model=net)
optimizer = SGD(net.parameters(), lr=0.1, momentum=0.9)

n_epochs = 10
scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.01)
qrnn_model.train(training_data=training_data,
              validation_data=validation_data,
              keys=("box", "label"),
              n_epochs=n_epochs,
              optimizer=optimizer,
              scheduler=scheduler,
              mask=fillvalue,
              device=device);

qrnn_model.save(os.path.join(path_to_save_model, 'conv_1'))
plotPerformance(validation_data, qrnn_model, 'conv_1.png')


n_epochs = 20
scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.001)
qrnn_model.train(training_data=training_data,
              validation_data=validation_data,
              keys=("box", "label"),
              n_epochs=n_epochs,
              optimizer=optimizer,
              scheduler=scheduler,
              mask=fillvalue,
              device=device);

qrnn_model.save(os.path.join(path_to_save_model, 'conv_2'))
plotPerformance(validation_data, qrnn_model, 'conv_2.png')

n_epochs = 40
scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.0001)
qrnn_model.train(training_data=training_data,
              validation_data=validation_data,
              keys=("box", "label"),
              n_epochs=n_epochs,
              optimizer=optimizer,
              scheduler=scheduler,
              mask=fillvalue,
              device=device);


qrnn_model.save(os.path.join(path_to_save_model, 'conv_3'))
plotPerformance(validation_data, qrnn_model, 'conv_3.png')




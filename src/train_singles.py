import numpy as np
import torch
from torchvision import transforms, utils
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'gpu'
print('device: ', device)

from quantnn.qrnn import QRNN
from quantnn.models.pytorch import BatchedDataset

from models.singles_fc import Net

# ARGUMENTS
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

# FUNCTIONS
# Plot performance on validation data
def plotPerformance(validation_data, qrnn_fc, filename):

        y_true = []
        y_pred_fc = []
        for x, y in validation_data:
                y_true += [y.detach().numpy()]
                y_pred_fc += [qrnn_fc.posterior_mean(x=x).cpu().detach().numpy()]
        y_true = np.concatenate(y_true, axis=0)
        y_pred_fc = np.concatenate(y_pred_fc, axis=0)

        bins = np.logspace(-2, 2, 41)
        indices = y_true >= 0.0
        freqs_fc, _, _ = np.histogram2d(y_true[indices], y_pred_fc[indices], bins=bins)

        indices = y_val >= 0.0

        f, ax = plt.subplots(figsize=(8, 9))

        p = ax.pcolormesh(bins, bins, freqs_fc.T)
        ax.set_xlim([1e-2, 1e2])
        ax.set_ylim([1e-2, 1e2])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Reference rain rate [mm / h]")
        ax.set_ylabel("Predicted rain rate [mm / h]")
        ax.set_title("(a) Fully-connected", loc="left")
        ax.plot(bins, bins, c="grey", ls="--")
        f.colorbar(p, ax=ax, orientation="horizontal", label="Surface precipitation [mm / h]")
        ax.set_aspect(1.0)

        plt.tight_layout()
        plt.savefig(os.path.join(path_to_storage, 'images', filename))
        plt.close(f)


# SETUP
channels = list(range(8,17))
channels.remove(12)

fillvalue = -1

n_epochs = 20
BATCH_SIZE = 32

quantiles = np.linspace(0.01, 0.99, 99)


X_train = np.load(path_to_data+'train/X_singles_dataset.npy')
y_train = np.load(path_to_data+'train/y_singles_dataset.npy')
X_val = np.load(path_to_data+'validation/X_singles_dataset.npy')
y_val = np.load(path_to_data+'validation/y_singles_dataset.npy')

#subs = 100
#X_train = X_train[:subs].astype(np.float32)
#y_train = y_train[:subs].astype(np.float32)
#X_val = X_val[:subs].astype(np.float32)
#y_val = y_val[:subs].astype(np.float32)
#print('size of training data: ', X_train.shape)


def Standardize(X, path_to_data):
    stats = np.load(path_to_data+'train/X_singles_stats.npy')
    return ((X-stats[0,:])/stats[1,:]).astype(np.float32)
    
X_train = Standardize(X_train, path_to_data)
X_val = Standardize(X_val, path_to_data)

model_fc = Net(len(quantiles), len(channels))
qrnn_fc = QRNN(quantiles=quantiles, model=model_fc)


training_data = BatchedDataset((X_train, y_train), BATCH_SIZE)
validation_data = BatchedDataset((X_val, y_val), BATCH_SIZE)
optimizer = SGD(model_fc.parameters(), lr=0.1, momentum=0.9)



n_epochs = 10
scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.01)
qrnn_fc.train(training_data=training_data,
              validation_data=validation_data,
              n_epochs=n_epochs,
              optimizer=optimizer,
              scheduler=scheduler,
              mask=-1,
              device=device);

qrnn_fc.save(os.path.join(path_to_storage, 'saved_models', 'singles_fc1'))
plotPerformance(validation_data, qrnn_fc, 'singles_fc1.png')


n_epochs = 20
scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.001)
qrnn_fc.train(training_data=training_data,
              validation_data=validation_data,
              n_epochs=n_epochs,
              optimizer=optimizer,
              scheduler=scheduler,
              mask=-1,
              device=device);

qrnn_fc.save(os.path.join(path_to_storage, 'saved_models', 'singles_fc2'))
plotPerformance(validation_data, qrnn_fc, 'singles_fc2.png')

n_epochs = 40
scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.0001)
qrnn_fc.train(training_data=training_data,
              validation_data=validation_data,
              n_epochs=n_epochs,
              optimizer=optimizer,
              scheduler=scheduler,
              mask=-1,
              device=device);

qrnn_fc.save(os.path.join(path_to_storage, 'saved_models', 'singles_fc3'))
plotPerformance(validation_data, qrnn_fc, 'singles_fc3.png')

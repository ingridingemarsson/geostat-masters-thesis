import numpy as np
import torch
from torchvision import transforms, utils
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device: ', device)

from quantnn.qrnn import QRNN
from quantnn.models.pytorch import BatchedDataset

from models.singles_fc import Net


# SETUP
channels = list(range(8,17))
channels.remove(12)

fillvalue = -1

n_epochs = 20
BATCH_SIZE = 32

quantiles = np.linspace(0.01, 0.99, 99)


path_to_data = 'dataset/data/dataset-singles/'
X_train = np.load(path_to_data+'train/X_singles_dataset.npy')
y_train = np.load(path_to_data+'train/y_singles_dataset.npy')
X_val = np.load(path_to_data+'validation/X_singles_dataset.npy')
y_val = np.load(path_to_data+'validation/y_singles_dataset.npy')

#subs = 100000
#X_train = X_train[:subs].astype(np.float32)
#y_train = y_train[:subs].astype(np.float32)
#X_val = X_val[:subs].astype(np.float32)
#y_val = y_val[:subs].astype(np.float32)
#print('size of training data: ', X_train.shape)


def Standardize(X, path_to_data):
    stats = np.load(path_to_data+'train/X_singles_stats.npy')
    return (X-stats[0,:])/stats[1,:].astype(np.float32)
    
X_train = Standardize(X_train, path_to_data).astype(np.float32)
X_val = Standardize(X_val, path_to_data).astype(np.float32)

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

qrnn_fc.save('models/saved_models/hej1')

n_epochs = 20
scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.001)
qrnn_fc.train(training_data=training_data,
              validation_data=validation_data,
              n_epochs=n_epochs,
              optimizer=optimizer,
              scheduler=scheduler,
              mask=-1,
              device=device);

qrnn_fc.save('models/saved_models/hej2')

n_epochs = 40
scheduler = CosineAnnealingLR(optimizer, n_epochs, 0.0001)
qrnn_fc.train(training_data=training_data,
              validation_data=validation_data,
              n_epochs=n_epochs,
              optimizer=optimizer,
              scheduler=scheduler,
              mask=-1,
              device=device);


qrnn_fc.save('models/saved_models/hej3')



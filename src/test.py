import numpy as np
import os
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
	nargs='+',
	default=["../dataset/data/dataset-test/test/"]
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
	default=64
	)
parser.add_argument(
	"--channel_inds", 
	help="Subset of avalible channel indices", 
	nargs="+", 
	type=int,
	default=list(range(0,8)))
args = parser.parse_args()



BATCH_SIZE = args.BATCH_SIZE

# SETUP
channel_inds = args.channel_inds
num_channels = len(channel_inds)

quantiles = np.linspace(0.01, 0.99, 99)


path_to_test_data = args.path_to_data
path_to_storage = args.path_to_storage


xception = QRNN.load('../results/xception64_[100]_0.01__boxes_100_0.01_0_t5412_v1354[0, 1, 2, 3, 4, 5, 6, 7]_Adam_1622288705.386947.pckl') #xception.pckl')



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

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    return(dataset, dataloader)



keys=("box", "label")

path_to_stats = os.path.join(Path(path_to_test_data).parents[0], 'train', 'stats.npy') #os.path.join(path_to_data, 'stats.npy')
path_to_test_data_files = os.path.join(path_to_test_data,'npy_files') # os.path.join(path_to_data,'dataset-boxes', 'test', 'npy_files')

test_dataset, test_data = importData(BATCH_SIZE, path_to_test_data_files, path_to_stats, channel_inds)



y_true_tot = []
y_pred_tot = []
#crps_tot = []

with torch.no_grad():
    for batch_index, batch_data in enumerate(test_data):
        print(batch_index)
        
        boxes = batch_data['box']
        y_true = batch_data['label']
        
        mask = (torch.less(y_true, 0))
        
        y_pred = xception.posterior_mean(boxes)
        #crps = xception.crps(x=boxes, y_true=y_true)
        
        y_true_tot += [y_true[~mask].detach().numpy()]
        y_pred_tot += [y_pred[~mask].detach().numpy()]
        #crps_tot += [crps[~mask].detach().numpy()]

        
y_true_tot_c = np.concatenate(y_true_tot, axis=0)
y_pred_tot_c = np.concatenate(y_pred_tot, axis=0)        
MSE = np.mean(np.square(np.subtract(y_true_tot_c, y_pred_tot_c)))

print(MSE)






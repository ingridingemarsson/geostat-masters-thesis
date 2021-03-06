import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self, num_quantiles, num_channels):

		super(Net, self).__init__()
		
		self.l1 = nn.Linear(num_channels, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 256)
		self.l4 = nn.Linear(256, 256)
		self.l5 = nn.Linear(256, 256)
		self.l6 = nn.Linear(256, 256)
		self.l7 = nn.Linear(256, 256)
		self.l8 = nn.Linear(256, num_quantiles)


    
	def forward(self, x):

		x = F.relu(self.l1(x))
		x = F.relu(self.l2(x))
		x = F.relu(self.l3(x))
		x = F.relu(self.l4(x))
		x = F.relu(self.l5(x))
		x = F.relu(self.l6(x))
		x = F.relu(self.l7(x))
		x = self.l8(x)
        
		return x

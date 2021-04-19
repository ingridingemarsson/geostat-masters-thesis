import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_quantiles, num_channels):
    		
        super(Net, self).__init__()
    
        self.c1 = nn.Conv2d(num_channels, 16*num_channels, 1, groups=num_channels)
        self.c2 = nn.Conv2d(16*num_channels, 16*num_channels, 1, groups=num_channels)
        self.c3 = nn.Conv2d(16*num_channels, 16*num_channels, 1)
        self.c4 = nn.Conv2d(16*num_channels, num_quantiles, 1)
        self.b1 = nn.BatchNorm2d(16*num_channels)
        self.b2 = nn.BatchNorm2d(16*num_channels)
        self.b3 = nn.BatchNorm2d(16*num_channels)
    
    
    def forward(self, x):

        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        
        x = self.c4(x)
        
        return x

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_quantiles, num_channels):
    		
        super(Net, self).__init__()
    
        self.c1 = nn.Conv2d(num_channels, 256, 1)
        self.c2 = nn.Conv2d(256, 256, 1)
        self.c3 = nn.Conv2d(256, 256, 1)
        self.c4 = nn.Conv2d(256, num_quantiles, 1)
        self.b1 = nn.BatchNorm2d(256)
        self.b2 = nn.BatchNorm2d(256)
        self.b3 = nn.BatchNorm2d(256)
    
    
    def forward(self, x):

        x = F.relu(self.b1(self.c1(x)))
        x = F.relu(self.b2(self.c2(x)))
        x = F.relu(self.b3(self.c3(x)))
        
        x = self.c4(x)
        
        return x

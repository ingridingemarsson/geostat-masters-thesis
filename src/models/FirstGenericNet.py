import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_quantiles, num_channels):
    		
        super(Net, self).__init__()
    
        self.c1 = nn.Conv2d(num_channels, 128, 1)
        self.c2 = nn.Conv2d(128, 128, 1)
        self.c3 = nn.Conv2d(128, num_quantiles, 1)
        self.b = nn.BatchNorm2d(128)
    
    
    def forward(self, x):

        x = F.relu(self.b(self.c1(x)))
        
        for i in range(2):
            x = F.relu(self.b(self.c2(x)))
        
        x = self.c3(x)
        
        return x
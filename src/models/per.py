import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_quantiles, num_channels):
    		
        super(Net, self).__init__()
    
        n = 2
        
        self.c1 = nn.Conv2d(num_channels, num_channels*n, 3, groups=num_channels, padding=1)
        self.m1 = nn.MaxPool2d(2)
        self.c2 = nn.Conv2d(num_channels*n, num_channels*n*2, 3, groups=num_channels, padding=1)
        self.m2 = nn.MaxPool2d(2)
        self.t1 = nn.ConvTranspose2d(num_channels*n*2, num_channels*n*2, 6, stride=2, padding=2)
        self.t2 = nn.ConvTranspose2d(num_channels*n*2, num_channels*n*4, 6, stride=2, padding=2)
        self.c3 = nn.Conv2d(num_channels*n*4, num_channels*n*8, 3, padding=1)
        self.c4 = nn.Conv2d(num_channels*n*8, num_quantiles, 9, padding=4)

    def forward(self, x):
        x = F.relu(self.c1(x))
        x = F.relu(self.m1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.m2(x))
        x = F.relu(self.t1(x))
        x = F.relu(self.t2(x))
        x = F.relu(self.c3(x))
        x = F.relu(self.c4(x))
        
        return x

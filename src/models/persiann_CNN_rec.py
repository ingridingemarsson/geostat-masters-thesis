import torch.nn as nn
import torch.nn.functional as F

# From PERSIANN-CNN: Precipitation Estimation from Remotely Sensed Information Using Artificial Neural Networks–Convolutional Neural Networks

class Net(nn.Module):
    def __init__(self, num_quantiles, num_channels, n=2):
        super(Net, self).__init__()
        
        self.z1 = nn.ZeroPad2d((2, 1, 2, 1))
        self.c1 = nn.Conv2d(num_channels, num_channels*2**n, kernel_size=4, groups=num_channels) 
        self.c2 = nn.Conv2d(num_channels*2**n, num_channels*2**(n+1), kernel_size=4, groups=num_channels) 
        self.z2 = nn.ZeroPad2d((1, 0, 1, 0))
        self.p = nn.MaxPool2d(2)
        
        self.t1 = nn.ConvTranspose2d(num_channels*2**(n+1), num_channels*2**(n+1), 5, stride=2, padding=2, output_padding=1, groups=1)
        self.t2 = nn.ConvTranspose2d(num_channels*2**(n+1), num_channels*2**(n+2), 5, stride=2,  padding=2, output_padding=1, groups=1)
        self.c3 = nn.Conv2d(num_channels*2**(n+2), num_channels*2**(n+3), 4, groups=1) 
        self.c4 = nn.Conv2d(num_channels*2**(n+3), num_quantiles, 9, padding=4, groups=1) 
        
        
    def forward(self, x):

        x = F.relu(self.c1(self.z1(x)))
        x = F.relu(self.p(self.z2(x)))
        x = F.relu(self.c2(self.z1(x)))
        x = F.relu(self.p(self.z2(x)))
        
        x = F.relu(self.t1(x))
        x = F.relu(self.t2(x))
        x = F.relu(self.c3(self.z1(x)))
        x = F.relu(self.c4(x))

        return x

        

    
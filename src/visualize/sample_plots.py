import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import torch

def plotRandomSample(dataset, net = None, quantile_num = 0):
    ncols = 3
    if not net == None:
        ncols = 4
    
    f = plt.figure(figsize=(15, 6))
    gs =  GridSpec(2, ncols, figure=f, height_ratios=[1.0, 0.1])
    index = np.random.randint(len(dataset))
    
    precip_norm = LogNorm(1e-2, 1e2)

    ax = plt.subplot(gs[0, 0])
    m = ax.imshow(dataset[index]['box'].numpy()[0], cmap=plt.get_cmap('inferno'))
    ax.grid(False)
    ax.set_title("(a) channel 8", loc="left")
    ax = plt.subplot(gs[1, 0])
    plt.colorbar(m, cax=ax, orientation="horizontal", label="Normalized brightness temperature")

    ax = plt.subplot(gs[0, 1])
    m = ax.imshow(dataset[index]['box'].numpy()[1], cmap=plt.get_cmap('inferno'))
    ax.grid(False)
    ax.set_title("(b) channel 13", loc="left")
    ax = plt.subplot(gs[1, 1])
    plt.colorbar(m, cax=ax, orientation="horizontal", label="Normalized brightness temperature")

    ax = plt.subplot(gs[0, 2])
    m = ax.imshow(dataset[index]['label'].numpy(), norm=precip_norm, cmap=plt.get_cmap('GnBu'))
    ax.imshow(dataset[index]['label']!=-1,  cmap=plt.get_cmap('binary_r'), alpha = 0.02)
    ax.grid(False)
    ax.set_title("(c) Reference rain rate", loc="left")
    ax = plt.subplot(gs[1, 2])
    plt.colorbar(m, cax=ax, orientation="horizontal", label=r"Rain rate [mm/h]")

    if not net == None:
        ax = plt.subplot(gs[0, 3])
        m = ax.imshow(torch.squeeze(net(dataset[index]['box'].unsqueeze(0))[:,quantile_num]).detach().numpy(),
                      norm=precip_norm, cmap=plt.get_cmap('GnBu'))
        ax.imshow(dataset[index]['label']!=-1,  cmap=plt.get_cmap('binary_r'), alpha = 0.02)
        ax.grid(False)
        ax.set_title("(d) Model prediction rain rate", loc="left")
        ax = plt.subplot(gs[1, 3])
        plt.colorbar(m, cax=ax, orientation="horizontal", label=r"Rain rate [mm/h]")


    plt.tight_layout()
    

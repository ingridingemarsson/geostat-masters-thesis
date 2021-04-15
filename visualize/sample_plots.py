import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import torch

def plotRandomSample(dataset, qrnn = None, device='cpu', c=[0,4], uselog=False):
    ncols = 3
    if not qrnn == None:
        ncols = 4
    
    f = plt.figure(figsize=(15, 6))
    gs =  GridSpec(2, ncols, figure=f, height_ratios=[1.0, 0.1])
    index = np.random.randint(len(dataset))
    data = dataset[index]
    precip_norm = LogNorm(1e-2, 1e2)

    ax = plt.subplot(gs[0, 0])
    m = ax.imshow(data['box'].numpy()[c[0]], cmap=plt.get_cmap('inferno'))
    ax.grid(False)
    ax.set_title("(a) channel c[0]", loc="left")
    ax = plt.subplot(gs[1, 0])
    plt.colorbar(m, cax=ax, orientation="horizontal", label="Normalized brightness temperature")

    ax = plt.subplot(gs[0, 1])
    m = ax.imshow(data['box'].numpy()[c[1]], cmap=plt.get_cmap('inferno'))
    ax.grid(False)
    ax.set_title("(b) channel c[1]", loc="left")
    ax = plt.subplot(gs[1, 1])
    plt.colorbar(m, cax=ax, orientation="horizontal", label="Normalized brightness temperature")

    ax = plt.subplot(gs[0, 2])
    y_true = data['label'].numpy()
    if uselog:
    	y_true = np.where(y_true != -1, np.exp(y_true), y_true) 
    m = ax.imshow(y_true, norm=precip_norm, cmap=plt.get_cmap('GnBu'))
    ax.imshow(y_true!=-1,  cmap=plt.get_cmap('binary_r'), alpha = 0.02)
    ax.grid(False)
    ax.set_title("(c) Reference rain rate", loc="left")
    ax = plt.subplot(gs[1, 2])
    plt.colorbar(m, cax=ax, orientation="horizontal", label=r"Rain rate [mm/h]")

    if not qrnn == None:
        y_pred = torch.squeeze(qrnn.posterior_mean(x=data['box'].unsqueeze(0).to(device))).cpu().detach().numpy()
        if uselog:
            y_pred = np.where(y_pred != -1, np.exp(y_pred), y_pred) 
        ax = plt.subplot(gs[0, 3])
        m = ax.imshow(y_pred, norm=precip_norm, cmap=plt.get_cmap('GnBu'))
        ax.imshow(y_true!=-1,  cmap=plt.get_cmap('binary_r'), alpha = 0.02)
        ax.grid(False)
        ax.set_title("(d) Model prediction rain rate", loc="left")
        ax = plt.subplot(gs[1, 3])
        plt.colorbar(m, cax=ax, orientation="horizontal", label=r"Rain rate [mm/h]")


    plt.tight_layout()
    return(index)

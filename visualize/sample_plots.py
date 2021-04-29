import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import torch

def plotRandomSample(dataset, channels, qrnn = None, device='cpu', cha=[0,4], uselog=False):
    ncols = len(cha)+1
    if not qrnn == None:
        ncols += 1
    
    f = plt.figure(figsize=(15, 6))
    gs =  GridSpec(2, ncols, figure=f, height_ratios=[1.0, 0.1])
    index = np.random.randint(len(dataset))
    data = dataset[index]
    precip_norm = LogNorm(1e-2, 1e2)

    i = 0
    for c in cha:
	    ax = plt.subplot(gs[0, i])
	    m = ax.imshow(data['box'].numpy()[c], cmap=plt.get_cmap('inferno'))
	    ax.grid(False)
	    ax.set_title("(a) channel "+str(channels[c]), loc="left")
	    ax = plt.subplot(gs[1, i])
	    plt.colorbar(m, cax=ax, orientation="horizontal", label="Normalized brightness temperature")
	    i+=1

    ax = plt.subplot(gs[0, len(cha)])
    y_true = data['label'].numpy()
    if uselog:
    	y_true = np.where(y_true != -1, np.exp(y_true), y_true) 
    m = ax.imshow(y_true, norm=precip_norm, cmap=plt.get_cmap('GnBu'))
    ax.imshow(y_true!=-1,  cmap=plt.get_cmap('binary_r'), alpha = 0.02)
    ax.grid(False)
    ax.set_title("(c) Reference rain rate", loc="left")
    ax = plt.subplot(gs[1, len(cha)])
    plt.colorbar(m, cax=ax, orientation="horizontal", label=r"Rain rate [mm/h]")

    if not qrnn == None:
        y_pred = torch.squeeze(qrnn.posterior_mean(x=data['box'].unsqueeze(0).to(device))).cpu().detach().numpy()
        if uselog:
            y_pred = np.where(y_pred != -1, np.exp(y_pred), y_pred) 
        ax = plt.subplot(gs[0, len(cha)+1])
        m = ax.imshow(y_pred, norm=precip_norm, cmap=plt.get_cmap('GnBu'))
        ax.imshow(y_true!=-1,  cmap=plt.get_cmap('binary_r'), alpha = 0.02)
        ax.grid(False)
        ax.set_title("(d) Model prediction rain rate", loc="left")
        ax = plt.subplot(gs[1, len(cha)+1])
        plt.colorbar(m, cax=ax, orientation="horizontal", label=r"Rain rate [mm/h]")


    plt.tight_layout()
    return(index)

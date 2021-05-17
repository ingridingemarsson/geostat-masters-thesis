import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
import torch

def plotRandomSample(dataset, channels, qrnn=None, index=None, device='cpu', cha=[0,4], data_type='boxes'):
    ncols = len(cha)+1
    if not qrnn == None:
        ncols += 1
    
    f = plt.figure(figsize=(15, 6))
    gs =  GridSpec(2, ncols, figure=f, height_ratios=[1.0, 0.1])
    
    if index==None:
    	index = np.random.randint(len(dataset))
    data = dataset[index]
    precip_norm = LogNorm(1e-2, 1e2)
    
    y_true = data['label'].numpy()

    i = 0
    for c in cha:
        ax = plt.subplot(gs[0, i])
        m = ax.imshow(data['box'].numpy()[c], cmap=plt.get_cmap('inferno'))
        ax.contour(y_true!=-1,  cmap=plt.get_cmap('binary_r'),  alpha = 0.7, linewidths=0.1)
        ax.grid(False)
        ax.set_title("(a) channel "+str(channels[c]), loc="left")
        ax = plt.subplot(gs[1, i])
        plt.colorbar(m, cax=ax, orientation="horizontal", label="Normalized brightness temperature")
        i+=1

    ax = plt.subplot(gs[0, len(cha)])

    m = ax.imshow(y_true, norm=precip_norm, cmap=plt.get_cmap('GnBu'))
    #ax.imshow(y_true!=-1,  cmap=plt.get_cmap('binary_r'), alpha = 0.02)
    ax.contour(y_true!=-1,  cmap=plt.get_cmap('binary_r'),  alpha = 0.7, linewidths=0.1)
    ax.grid(False)
    ax.set_title("(c) Reference rain rate", loc="left")
    ax = plt.subplot(gs[1, len(cha)])
    plt.colorbar(m, cax=ax, orientation="horizontal", label=r"Rain rate [mm/h]")

    if not qrnn==None:
        if (data_type=='boxes'):
            y_pred = torch.squeeze(qrnn.posterior_mean(x=data['box'].unsqueeze(0).to(device))).cpu().detach().numpy()
        elif (data_type=="singles"):
            x = data['box'].unsqueeze(0).to(device)
            X = torch.transpose(torch.squeeze(torch.flatten(x, start_dim=2)), 0, 1)
            y_pred = qrnn.posterior_mean(X).cpu().detach()
            y_pred = torch.reshape(y_pred, (int(np.sqrt(y_pred.shape[0])), int(np.sqrt(y_pred.shape[0])))).numpy()

        ax = plt.subplot(gs[0, len(cha)+1])
        m = ax.imshow(y_pred, norm=precip_norm, cmap=plt.get_cmap('GnBu'))
        #ax.imshow(y_true!=-1,  cmap=plt.get_cmap('binary_r'), alpha = 0.02)
        ax.contour(y_true!=-1,  cmap=plt.get_cmap('binary_r'),  alpha = 0.7, linewidths=0.1)
        ax.grid(False)
        ax.set_title("(d) Model prediction rain rate", loc="left")
        ax = plt.subplot(gs[1, len(cha)+1])
        plt.colorbar(m, cax=ax, orientation="horizontal", label=r"Rain rate [mm/h]")


    plt.tight_layout()
    return(index)

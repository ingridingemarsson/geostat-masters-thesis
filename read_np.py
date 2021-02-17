import numpy as np
import matplotlib.pyplot as plt

X = np.load('datasetX.npy')
y = np.load('datasety.npy')

frac = 0.046
pad = 0.04
im = 18

fig, axs = plt.subplots(ncols = 3, sharex=True, sharey=True)

pos0 = axs[0].imshow(X[im][0])
pos0.set_clim(np.nanmin(X[:][0]), np.nanmax(X[:][0]))
fig.colorbar(pos0, ax=axs[0], fraction=frac, pad=pad)

pos1 = axs[1].imshow(X[im][1])
pos1.set_clim(np.nanmin(X[:][1]), np.nanmax(X[:][1]))
fig.colorbar(pos1, ax=axs[1], fraction=frac, pad=pad)

pos2 = axs[2].imshow(y[im])
pos2.set_clim(np.nanmin(y), np.nanmax(y))
fig.colorbar(pos2, ax=axs[2], fraction=frac, pad=pad)

plt.show()



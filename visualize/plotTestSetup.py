import numpy as np
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

### Settings 

plt.style.use('seaborn-whitegrid')
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

SMALL_SIZE = 16 #8
MEDIUM_SIZE = 18 #10
BIGGER_SIZE = 20 #12
matplotlib.rc('font', size=SMALL_SIZE)          # controls default text sizes
matplotlib.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
matplotlib.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
matplotlib.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

color_neutral = '#990909'
color_grid = "#e9e9e9"


variable_dict = {'HE_precip': {'color': '#d1b126', 'label': 'HE', 'alpha': 0.4},
                 'HE_precip_corr': {'color':'#ec6726', 'label': 'HE corrected', 'alpha': 0.4},
                 'mlp_posterior_mean': {'color': '#327a4f',
                 'label': 'MLP posterior mean', 'alpha': 0.4},
                 'mlp_Q0.95': {'color': '#327a4f',
                 'label': 'MLP 95th quantile', 'alpha': 0.4},
                 'xception_posterior_mean': {'color': '#72196d','label': 'CNN posterior mean', 'alpha': 0.4},
                 'xception_Q0.95':  {'color': '#72196d','label': 'CNN 95th quantile', 'alpha': 0.4},
                 'gauge_precip': {'color': '#64a6a1', 'label': 'Gauges', 'alpha': 0.4},
                 'gpm': {'color': '#64a6a1', 'label': 'GPM label', 'alpha': 0.4},
                 'gpm_var': {'color': '#669aa6', 'label': 'GPM label', 'alpha': 0.4}
                }

#figure size
figsize_single_plot = (9,6)
figsize_two_cols = (12,6)
figsize_four = (13,10)

#cmap
big = cm.get_cmap('magma', 512)
newcmp = ListedColormap(big(np.linspace(0.1, 0.9, 256)))


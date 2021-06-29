import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from matplotlib import cm

import plotTestSetup as setup



def plotFalse(data_dict, bins, main_var, var_list, ty='FP', threshold=1e-1, crop_at=10.1, filename=None, quantity='gauges'):

    histtype=['bar']+['step']*(len(var_list)-1)
    alpha = [setup.variable_dict[var_list[0]]['alpha']]+[1]*(len(var_list)-1)

    fig, ax = plt.subplots(ncols=2, figsize=setup.figsize_two_cols, sharey=True)

    def rangeSubplotFP(data_dict, bins, main_var, other_vars, axnum=0, title="Whole range"):
        for i in range(len(other_vars)):
            pp = data_dict[other_vars[i]][data_dict[main_var]<=threshold]
            ax[axnum].hist(pp[pp>threshold],
                       bins=bins,
                       color=setup.variable_dict[other_vars[i]]['color'],
                       histtype=histtype[i], alpha=alpha[i],
                       label=setup.variable_dict[other_vars[i]]['label'])    
        ax[axnum].set_yscale("log")
        ax[axnum].grid(True,which="both",ls="--",c=setup.color_grid) 
        ax[axnum].set_xlabel(quantity.capitalize())
        ax[axnum].set_title(title)
        
        
    def rangeSubplotFN(data_dict, bins, main_var, other_vars, axnum=0, title="Whole range"):
        for i in range(len(other_vars)):
            yp = data_dict[main_var][data_dict[other_vars[i]]<=threshold]
            ax[axnum].hist(yp[yp>threshold],
                       bins=bins,
                       color=setup.variable_dict[other_vars[i]]['color'],
                       histtype=histtype[i], alpha=alpha[i],
                       label=setup.variable_dict[other_vars[i]]['label'])    
        ax[axnum].set_yscale("log")
        ax[axnum].grid(True,which="both",ls="--",c=setup.color_grid) 
        ax[axnum].set_xlabel(quantity.capitalize())
        ax[axnum].set_title(title)    


    if ty == 'FP':
        rangeSubplotFP(data_dict, bins, main_var, var_list, axnum=0, title="Whole range")
        subs2 = np.argmin(np.abs(bins-crop_at))+1
        rangeSubplotFP(data_dict, bins[:subs2],  main_var, var_list, 
                         axnum=1, title='Range below '+str(round(bins[subs2-1],1)) + ' mm')
    elif ty == 'FN':
        rangeSubplotFN(data_dict, bins, main_var, var_list, axnum=0, title="Whole range")
        subs2 = np.argmin(np.abs(bins-crop_at))+1
        rangeSubplotFN(data_dict, bins[:subs2],  main_var, var_list, 
                         axnum=1, title='Range below '+str(round(bins[subs2-1],1)) + ' mm')        

    ax[0].set_ylabel('Frequency')
    plt.setp(ax[0].spines.values(), color='black')

    handles, labels = ax[0].get_legend_handles_labels()
    main_var_ind = labels.index(setup.variable_dict[var_list[0]]['label'])
    labels.pop(main_var_ind)
    main_var_handle = handles[main_var_ind]
    handles.pop(main_var_ind)
    new_handles = [Line2D([], [], c=h.get_edgecolor(),  linestyle=h.get_linestyle()) for h in handles]
    labels = [setup.variable_dict[var_list[0]]['label']] + labels
    new_handles = [main_var_handle] + new_handles

    fig.legend(handles=new_handles, labels=labels, loc="upper center", ncol=5, borderaxespad=-0.30)
    plt.tight_layout()

    if filename!=None:
        plt.savefig(filename, bbox_inches='tight') #'../plots/thesis/error_gauge.pdf'
        
        
        
def plotError(data_dict, bins, main_var, var_list, crop_at=[-10.1,10.1], filename=None, quantity='gauges'):

    histtype=['bar']+['step']*(len(var_list)-1)
    alpha = [setup.variable_dict[var_list[0]]['alpha']]+[1]*(len(var_list)-1)

    fig, ax = plt.subplots(ncols=2, figsize=setup.figsize_two_cols, sharey=True)

    def rangeSuplotErr(data_dict, bins, main_var, other_vars, axnum=0, title="Whole range"):

        for i in range(len(other_vars)):
            ax[axnum].hist(
                np.subtract(data_dict[other_vars[i]], data_dict[main_var]),
                histtype=histtype[i], bins=bins, alpha=alpha[i],
                label=setup.variable_dict[other_vars[i]]['label'], 
                color=setup.variable_dict[other_vars[i]]['color'], 
                linewidth=1.2)

        ax[axnum].set_yscale("log")
        ax[axnum].grid(True, which="both", ls="--", c=setup.color_grid)
        ax[axnum].set_xlabel(quantity.capitalize())
        ax[axnum].set_title(title)

    rangeSuplotErr(data_dict, bins, main_var, var_list, axnum=0, title="Whole range")
    subs1 = np.argmin(np.abs(bins-crop_at[0]))
    subs2 = np.argmin(np.abs(bins-crop_at[1]))+1
    rangeSuplotErr(data_dict, bins[subs1:subs2],  main_var, var_list, 
                     axnum=1, title='Range from ' +str(round(bins[subs1],1))+' to '+str(round(bins[subs2-1],1)) + ' mm')

    ax[0].set_ylabel('Frequency')
    plt.setp(ax[0].spines.values(), color='black')

    handles, labels = ax[0].get_legend_handles_labels()
    main_var_ind = labels.index(setup.variable_dict[var_list[0]]['label'])
    labels.pop(main_var_ind)
    main_var_handle = handles[main_var_ind]
    handles.pop(main_var_ind)
    new_handles = [Line2D([], [], c=h.get_edgecolor(),  linestyle=h.get_linestyle()) for h in handles]
    labels = [setup.variable_dict[var_list[0]]['label']] + labels
    new_handles = [main_var_handle] + new_handles

    fig.legend(handles=new_handles, labels=labels, loc="upper center", ncol=5, borderaxespad=-0.30)
    plt.tight_layout()

    if filename!=None:
        plt.savefig(filename, bbox_inches='tight') #'../plots/thesis/error_gauge.pdf'       
        
        
        
def plotDistribution(data_dict, bins, main_var, var_list, crop_at=10.1, filename=None, quantity='gauges', linestyles=None):

    histtype='step'
    fig, ax = plt.subplots(ncols=2, figsize=setup.figsize_two_cols, sharey=True)
    if (linestyles == None) & (len(var_list)>0):
        linestyles = ['solid']*len(var_list)

    def rangeSuplotDists(data_dict, bins, main_var, other_vars, axnum=0, title="Whole range"):

        ax[axnum].hist(
            data_dict[main_var], 
            histtype='bar',
            bins=bins,
            alpha=setup.variable_dict[main_var]['alpha'], 
            label=setup.variable_dict[main_var]['label'],
            color=setup.variable_dict[main_var]['color'],
            linewidth=0.0, 
            rasterized=True)

        for i in range(len(other_vars)):
            ax[axnum].hist(
                data_dict[other_vars[i]], 
                histtype=histtype, bins=bins, 
                label=setup.variable_dict[other_vars[i]]['label'], 
                color=setup.variable_dict[other_vars[i]]['color'], 
                linestyle=linestyles[i],
                linewidth=1.2)

        ax[axnum].set_yscale("log")
        ax[axnum].grid(True, which="both", ls="--", c=setup.color_grid)
        ax[axnum].set_xlabel(quantity.capitalize())
        ax[axnum].set_title(title)

    rangeSuplotDists(data_dict, bins, main_var, var_list, axnum=0, title="Whole range")
    subs2 = list(bins).index(crop_at)+1
    rangeSuplotDists(data_dict, bins[:subs2],  main_var, var_list, 
                     axnum=1, title='Range below '+str(round(bins[subs2-1],1)) + ' mm')

    ax[0].set_ylabel('Frequency')
    plt.setp(ax[0].spines.values(), color='black')

    handles, labels = ax[0].get_legend_handles_labels()
    main_var_ind = labels.index(setup.variable_dict[main_var]['label'])
    labels.pop(main_var_ind)
    main_var_handle = handles[main_var_ind]
    handles.pop(main_var_ind)
    new_handles = [Line2D([], [], c=h.get_edgecolor(),  linestyle=h.get_linestyle()) for h in handles]
    labels = [setup.variable_dict[main_var]['label']] + labels
    new_handles = [main_var_handle] + new_handles

    fig.legend(handles=new_handles, labels=labels, loc="upper center", ncol=5, borderaxespad=-0.30)
    plt.tight_layout()

    if filename!=None:
        plt.savefig(filename, bbox_inches='tight') #'../plots/thesis/pdf_gauge.pdf'
        
        
        
def hist2D(data_dict, y_true, y_preds, norm_type=None, quantity='gauges'):
    fig, ax = plt.subplots(ncols=len(y_preds), figsize=setup.figsize_two_cols, sharey=True)
    
    y_true = data_dict[y_true]
    bins = np.logspace(-1, 2, 51)
    
    old_vmax = 0
    freqs_list = []
    for i in range(len(y_preds)):
        y_pred = data_dict[y_preds[i]]

        freqs, _, _ = np.histogram2d(y_true, y_pred, bins=bins)
        freqs[freqs==0.0] = np.nan
        extend = 'neither'


        if norm_type==None:
            freqs_normed = freqs
            colorbar_label = 'Frequency'
            vmax = np.max([old_vmax,np.nanmax(freqs)])
            old_vmax = vmax
        elif norm_type=='colwise':
            freqs_normed = freqs
            colorbar_label = 'Frequency, normalized columnwise'
            for col_ind in range(freqs.shape[0]):
                if np.isnan(freqs[col_ind, :]).all():
                    freqs_normed[col_ind, :] = np.array([np.nan] * freqs.shape[1])
                else:
                    freqs_normed[col_ind, :] = freqs[col_ind, :] / np.nansum(freqs[col_ind, :])
            vmax=np.max([old_vmax, np.percentile(freqs_normed[np.isnan(freqs_normed)==False], 95)])
            old_vmax = vmax
            extend = 'max'
            print(vmax)     
        
        freqs_list.append(freqs_normed)
            
    for i in range(len(y_preds)):
        freqs_normed = freqs_list[i]
        m = ax[i].pcolormesh(bins, bins, freqs_normed.T, cmap=setup.newcmp,
                        vmax=vmax,
                          linewidth=0.0, rasterized=True)
        
        ax[i].set_xlim([1e-1, 1e2])
        ax[i].set_ylim([1e-1, 1e2])
        ax[i].set_xscale("log")
        ax[i].set_yscale("log")
        ax[i].set_xlabel("Reference "+quantity)
        ax[i].plot(bins, bins, c="grey", ls="--")
        ax[i].grid(True,which="both",ls="--",c=setup.color_grid)
        ax[i].set_aspect('equal')
        ax[i].set_title(setup.variable_dict[y_preds[i]]['label'])
        
        ax[0].set_ylabel("Predicted "+quantity)
        
    fig.subplots_adjust(wspace=0.08)
    fig.colorbar(m, ax=ax, fraction=0.023, pad=0.021, extend=extend).set_label(label=colorbar_label, size=18)


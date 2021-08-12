"""
Functions to plot the parameters
"""

import HEA.plot.tools as pt
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def plot_params(value_err_params_cats, label_params, label_cats, color_cats,
                    list_categories=None, list_params=None,
                    offset_cats=None,
                    fig_name=None,
                    folder_name=None,
                    xlabel="Value",
                    ylabel="Parameters",
                    marker='o', markersize=5, 
                    zero_line=False,
                    elinestyle_cats={},
                    capsize=3, capthick=1,
                    title=None,
                    **kwargs
                    ):
    """ Plot the value of some parameters for different categories, in a 2D plot:

    * y-axis: one parameter, with an offset to show the parameter for different categories
    * x-axis: value of the parameter

    Parameters
    ----------
    value_err_params_cats: dict[str:dict[str:dict['v':float, 'e':float]]
        Associates the name of a category with another dictionnary
        that gives the value (``'v'``) and error (``'e'``) of each parameter
    label_params: dict[str:str]
        Associates a parameter name with its label
    label_cats: dict[str:str]
        Associates a category name with its label
    color_cats: dict[str:str]
        Associates a category name with its color
    list_categories: list(str)
        List of the categories. If not provided, taken to be all the keys
        of the ``value_err_params_cats`` dictionnary.
    list_params: list(str)
        List of the parameters. If not provided, taken to be all the keys of 
        the dictionnary of the first category of the
        ``value_err_params_cats`` dictionnary
    offset_cats: dict[str: float] or float
        associates a category with the offset to apply to it in the plot.
        If not provided, computed with an offset of step = -0.1.
        If float, the step is given by offset_cats. An offset of 1 corresponds
        to the next parameter.
    xlabel, ylabel: str
        Label of the x and y axis of the plot
    fig_name, folder_name: str
        Name of the figure and of the folder where to save the plot.
        If ``fig_name`` is not provided, the plot is not saved.
    zero_line: bool
        Do we plot a ``x=0`` line?
    elinestyle_cats: dict[str]
        Associates a category with the line style of the error bars
    marker, markersize, kwargs:
        passed to :py:func:`matplotlib.pyplot.plot`
    """

    value_err_params_cats = deepcopy(value_err_params_cats)
    #print(value_err_params_cats['true'])

    ## LIST OF PARAMS AND CATEGORIES ===========================================    

    if list_categories is None:
        list_categories = list(value_err_params_cats.keys())
    if list_params is None:
        list_params = list(value_err_params_cats[list_categories[0]].keys())

    ## Kwargs =================================================================
    
    kwargs_cats = {}
    for cat in list_categories:
        kwargs_cats[cat] = {}
        for kwarg_name, kwarg_dict  in kwargs.items():
            if isinstance(kwarg_dict, dict):
                if cat in kwarg_dict:
                    kwargs_cats[cat][kwarg_name] = kwarg_dict[cat]
            else:
                kwargs_cats[cat] = kwarg_dict
    # FILL THE X AND Y VALUES AND LABELS ======================================
    # y
    y = np.arange(1, len(list_params)+1)
    
    # x +/- xerr
    x_cats = {}
    xerr_cats = {}
    
    # Instantiate the lists of values and errors
    for cat in list_categories:
        x_cats[cat] = []
        xerr_cats[cat] = []

    

    for cat in list_categories:
        for param_name in list_params:
            if isinstance(value_err_params_cats[cat][param_name], dict):
                x_cats[cat].append(value_err_params_cats[cat][param_name]['v'])
                if 'e' in value_err_params_cats[cat][param_name]:
                    xerr_cats[cat].append(value_err_params_cats[cat][param_name]['e'])
                else:
                    xerr_cats[cat].append(0)
            else:
                x_cats[cat].append(value_err_params_cats[cat][param_name])
                xerr_cats[cat].append(0)

    # labels of the parameters
    yticks_labels = []
    for param_name in list_params:
        if isinstance(label_params[param_name], dict):
            yticks_labels.append(label_params[param_name]['latex'])
        else:
            yticks_labels.append(label_params[param_name])

    ## OFFSETS ===============================================
    if not isinstance(offset_cats, dict):
        if isinstance(offset_cats, float):
            step = offset_cats
        else:
            step = -0.1
        offset_cats = {}
        i = 0
        for cat in list_categories:
            offset_cats[cat] = i * step
            i+=1

    ## PLOT =================================================
    
    fig, ax = plt.subplots(figsize=(8, 8))

    for cat in list_categories:
        print("Category:", cat)
        eb = ax.errorbar(
            x=x_cats[cat], y=y + offset_cats[cat],
            xerr=xerr_cats[cat],
            label=label_cats[cat],
            color=color_cats[cat],
            marker=marker,
            linestyle='', markersize=markersize,
            capsize=capsize, capthick=capthick,
            **kwargs_cats[cat]
        )

        if cat in elinestyle_cats and elinestyle_cats[cat] is not None:
            eb[-1][0].set_linestyle(elinestyle_cats[cat])

    

    # x and y labels
    ax.set_xlabel(xlabel, fontsize=25)
    ax.set_ylabel(ylabel, fontsize=25)

    # Ticks
    ax.set_yticks(y)
    ax.set_yticklabels(yticks_labels)

    # Legend, grid
    pt.fix_plot(ax, show_leg=False)
    pt.show_grid(ax)

    if zero_line:
        ax.axvline(0, color='k', linestyle='--', alpha=0.2)
    ax.legend(bbox_to_anchor=(0.5, 1), loc='lower center', fontsize=20)
    if title is not None:
        fig.suptitle(title, fontsize=25)
        
    # Save the figure    
    if fig_name is not None:
        pt.save_fig(
            fig, 
            fig_name=fig_name, 
            folder_name=folder_name
        )



def plot_matrix(matrix, param_names, latex_params, fig_name=None, folder_name=None, save_fig=True, title=None):
    fig, ax1 = plt.subplots(ncols=1, figsize=(12, 10))  # 1 plot

    opts = {'cmap': plt.get_cmap("RdBu"),  # red blue color mode
            'vmin': -1, 'vmax': +1}  # correlation between -1 and 1
    mask = np.tri(matrix.shape[0], k=-1)
    matrix = np.transpose(np.ma.array(matrix, mask=mask))
    
    heatmap1 = ax1.pcolor(matrix, **opts)  # create a pseudo color plot
    cbar = plt.colorbar(heatmap1, ax=ax1)  # color bar
#     ticklabs = cbar.ax.get_yticklabels()
#     cbar.ax.set_yticklabels(ticklabs, fontsize=20)
    cbar.ax.tick_params(labelsize=20)

#     title = "Correlations"
    if title is not None:
        ax1.set_title(title, fontsize=25)

    labels = [None] * len(param_names)
    for i, param_name in enumerate(param_names):
        latex_branch = latex_params[param_name]['latex']
        labels[i] = latex_branch
    # shift location of ticks to center of the bins
    ax1.set_xticks(np.arange(len(labels)) + 0.5, minor=False)
    ax1.set_yticks(np.arange(len(labels)) + 0.5, minor=False)
    ax1.set_xticklabels(labels, minor=False, ha='right', rotation=70, fontsize=20)
    ax1.set_yticklabels(labels, minor=False, fontsize=20)
    for (i, j), z in np.ndenumerate(matrix):
        if i > j:
            ax1.text(j+0.5, i+0.5, '{:0.2f}'.format(z), ha='center', va='center', fontsize=15)
    
    plt.tight_layout() 
    if fig_name is not None:
        pt.save_fig(
            fig, 
            fig_name=fig_name, 
            folder_name=folder_name
        )

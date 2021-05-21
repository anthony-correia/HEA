"""
Functions to plot the parameters
"""

import HEA.plot.tools as pt
import numpy as np
import matplotlib.pyplot as plt

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
        print(kwargs_cats[cat])
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
    
    # Save the figure    
    if fig_name is not None:
        pt.save_fig(
            fig, 
            fig_name=fig_name, 
            folder_name=folder_name
        )
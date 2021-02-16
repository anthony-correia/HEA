"""
* Plot histograms
* Plot 2D histograms
* Plot scatter plots
* Plot histograms of the quotient of 2 branches
"""


import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame
from pandas.core.series import Series


from matplotlib.colors import LogNorm #matplotlib.colors.LogNorm()

import HEA.plot.tools as pt
from HEA.tools.da import add_in_dic, el_to_list

from HEA.config import default_fontsize
from HEA.tools import string

#Gives us nice LaTeX fonts in the plots
from matplotlib import rc, rcParams
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False



#################################################################################################
################################ subfunctions for plotting ######################################
################################################################################################# 

def get_bin_width(low, high, n_bins):
    """return bin width
    
    Parameters
    ----------
    low : Float
        low value of the range
    high : Float
        high value of the range
    n_bins : Int
        number of bins in the given range
    
    Returns
    -------
    Float
        Width of the bins
    """
    return float((high-low)/n_bins)


def get_count_err(data, n_bins, low, high, weights=None):
    """ get counts and error for each bin
    
    Parameters
    ----------
    data          : pandas.Series
        data to plot
    n_bins        : int
        number of bins
    low           : float
        low limit of the distribution
    high          : float
        high limit of the distribution  
    weights       : pandas.Series, numpy.array
        weights of each element in data
    
    Returns
    -------
    counts  : np.array
        number of counts in each bin
    edges   : np.array
        Edges of the bins
    centres : np.array
        Centres of the bins
    err     : np.array
        Errors in the count, for each bin
    """
    counts, edges = np.histogram(data, range = (low,high), bins=n_bins, weights=weights)
    centres = (edges[:-1] + edges[1:])/2.    
    err = np.sqrt(counts)
    
    return counts, edges, centres, err

def plot_hist_alone(ax, data, n_bins, low, high, 
                    color, bar_mode=True, alpha=1, 
                    density=False, 
                    label=None, show_ncounts=False, 
                    weights=None,
                    orientation='vertical', 
                    **params):
    """  Plot histogram
    
    Parameters
    ----------
    * If ``bar_mode``: Points with error bars
    * Else: histogram with bars
    
    ax            : matplotlib.axes.Axes
        axis where to plot
    data          : pandas.Series
        data to plot
    n_bins        : int
        number of bins
    low           : float
        low limit of the distribution
    high          : float
        high limit of the distribution
    color         : str
        color of the distribution
    bar_mode     : bool
    
        * if ``True``, plot with bars
        * else, plot with points and error bars
        
    alpha         : float between 0 and 1
        transparency of the bar histogram
    density       : bool
        if ``True``, divide the numbers of counts in the histogram by the total number of counts
    label         : str
        label of the histogram
    show_ncounts : bool
        if True, show the number of counts in each dataset of ``dfs``
    weights       : pandas.Series, numpy.array
        weights of each element in data    
    orientation  : 'vertical' or 'horizontal'
        orientation of the histogram
    **params     : dict
        parameters passed to the ``ax.bar``, or ``ax.barh`` or ``ax.errorbar`` functions
    
    Returns
    -------
    counts  : np.array
        number of counts in each bin
    edges   : np.array
        Edges of the bins
    centres : np.array
        Centres of the bins
    err     : np.array
        Errors in the count, for each bin
    """
    
    counts, edges, centres, err = get_count_err(data, n_bins, low, high, weights=weights)
    bin_width = get_bin_width(low,high,n_bins)
    n_candidates = counts.sum()
    
    if show_ncounts:
        if label is None:
            label = ""
        else:
            label += ": "
        label += f" {n_candidates} events"
        
    if density:
        counts = counts/(n_candidates*bin_width)
        err = err/(n_candidates*bin_width)
    
    if bar_mode:
        if orientation=='vertical':
            ax.bar(centres, counts, centres[1]-centres[0], color=color, alpha=alpha, edgecolor=None, label=label,
                  **params)
            ax.step(edges[1:],counts, color=color)
        elif orientation=='horizontal':
            ax.barh(centres, counts, centres[1]-centres[0], color=color, alpha=alpha, edgecolor=None, label=label,
                  **params)
            ax.step(counts, edges[:-1], color=color)
    else:
        if orientation=='vertical':
            ax.errorbar(centres, counts, yerr=err, color=color, ls='', marker='.', label=label) 
        elif orientation=='horizontal':
            ax.errorbar(counts, centres, xerr=err, color=color, ls='', marker='.', label=label) 
    if orientation=='vertical':     
        ax.set_xlim(low,high)
    elif orientation=='horizontal':     
        ax.set_ylim(low,high)
        
    return counts, edges, centres, err


### Set labels -------------------------------------------------------------------

def set_label_candidates_hist (ax, bin_width, pre_label, unit=None, 
                               fontsize=default_fontsize['label'], axis='y'):
    """ set the typical y-label of a 1D histogram
    
    Parameters
    ----------
    ax            : matplotlib.axes.Axes
        axis where to plot the label
    bin_width     : float
        bin width of the histogram
    pre_label     : str
        Label to put before showing the width of the bins (e.g., "Number of candidates", "proportion of candidates")
    unit          : str
        unit of the branch that was plotted
    fontsize      : float
        fontsize of the label
    axis          : ``'x'`` or ``'y'`` or ``'both'``
        axis where to set the label
    """
    
    
    label = f"{pre_label} / ({bin_width:.3g}{pt._unit_between_brackets(unit, show_bracket=False)})"
    if axis == 'x' or axis == 'both':
        ax.set_xlabel(label, fontsize=fontsize)
    if axis == 'y' or axis == 'both':
        ax.set_ylabel(label, fontsize=fontsize)

        
    
def set_label_hist(ax, latex_branch, unit, bin_width, 
                   density=False, data_name=None, 
                   fontsize=default_fontsize['label'],
                   orientation='vertical'):
    """ Set the xlabel and ylabel of a 1D histogram
    
    Parameters
    ----------    
    ax            : matplotlib.axes.Axes
        axis where to show the label
    latex_branch  : str
        latex name of the branch that was plotted
    unit          : str
        unit of the branch that was plotted
    bin_width     : float
        bin width of the histogram
    density       : bool
        If ``True``, the ylabel will be "Proportion of candidates" instead of "candidates'
    data_name     : str or None
        Name of the data, in case in needs to be specified in the label of the axis between parentheses
    fontsize      : float
        fontsize of the labels
    orientation  : 'vertical' or 'horizontal'
        orientation of the histogram
    """
    axis = {}
    if orientation=='vertical':
        axis['x'] = 'x'
        axis['y'] = 'y'
    elif orientation=='horizontal':
        axis['x'] = 'y'
        axis['y'] = 'x'
    
    
    #Set the x label
    fontsize_x = fontsize
    if len(latex_branch) > 50:
        fontsize_x -= 7
    pt.set_label_branch(ax, latex_branch, unit=unit, 
                       data_name=data_name, fontsize=fontsize_x, axis=axis['x'])
    
    pre_label = "Proportion of candidates" if density else "Candidates"
    
    
    set_label_candidates_hist(ax, bin_width, pre_label=pre_label, unit=unit, 
                              fontsize=fontsize, axis=axis['y'])

def set_label_2Dhist(ax, latex_branches, units,
                     fontsize=default_fontsize['label']):
    """ Set the xlabel and ylabel of a 2D histogram
    
    Parameters
    ----------
    ax            : matplotlib.axes.Axes
        axis where to plot
    latex_branches : [str, str]
        latex names of the branches that were plotted
    units         : [str, str]
        units of the branches that were plotted 
    fontsize       : float
        fontsize of the label
    """
        
    pt.set_label_branch(ax, latex_branches[0], unit=units[0], fontsize=fontsize, axis='x')
    pt.set_label_branch(ax, latex_branches[1], unit=units[1], fontsize=fontsize, axis='y')
    
def set_label_divided_hist(ax, latex_branch, unit, bin_width, names_data, fontsize=default_fontsize['label']):
    """ 
    Set the xlabel and ylabel of a "divided" histogram
    
    Parameters
    ----------
    ax            : matplotlib.axes.Axes
        axis where to plot
    latex_branches : [str, str]
        latex names of the branches that were plotted
    unit          : str
        unit of the quantity that was plotted
    bin_width     : float
        bin width of the histogram
    names_data    : [str, str]
        list of the 2 names of the data (for which a common branch was divided)
    fontsize      : float
        fontsize of the label
    
    """
    
    #Set the x label
    pt.set_label_branch(ax, latex_branch, unit=unit, 
                       data_name=None, fontsize=fontsize, axis='x')
        
    pre_label = ("candidates[%s] / candidates[%s] \n")%(names_data[0],names_data[1])
    
    set_label_candidates_hist(ax, bin_width, pre_label=pre_label, 
                              unit=unit, fontsize=25, axis='y')
  

    

#################################################################################################
################################# Main plotting function ########################################
################################################################################################# 


def end_plot_function(fig, save_fig=True, fig_name=None, folder_name=None, default_fig_name=None, ax=None):
    """ tight the layout and save the file or just return the ``matplotlib.figure.Figure`` and ``matplotlib.axes.Axes``
    
    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure of the plot
    fig_name : str
        name of the file that will be saved
    folder_name : str
        name of the folder where the figure will be saved
    default_fig_name : str
        name of the figure that will be saved, in the case ``fig_name`` is ``None``
    ax : matplotlib.figure.Axes
        Axis of the plot
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``axis_mode`` is ``False``)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``axis_mode`` is ``False``)
    """
    plt.tight_layout()
    
    if save_fig and fig is not None:
        pt.save_fig(fig, fig_name=fig_name, folder_name=folder_name, default_fig_name=default_fig_name)
    
    if fig is not None:
        return fig, ax

    
def get_fig_ax(ax=None, orientation='vertical'):
    """ Return a figure and an axis in the case where ``ax`` is ``None``
    
    Parameters
    ----------
    ax            : matplotlib.axes.Axes
        default axis
    orientation  : 'vertical' or 'horizontal'
        orientation of the plot:
        
        * ``'vertical'``: figure size is (8, 6)
        * ``'horizontal'``: figure size is (6, 8)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` was None)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``axis_mode`` is ``False``)   

    """
    if ax is None:
        if orientation=='vertical':
            fig, ax = plt.subplots(figsize=(8,6))
        elif orientation=='horizontal':
            fig, ax = plt.subplots(figsize=(6,8))
    else:
        save_fig = False
        fig = None
    
    return fig, ax
    
    
def plot_hist(dfs, branch, latex_branch=None, unit=None, weights=None,
              low=None, high=None, n_bins=100, colors=None, alpha=None,
              bar_mode=False, density=None, orientation='vertical',
              title=None,pos_text_LHC=None,
              fig_name=None, folder_name=None, 
              fontsize_label=default_fontsize['label'],
              save_fig=True, ax=None, 
              factor_ymax=None, 
              show_leg=None, loc_leg='best', 
              **params):
    """ Save the histogram(s) of branch of the data given in ``dfs``
    
    Parameters
    ----------
    dfs             : dict(str:pandas.Dataframe)
        Dictionnary {name of the dataframe : pandas dataframe}
    branch          : str
        name of the branch in the dataframe
    latex_branch    : str
        Latex name of the branch (for the labels of the plot)
    unit            : str
        Unit of the physical quantity
    weights         : numpy.array
        weights passed to plt.hist 
    low             : float
        low value of the distribution
    high            : float
        high value of the distribution
    n_bins          : int
        Desired number of bins of the histogram
    colors          : str or list(str)
        color(s) used for the histogram(s)
    alpha           : str or list(str)
        transparancy(ies) of the histograms
    bar_mode       : bool
        if True, plot with bars, else, plot with points and error bars
    density         : bool
        if True, divide the numbers of counts in the histogram by the total number of counts
    orientation     : 'vertical' or 'horizontal'
        orientation of the histogram
    title           : str
        title of the figure to show at the top of the figure
    pos_text_LHC    : dict, list or str
        passed to :py:func:`HEA.plot.tools.set_text_LHCb` as the ``pos`` argument.
    fig_name       : str
        name of the saved figure
    folder_name     : str
        name of the folder where to save the figure
    fontsize_label  : float
        fontsize of the label
    save_fig        : bool
        specifies if the figure is saved
    factor_ymax     : float
        multiplicative factor of ymax
    ax            : matplotlib.axes.Axes
        axis where to plot
    show_leg        : bool
        True if the legend needs to be shown
    loc_leg         : str
        location of the legend    
    **params       : dict
        passed to :py:func:`plot_hist_alone`
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    """
    if not isinstance(dfs,dict):
        dfs = {"" : dfs}
    
    if density is None:
        density = len(dfs)>1 # if there are more than 2 histograms
      
    
    
    fig, ax = get_fig_ax(ax, orientation)
    
    
    if isinstance(dfs,dict):
        data_names = list(dfs.keys())
    
    if latex_branch is None:
        latex_branch = string._latex_format(branch)
        
    ax.set_title(title, fontsize=fontsize_label)
    
    #First loop to determine the low and high value
    low, high = pt._redefine_low_high(low,high,[df[branch] for df in dfs.values()])
    bin_width = get_bin_width(low,high,n_bins)
    
    # colors
    if colors is None:
        colors = ['r','b','g','k']
    if not isinstance(colors,list):
        colors = [colors]
    
    
    weights = el_to_list(weights,len(dfs))
    alpha   = el_to_list(alpha  ,len(dfs))
    
    for i, (data_name, df) in enumerate(dfs.items()):
        if alpha[i] is None:
            alpha[i] = 0.5 if len(dfs)>1 else 1
        _,_,_,_ = plot_hist_alone(ax, df[branch], n_bins, low, high, colors[i], bar_mode, 
                                  alpha=alpha[i], density=density, label=data_name, weights=weights[i],
                                  orientation=orientation,
                                  **params)
              
              
    #Some plot style stuff
    if factor_ymax is None:
        factor_ymax = 1 + 0.15 * len(data_names)
        
    if show_leg is None:
        show_leg = len(dfs)>1
        
    set_label_hist(ax, latex_branch, unit, bin_width, density=density, fontsize=fontsize_label, 
                   orientation=orientation)
    
    if orientation=='vertical':
        axis_y = 'y'
    elif orientation=='horizontal':
        axis_y = 'x'
    pt.fix_plot(ax, factor_ymax=factor_ymax, show_leg=show_leg, pos_text_LHC=pos_text_LHC, loc_leg=loc_leg, axis=axis_y)
    
    return end_plot_function(fig, save_fig=save_fig, fig_name=fig_name, folder_name=folder_name,
                      default_fig_name=f'{branch}_{string.list_into_string(data_names)}', 
                      ax=ax)



def plot_hist_var (datas, branch, latex_branch=None, unit=None, data_names=None, **kwargs):
    """ plot the histogram(s) of data
    
    Parameters
    ----------
    datas      : pandas.Series or list(pandas.Series)
        dataset or list of datasets (a dataset is an array of float)
    branch     : str
        name of the branch, used to cook the name of the figure that will be saved
    latex_branch    : str
        Latex name of the branch (for the labels of the plot)
    unit            : str
        Unit of the physical quantity
    data_names : str or list(str)
        name of the datasets
    **kwargs   : dict
        passed to plot_hist
   
   Returns
   -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    """
    
    
    # data into list of data
    if not isinstance(datas[0], np.ndarray) and not isinstance(datas[0], Series):
        datas = [datas]
        
    data_names = el_to_list(data_names, len(datas))
    assert len(data_names) == len(datas)
    
    dfs = {}
    for data, data_name in zip(datas, data_names):
        df = DataFrame()
        df[branch] = np.array(data)
        dfs[data_name] = df
    
    return plot_hist(dfs, branch, latex_branch=latex_branch, unit=unit, **kwargs)   

def plot_divide(dfs, branch, latex_branch, unit, low=None, high=None, n_bins=100, 
                fig_name=None, folder_name=None, 
                save_fig=True, ax=None,
                pos_text_LHC=None):
    """ plot the (histogram of the dataframe 1 of branch)/(histogram of the dataframe 1 of branch) after normalisation
    
    Parameters
    ----------
    
    dfs             : dict(str:pandas.Dataframe)
        Dictionnary {name of the dataframe : pandas dataframe}
    branch          : str
        name of the branch in the dataframe
    latex_branch    : str
        Latex name of the branch (for the labels of the plot)
    unit            : str
        Unit of the physical quantity
    low             : float
        low value of the distribution
    high            : float
        high value of the distribution
    n_bins          : int
        Desired number of bins of the histogram
    fig_name       : str
        name of the saved figure
    folder_name     : str
        name of the folder where to save the figure
    save_fig        : bool
        specifies if the figure is saved
    ax            : matplotlib.axes.Axes
        axis where to plot
    pos_text_LHC    : dict, list or str
        passed to :py:func:`HEA.plot.tools.set_text_LHCb` as the ``pos`` argument.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    """
       
    
    fig, ax = get_fig_ax(ax)
    data_names = list(dfs.keys())
    
    
    # Compute the number of bins
    low, high = pt._redefine_low_high(low,high,[df[branch] for df in dfs.values()])
    bin_width = get_bin_width(low, high,n_bins)
    
    # Make the histogram, and get the bin centres and error on the counts in each bin
    list_dfs = list(dfs.values())
    names_data = list(dfs.keys())
    
    counts1, bin_edges = np.histogram(list_dfs[0][branch], n_bins, range=(low,high))
    counts2, _ = np.histogram(list_dfs[1][branch], n_bins, range=(low,high))
    bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
    
    err1 = np.sqrt(counts1)
    err2 = np.sqrt(counts2)
    
    #division
    with np.errstate(divide='ignore', invalid='ignore'): 
        division = counts1*counts2.sum()/(counts2*counts1.sum()) 
    err = division*np.sqrt((err1/counts1)**2+(err2/counts2)**2)

    ax.errorbar(bin_centres, division, yerr=err, fmt='o', color='k')
    ax.plot([low,high], [1.,1.], linestyle='--', color='b',marker='')
    
    
    # Labels
    set_label_divided_hist(ax, latex_branch, unit, bin_width, names_data, fontsize=25)    

    # Set lower and upper range of the x and y axes
    pt.fix_plot(ax, factor_ymax=1.1, show_leg=False, fontsize_ticks=20., ymin_to0=False, pos_text_LHC=pos_text_LHC)
    
    # Save
    return end_plot_function(fig, save_fig=save_fig, fig_name=fig_name, folder_name=folder_name,
                      default_fig_name=f"{branch.replace('/','d')}_{string.list_into_string(data_names,'_d_')}", 
                      ax=ax)


def plot_hist2d(df, branches, latex_branches, units, 
                low=None, high=None, n_bins=100, 
                log_scale=False, title=None, 
                fig_name=None, folder_name=None,
                data_name=None,
                save_fig=True, ax=None,
                pos_text_LHC=None):
    """  Plot a 2D histogram of 2 branches.
    
    Parameters
    ----------
    df                : pandas.Dataframe
        Dataframe that contains the 2 branches to plot
    branches          : [str, str]
        names of the two branches
    latex_branches    : [str, str]
        latex names of the two branches
    units             : str or [str, str]
        Common unit or list of two units of the two branches
    n_bins            : int or [int, int]
        number of bins
    log_scale         : bool
        if true, the colorbar is in logscale
    low               : float or [float, float]
        low  value(s) of the branches
    high              : float or [float, float] 
        high value(s) of the branches
    title             : str
        title of the figure
    fig_name       : str
        name of the saved figure
    folder_name     : str
        name of the folder where to save the figure
    data_name         : str
        name of the data, this is used to define the name of the figure,
        in the case ``fig_name`` is not defined.
    save_fig        : bool
        specifies if the figure is saved
    ax            : matplotlib.axes.Axes
        axis where to plot
    pos_text_LHC    : dict, list or str
        passed to :py:func:`HEA.plot.tools.set_text_LHCb` as the ``pos`` argument.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    """
    
    
    ## low, high and units into a list of size 2
    low = el_to_list(low,2)
    high = el_to_list(high,2)
    
    units = el_to_list(units,2)
    
    for i in range(2):
        low[i],high[i] = pt._redefine_low_high(low[i],high[i], df[branches[i]])
    
    ## Plotting
    fig, ax = get_fig_ax(ax)
    
    title = string.add_text(data_name, title, default=None)
    
    ax.set_title(title, fontsize=25)
    
    if log_scale:
        _,_,_,h = ax.hist2d(df[branches[0]], df[branches[1]], 
                             range = [[low[0],high[0]],[low[1],high[1]]],bins = n_bins,norm=LogNorm())
    else:
        _,_,_,h = ax.hist2d(df[branches[0]], df[branches[1]], 
                             range = [[low[0],high[0]],[low[1],high[1]]],bins = n_bins)
    
    ## Label, color bar
    pt.set_label_ticks(ax)
    pt.set_text_LHCb(ax, pos=pos_text_LHC)
    
    set_label_2Dhist(ax, latex_branches, units, fontsize=25)
    cbar = plt.colorbar(h)
    cbar.ax.tick_params(labelsize=20)
    
    return end_plot_function(fig, save_fig=save_fig, fig_name=fig_name, folder_name=folder_name,
                      default_fig_name=string.add_text(string.list_into_string(branches,'_vs_'),data_name,'_'), 
                      ax=ax)
    
def plot_scatter2d(dfs, branches, latex_branches, units=[None, None], 
                   low=None, high=None, n_bins=100, 
                   colors=['g', 'r', 'o', 'b'],
                   data_name=None,
                   title=None, 
                   fig_name=None, folder_name=None, 
                   fontsize_label=default_fontsize['label'],
                   save_fig=True, ax=None, get_sc=False,
                   pos_text_LHC=None, **params):
    """  Plot a 2D histogram of 2 branches.
    
    Parameters
    ----------
    dfs               : pandas.Dataframe or list(pandas.Dataframe)
        Dataset or list of datasets.
    branches          : [str, str]
        names of the two branches
    latex_branches    : [str, str]
        latex names of the two branches
    units             : str or [str, str]
        Common unit or list of two units of the two branches
    n_bins            : int or [int, int]
        number of bins
    log_scale         : bool
        if true, the colorbar is in logscale
    low               : float or [float, float]
        low  value(s) of the branches
    high              : float or [float, float] 
        high value(s) of the branches
    data_name         : str
        name of the data, this is used to define the name of the figure,
        in the case ``fig_name`` is not defined, and define the legend if there is more than 1 dataframe.
    colors            : str or list(str)
        color(s) used for the histogram(s)
    title             : str
        title of the figure
    fig_name          : str
        name of the saved figure
    folder_name       : str
        name of the folder where to save the figure
    fontsize_label    : float
        fontsize of the label of the axes
    save_fig          : bool
        specifies if the figure is saved
    ax            : matplotlib.axes.Axes
        axis where to plot
    get_sc            : bool
        if True: get the scatter plot
    pos_text_LHC    : dict, list or str
        passed to :py:func:`HEA.plot.tools.set_text_LHCb` as the ``pos`` argument.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    scs : matplotlib.PathCollection or list(matplotlib.PathCollection)
        scatter plot or list of scatter plots (only if ``get_sc`` is ``True``)
    """
    
    ## low, high and units into a list of size 2
    low = el_to_list(low,2)
    high = el_to_list(high,2)
    
    units = el_to_list(units,2)
    
    if ax is not None:
        save_fig=False
        
    fig, ax = get_fig_ax(ax)
        
    title = string.add_text(None, title, default=None)
    
    ax.set_title(title, fontsize=25)
    
    scs = [None]*len(dfs)
    for k, (data_name, df) in enumerate(dfs.items()):
        scs[k] = ax.scatter(df[branches[0]], df[branches[1]], 
                   c=colors[k], label=data_name, **params)
    if len(scs)==1:
        scs = scs[0]
        
    ax.set_xlim([low[0],high[0]])
    ax.set_ylim([low[1],high[1]])

    
    ## Label, color bar
    pt.set_label_ticks(ax)
    pt.set_text_LHCb(ax, pos=pos_text_LHC)
    
    set_label_2Dhist(ax, latex_branches, units, fontsize=fontsize_label)
    
    ## Save the data
    if save_fig:
        pt.save_fig(fig, fig_name, folder_name, 
                     string.add_text(string.list_into_string(branches,'_vs_'), 
                                 string.list_into_string(data_name,'_'),'_'))
    
    if fig is not None:
        if get_sc:
            return fig, ax, scs
        else:
            return fig, ax
    else:
        if get_sc:
            return scs

#################################################################################################
##################################### Automatic label plots #####################################
#################################################################################################         


    
def plot_hist_auto(dfs, branch, cut_BDT=None, **kwargs):
    """ Retrieve the latex name of the branch and unit.
    Then, plot histogram with :py:func:`plot_hist`.
    
    Parameters
    ----------
    
    dfs             : dict(str:pandas.Dataframe)
        Dictionnary {name of the dataframe : pandas dataframe}
    cut_BDT         : float or str
        ``BDT > cut_BDT`` cut. Used in the name of saved figure.
    branch          : str
        branch (for instance: ``'B0_M'``), which should be in the dataframe(s)
    **kwargs        : dict
        arguments passed in :py:func:`plot_hist` (except ``branch``, ``latex_branch`` and ``unit``)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    """
    
    ## Retrieve particle name, and branch name and unit.
#     particle, var = retrieve_particle_branch(branch)
    
#     name_var = branches_params[var]['name']
#     unit = branches_params[var]['unit']
#     name_particle = particle_names[particle]
    
    latex_branch, unit = pt.get_latex_branches_units(branch)
    data_names = string.list_into_string(list(dfs.keys()))
    
    add_in_dic('fig_name', kwargs)
    add_in_dic('title', kwargs)
    kwargs['fig_name'] = pt._get_fig_name_given_BDT_cut(fig_name=kwargs['fig_name'], cut_BDT=cut_BDT, 
                                                        branch=branch, data_name=data_names)
    kwargs['title'] = pt._get_title_given_BDT_cut(title=kwargs['title'], cut_BDT=cut_BDT)

    # Name of the folder = list of the names of the data
    pt._set_folder_name_from_data_name(kwargs, data_names)
    
    return plot_hist(dfs, branch, latex_branch, unit, **kwargs)


def plot_divide_auto(dfs, branch, **kwargs): 
    """Retrieve the latex name of the branch and unit. Set the folder name to the name of the datasets.
    Then, plot a "divide" histogram with :py:func:`plot_divide`.
    
    Parameters
    ----------
    
    dfs             : dict(str:pandas.Dataframe)
        Dictionnary {name of the dataframe : pandas dataframe}
    branch          : str
        name of the branch in the dataframe
    kwargs          : dict
        arguments passed in :py:func:`plot_divide` (except ``branch``, ``latex_branch`` and ``unit``)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    """
    latex_branch, unit = pt.get_latex_branches_units(branch)
    pt._set_folder_name_from_data_name(kwargs, list(dfs.keys()))
        
    return plot_divide(dfs, branch, latex_branch, unit, **kwargs)
    


    
def plot_hist2d_auto(df, branches, **kwargs):
    """  Retrieve the latex name of the branch and unit.
    Then, plot a 2d histogram with :py:func:`plot_hist2d`.
    
    Parameters
    ----------
    df        : pandas.Dataframe
        Dataframe that contains the branches
    branches  : [str, str]
        names of the two branches
    **kwargs  : dict
        arguments passed in :py:func:`plot_hist_2D` (except ``branches``, ``latex_branches`` and ``units``)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    """
    
    latex_branches, units = pt.get_latex_branches_units(branches)
    add_in_dic('data_name', kwargs)
    pt._set_folder_name_from_data_name(kwargs, kwargs['data_name'])
    
    return plot_hist2d(df, branches, latex_branches=latex_branches, units=units, **kwargs)

def plot_scatter2d_auto(dfs, branches, **kwargs):
    """ Retrieve the latex name of the branch and unit.
    Then, plot a scatter plot with :py:func:`plot_scatter2d`.
    
    Parameters
    ----------
    dfs               : pandas.Dataframe or list(pandas.Dataframe)
        Dataset or list of datasets.
    branches          : [str, str]
    **kwargs  : dict
        arguments passed in :py:func:`plot_scatter2d_auto` (except ``branches``, ``latex_branches`` and ``units``)
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    scs : matplotlib.PathCollection or list(matplotlib.PathCollection)
        scatter plot or list of scatter plots (only if get_sc is True)
    """
    
    pt._set_folder_name_from_data_name(kwargs, list(dfs.keys()))
    
    latex_branches, units = pt.get_latex_branches_units(branches)  
    return plot_scatter2d(dfs, branches, latex_branches=latex_branches, units=units, **kwargs)
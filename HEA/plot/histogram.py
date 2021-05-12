"""
* Plot histograms
* Plot 2D histograms
* Plot scatter plots
* Plot histograms of the quotient of 2 branches
"""


import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LogNorm  # matplotlib.colors.LogNorm()

import HEA.plot.tools as pt
from HEA.tools.da import add_in_dic, el_to_list

from HEA.config import default_fontsize
from HEA.tools import string
from HEA.tools.dist import (
    get_count_err, 
    get_bin_width, 
    get_count_err_ratio,
    divide_counts
)
from HEA.tools.assertion import is_list_tuple

from HEA.tools.df_into_hist import (
    _redefine_low_high,
    dataframe_into_hist1D,
    dataframe_into_hist2D
)

# Gives us nice LaTeX fonts in the plots
from matplotlib import rc, rcParams
rc('font', **{'family': 'serif', 'serif': ['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False


##########################################################################
################################ subfunctions for plotting ###############
##########################################################################

def get_edges_from_centres(centres):
    """ Get the bin edges of a histogram from
    the bin centres. The bins are supposed to be
    equal-sized.
    
    Parameters
    ----------
    centres: array-like
        Bin centres of a histogram
    
    Returns
    -------
    edges: array-like
        Bin edges of the histogram
    """
    bin_width = centres[1] - centres[0]
    edges = centres - bin_width / 2
    edges = np.append(edges, centres[-1] + bin_width / 2)
    
    return edges

def plot_hist_alone_from_hist(ax, counts, err=None, color='b',
                              edges=None,
                              centres=None,
                              bar_mode=True, alpha=1,
                              label=None, show_ncounts=False,
                              weights=None,
                              orientation='vertical',
                              edgecolor=None,
                              show_xerr=False,
                              linestyle='-',
                              linewidth=2,
                              **kwargs):
    """  Plot histogram
    
    * If ``bar_mode``: Points with error bars
    * Else: histogram with bars

    Parameters
    ----------
    ax            : matplotlib.axes.Axes
        axis where to plot
    counts  : np.array
        number of counts in each bin
    edges   : np.array
        Edges of the bins
    centres : np.array
        Centres of the bins
    err     : np.array
        Errors in the count, for each bin
    bar_mode     : bool

        * if ``True``, plot with bars
        * else, plot with points and error bars

    alpha         : float between 0 and 1
        transparency of the bar histogram
    label         : str
        label of the histogram
    show_ncounts : bool
        if True, show the number of counts in each dataset of ``dfs``
    weights       : pandas.Series, numpy.array
        weights of each element in data
    orientation  : 'vertical' or 'horizontal'
        orientation of the histogram
    **kwargs     : dict
        parameters passed to the ``ax.bar``, or ``ax.barh`` or ``ax.errorbar`` functions
    """

    n_bins = len(counts)
    
    centres, edges = get_centres_edges(centres, edges)
            
    low = edges[0]
    high = edges[-1]
    
    bin_widths = edges[1:] - edges[:-1]
    n_candidates = counts.sum()

    if show_ncounts:
        if label is None:
            label = ""
        else:
            label += ": "
        label += f" {n_candidates} events"
    
    
    if bar_mode:
        colors = el_to_list(color, 2)
        if orientation == 'vertical':
            if colors[0] is not None:
                ax.bar(centres, counts, bin_widths, 
                       color=colors[0], 
                       alpha=alpha, edgecolor=edgecolor, 
                       label=label,
                       **kwargs)
            if colors[1] is not None:
                ax.step(edges, np.concatenate([np.array([counts[0]]), counts]), 
                        color=colors[1],
                        label=label if colors[0] is None else None,
                        linestyle=linestyle, linewidth=linewidth,
                       )
        elif orientation == 'horizontal':
            if colors[0] is not None:
                ax.barh(centres, counts, edges[1:] - edges[:-1], color=colors[1], 
                        alpha=alpha, edgecolor=edgecolor, 
                        label=label if colors[0] is None else None,
                        **kwargs)
            if colors[1] is not None:
                ax.step(np.concatenate([counts, np.array([counts[-1]])]), edges, color=colors[2],
                       linestyle=linestyle, linewidth=linewidth)
    else:
        if show_xerr:
            xerr = bin_widths / 2
        else:
            xerr = None
        if orientation == 'vertical':
            ax.errorbar(centres, counts, xerr=xerr, yerr=err, color=color,
                        ls='', marker='.', label=label, **kwargs)
        elif orientation == 'horizontal':
            ax.errorbar(counts, centres, xerr=err, yerr=xerr, color=color,
                        ls='', marker='.', label=label, 
                        **kwargs)
    
    if orientation == 'vertical':
        ax.set_xlim(low, high)
    elif orientation == 'horizontal':
        ax.set_ylim(low, high)

    return counts, edges, centres, err      


def plot_hist_alone(ax, data, n_bins, low, high,
                    color, 
                    weights=None, density=False,
                    cumulative=False, quantile_bin=False,
                    **kwargs):
    """  Plot histogram
    
    * If ``bar_mode``: Points with error bars
    * Else: histogram with bars

    Parameters
    ----------
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
    weights       : pandas.Series, numpy.array
        weights of each element in data
    density       : bool
        if ``True``, divide the numbers of counts in the histogram by the total number of counts
    cumulative    : bool
        if ``True``, return the cumulated event counts.
    **kwargs     : dict
        parameters passed to the 
        :py:func:`plot_hist_alone_from_hist`

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

    counts, edges, centres, err = get_count_err(
        data, n_bins, low, high, weights=weights,
        cumulative=cumulative, density=density, 
        quantile_bin=quantile_bin)
    bin_width = get_bin_width(low, high, n_bins)

#     counts, err = get_density_counts_err(counts, bin_width, err, density=density)

    plot_hist_alone_from_hist(ax, edges=edges, counts=counts, 
                              err=err,
                              color=color,
                              centres=centres,
                              **kwargs)
    
    return counts, edges, centres, err


# Set labels -------------------------------------------------------------

def set_label_candidates_hist(ax, bin_width, pre_label, unit=None,
                              density=False,
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
    label = pre_label
    if density=="all" or density=="bin_width" or density==True:
        label = f"{label} / ({bin_width:.3g}{pt._unit_between_brackets(unit, show_bracket=False)})"
    if axis == 'x' or axis == 'both':
        ax.set_xlabel(label, fontsize=fontsize)
    if axis == 'y' or axis == 'both':
        ax.set_ylabel(label, fontsize=fontsize)


def set_label_hist(ax, latex_branch, unit, bin_width,
                   density=False, cumulative=False,
                   data_name=None,
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
        If ``True``, the ylabel will be "Proportion of candidates" instead 
        of "candidates'
    cumulative    : bool
        if ``True``, return the cumulated event counts.
    data_name     : str or None
        Name of the data, in case in needs to be specified in the label of the axis between parentheses
    fontsize      : float
        fontsize of the labels
    orientation  : 'vertical' or 'horizontal'
        orientation of the histogram
    
    """
    axis = {}
    if orientation == 'vertical':
        axis['x'] = 'x'
        axis['y'] = 'y'
    elif orientation == 'horizontal':
        axis['x'] = 'y'
        axis['y'] = 'x'

    # Set the x label
    fontsize_x = fontsize
    if len(latex_branch) > 50:
        fontsize_x -= 7
    pt.set_label_branch(ax, latex_branch, unit=unit,
                        data_name=data_name, fontsize=fontsize_x, axis=axis['x'])
    if density=='all' or density==True or density=='candidates':
        
        if cumulative:
            pre_label = "Cumulative density of candidates"
        else:
            pre_label = "Proportion of candidates"
    else: 
        if cumulative:
            
            pre_label = "Cumulative number of candidates"
        else: 
            pre_label = "Candidates"

    set_label_candidates_hist(ax, bin_width, pre_label=pre_label, unit=unit,
                              fontsize=fontsize, axis=axis['y'], density=density)


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

    pt.set_label_branch(
        ax, latex_branches[0], unit=units[0], fontsize=fontsize, axis='x')
    pt.set_label_branch(
        ax, latex_branches[1], unit=units[1], fontsize=fontsize, axis='y')


def set_label_divided_hist(ax, latex_branch, unit, bin_width,
                           data_names, fontsize=default_fontsize['label']):
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
    data_names    : [str, str]
        list of the 2 names of the data (for which a common branch was divided)
    fontsize      : float
        fontsize of the label

    """

    # Set the x label
    pt.set_label_branch(ax, latex_branch, unit=unit,
                        data_name=None, fontsize=fontsize, axis='x')

    pre_label = (
        "candidates[%s] / candidates[%s] \n") % (data_names[0], data_names[1])

    set_label_candidates_hist(ax, bin_width, pre_label=pre_label,
                              unit=unit, fontsize=25, axis='y')


##########################################################################
################################# Main plotting function #################
##########################################################################


def end_plot_function(fig, save_fig=True, fig_name=None,
                      folder_name=None, default_fig_name=None, ax=None):
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
        pt.save_fig(fig, fig_name=fig_name, folder_name=folder_name,
                    default_fig_name=default_fig_name)

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
        if orientation == 'vertical':
            fig, ax = plt.subplots(figsize=(8, 6))
        elif orientation == 'horizontal':
            fig, ax = plt.subplots(figsize=(6, 8))
    else:
        save_fig = False
        fig = None

    return fig, ax

def get_centres_edges(centres=None, edges=None):
    """ Get edges and centres if one of them is None
    
    Parameters
    ----------
    centres: array-like or None
        bin centres
    edges : array-like or None
        bin edges
        
    Returns
    -------
    centres: array-like
        bin centres
    edges : array-like
        bin edges
    """
    assert not (centres is None and edges is None) # at least one given
    if centres is None:
        centres = (edges[1:] + edges[:-1])/2

    if edges is None:
        edges = get_edges_from_centres(centres)
    
    return centres, edges

def get_bin_width_from_edges(edges):
    """
    """
    bin_widths = edges[1:] - edges[:-1]
    
    if np.allclose(bin_widths, bin_widths[0]):
        bin_width = bin_widths[0]
    else:
        # if non-uniform bin width
        # bin width not well defined
        bin_width = None
    return bin_width

def plot_hist_counts(dfs, branch, latex_branch=None, unit=None, weights=None,
              low=None, high=None, n_bins=100, colors=None, alpha=None,
              bar_mode=False, density=None, orientation='vertical',
              labels=None,
              title=None, pos_text_LHC=None,
              fig_name=None, folder_name=None,
              fontsize_label=default_fontsize['label'],
              save_fig=True, ax=None,
              factor_ymax=None,
              fontsize_leg=default_fontsize['legend'],
              show_leg=None, loc_leg='best',
              ymin_to_0=True,
              centres=None, edges=None,
              cumulative=False,
              quantile_bin=False,
              edgecolors=None,
              **kwargs):
    """ Produce a plot of histogram(s) from counts and 
    bin edges.

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
    bar_mode       : bool or list(bools)
        if True, plot with bars, else, plot with points and error bars
    density         : bool
        if True, divide the numbers of counts in the histogram by the total number of counts
    orientation     : 'vertical' or 'horizontal'
        orientation of the histogram
    labels          : str or list(str)
        labels to add to the histogram
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
    fontsize_leg    : float
        fontsize of the legend
    show_leg        : bool
        True if the legend needs to be shown
    loc_leg         : str
        location of the legend
    centres       : array-like
        if the histogram is plotted using directly counts and err.
    cumulative    : bool
        if ``True``, return the cumulated event counts.
    **kwargs       : dict
        passed to :py:func:`plot_hist_alone`

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    """
        
    centres, edges = get_centres_edges(centres, edges)
    
    low = edges[0]
    high = edges[-1]

    bin_width = get_bin_width_from_edges(edges)
    
        
    fig, ax = get_fig_ax(ax, orientation)

    if isinstance(dfs, dict):
        data_names = list(dfs.keys())

    if latex_branch is None:
        latex_branch = string._latex_format(branch)

    ax.set_title(title, fontsize=fontsize_label)

    # colors
    if colors is None:
        colors = ['r', 'b', 'g', 'k']
    if not isinstance(colors, list):
        colors = [colors]

    weights = el_to_list(weights, len(dfs))
    alpha = el_to_list(alpha, len(dfs))
    bar_mode = el_to_list(bar_mode, len(dfs))
    labels = el_to_list(labels, len(dfs))
    edgecolors = el_to_list(edgecolors, len(dfs))
    
    
    for i, (data_name, df) in enumerate(dfs.items()):
        if quantile_bin:
            if edgecolors[i] is None:
                edgecolors[i] = colors[i]
            if alpha[i] is None:
                alpha[i] = 0.7
        
        
        if alpha[i] is None:
            alpha[i] = 0.5 if len(dfs) > 1 else 1
        
        label = string.add_text(data_name, labels[i], sep='')
        
        plot_hist_alone_from_hist(ax, df[0], df[1], color=colors[i],
                              edges=edges,
                              bar_mode=bar_mode[i], alpha=alpha[i],
                              label=label, 
                              weights=weights[i],
                              orientation=orientation,
                              edgecolor=edgecolors[i],
                              centres=centres,
                              **kwargs)
        
        # Some plot style stuff
        if factor_ymax is None:
            factor_ymax = 1 + 0.15 * len(data_names)

        if show_leg is None:
            show_leg = len(dfs) > 1

        set_label_hist(ax, latex_branch, unit, bin_width, 
                       density=density, cumulative=cumulative,
                       fontsize=fontsize_label,
                       orientation=orientation)

        if orientation == 'vertical':
            axis_y = 'y'
        elif orientation == 'horizontal':
            axis_y = 'x'
        pt.fix_plot(ax, factor_ymax=factor_ymax, show_leg=show_leg,
                    fontsize_leg=fontsize_leg,
                    pos_text_LHC=pos_text_LHC, 
                    loc_leg=loc_leg, axis=axis_y,
                   ymin_to_0=ymin_to_0)

    return end_plot_function(fig, save_fig=save_fig, fig_name=fig_name, folder_name=folder_name,
                             default_fig_name=f'{branch}_{string.list_into_string(data_names)}',
                             ax=ax)

def plot_hist(dfs, branch, weights=None,
              low=None, high=None, n_bins=100, 
              density=None,
              cumulative=False,
              quantile_bin=False,
              **kwargs):
    """ Plot histogram(s) from dataframes

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
    bar_mode       : bool or list(bools)
        if True, plot with bars, else, plot with points and error bars
    
    orientation     : 'vertical' or 'horizontal'
        orientation of the histogram
    labels          : str or list(str)
        labels to add to the histogram
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
    fontsize_leg    : float
        fontsize of the legend
    show_leg        : bool
        True if the legend needs to be shown
    loc_leg         : str
        location of the legend
    centres       : array-like
        if the histogram is plotted using directly counts and err.
    cumulative    : bool
        if ``True``, return the cumulated event counts.
    **kwargs       : dict
        passed to :py:func:`plot_hist_alone`

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    """
    
    dfs_counts, edges, density = dataframe_into_hist1D(
        dfs=dfs, 
        low=low, high=high, n_bins=n_bins, weights=weights,
        density=density, cumulative=cumulative, 
        quantile_bin=quantile_bin, branch=branch)
    
    return plot_hist_counts(
        dfs=dfs_counts, branch=branch,
        density=density,
        cumulative=cumulative,
        quantile_bin=quantile_bin,
        edges=edges, **kwargs
    )

    

def plot_divide_alone(ax, data1, data2, 
                      low=None, high=None, 
                      n_bins=None, 
                      color='k', 
                      label=None, bin_centres=None,
                      edges=None,
                      show_xerr=False,
                      **kwargs):
    """ Plot a "divide" histogram
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        axis where to plot
    data1:  array-like or 2-list(array-like)
        Data 1 or couple (counts, err)
    data2: array-like
        Data 2 or couple (counts, err)
    n_bins: int
        number of bins in the histogram
    low: float
        low value of the histogram
    high: float
        high value of the histogram
    color: str
        color of the error bars
    label: str
        label of the plot (for the legend)
    **kwargs: dict[str:2-list]
        passed to :py:func:`HEA.tools.dist.get_count_err_ratio`
    
    """
    
    counts_provided = is_list_tuple(data1) and is_list_tuple(data2)
    
    if counts_provided:
        counts1, err1 = data1
        counts2, err2 = data2
        division, err = divide_counts(
            counts1, counts2, 
            err1=err1, err2=err2, 
            **kwargs
        )
                
    else:
        assert n_bins is not None
        assert low is not None
        assert high is not None
        
        division, bin_centres, err = get_count_err_ratio(
            data1=data1,
            data2=data2,
            n_bins=n_bins,
            low=low, high=high,
            **kwargs
        )
    
    if show_xerr:
        if edges is None:
            edges = get_edges_from_centres(centres)
        bin_widths = edges[1:] - edges[:-1]
        
        xerr = bin_widths / 2
    else:
        xerr = None
    
    
    ax.axhline(1., linestyle='--', color='b', marker='')
    ax.errorbar(bin_centres, division, yerr=err, xerr=xerr, fmt='o', 
                color=color, label=label)
    
    
    if counts_provided:
        return division, err
    else:
        return division, bin_centres, err
    

def plot_divide(dfs, branch, latex_branch, unit, low=None, high=None, 
                n_bins=100, color='k', label=None,
                fig_name=None, folder_name=None,
                save_fig=True, ax=None,
                pos_text_LHC=None, **kwargs):
    """ plot the (histogram of the dataframe 1 of branch)/(histogram of the dataframe 1 of branch) 
    after normalisation

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
    color           : str
        color of the plotted data points
    label           : str
        label of the plotted data points
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
    **kwargs   : dict[str:2-list]
        passed to :py:func:`HEA.tools.dist.get_count_err_ratio`
        
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
    low, high = _redefine_low_high(
        low, high, [df[branch] for df in dfs.values()])
    bin_width = get_bin_width(low, high, n_bins)

    # Make the histogram, and get the bin centres and error on the counts in
    # each bin
    list_dfs = list(dfs.values())
    data_names = list(dfs.keys())
    
    plot_divide_alone(
        ax,
        data1=list_dfs[0][branch],
        data2=list_dfs[1][branch],
        n_bins=n_bins,
        low=low, high=high,
        label=label,
        color=color,
        **kwargs
    )
    
    # Labels
    set_label_divided_hist(ax, latex_branch, unit,
                           bin_width, data_names, fontsize=25)

    # Set lower and upper range of the x and y axes
    pt.fix_plot(ax, factor_ymax=1.1, show_leg=False,
                fontsize_ticks=20., ymin_to_0=False, pos_text_LHC=pos_text_LHC)

    # Save
    return end_plot_function(fig, save_fig=save_fig, fig_name=fig_name, folder_name=folder_name,
                             default_fig_name=f"{branch.replace('/','d')}_{string.list_into_string(data_names,'_d_')}",
                             ax=ax)

    

def plot_hist2d_counts(branches, counts, xedges, yedges, 
                       latex_branches=[None, None], units=None,
                       log_scale=False, title=None,
                       fig_name=None, folder_name=None,
                       data_name=None,
                       save_fig=True, ax=None,
                       vmin=None, vmax=None,
                       pos_text_LHC=None, **kwargs):
    """  Plot a 2D histogram of 2 branches directly from the counts
    and the edges.

    Parameters
    ----------
    counts: 2d array-like
        bin counts
    xedges, yedges: array-like
        Bin edges
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
    units = el_to_list(units, 2)
    for i in range(2):
        if latex_branches[i] is None:
            latex_branches[i] = string._latex_format(branches[i])
        
    
    # Plotting
    fig, ax = get_fig_ax(ax)

    title = string.add_text(data_name, title, default=None)

    ax.set_title(title, fontsize=25)
    
    if log_scale:
        kwargs['norm'] = LogNorm(vmin=vmin, vmax=vmax)
    else:
        kwargs['vmin'] = vmin
        kwargs['vmax'] = vmax
        
    X, Y = np.meshgrid(xedges, yedges)
    pcm = ax.pcolormesh(X, Y, counts, **kwargs)
    cbar = fig.colorbar(pcm)
    cbar.ax.tick_params(labelsize=20)

    # Label, color bar
    pt.set_label_ticks(ax)
    pt.set_text_LHCb(ax, pos=pos_text_LHC)

    set_label_2Dhist(ax, latex_branches, units, fontsize=25)
    

    return end_plot_function(fig, save_fig=save_fig, fig_name=fig_name, folder_name=folder_name,
                             default_fig_name=string.add_text(
                                 string.list_into_string(branches, '_vs_'), data_name, '_'),
                             ax=ax)




def plot_hist2d(branches, df,
                low=None, high=None, n_bins=100,
                normalise=False,
                **kwargs):
    """  Plot a 2D histogram of 2 branches from a dataframe

    Parameters
    ----------
    df                : pandas.Dataframe or list(array-like)
        Dataframe that contains the 2 branches to plot
        or two same-sized arrays, one for each variable
    branches          : [str, str]
        names of the two branches
    n_bins            : int or [int, int]
        number of bins
    log_scale         : bool
        if true, the colorbar is in logscale
    low               : float or [float, float]
        low  value(s) of the branches
    high              : float or [float, float]
        high value(s) of the branches
    normalise: bool
        Normalised?
    **kwargs:
        passed to :py:func:`plot_hist2d_counts`

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    """

    # Plotting
    
    counts, xedges, yedges = dataframe_into_hist2D(
        branches, df, 
        low=low, high=high, 
        n_bins=n_bins,
        normalise=normalise
    )
    
    return plot_hist2d_counts(
        counts=counts, xedges=xedges, yedges=yedges,
        branches=branches,
        **kwargs
    )


def plot_scatter2d(dfs, branches, latex_branches, units=[None, None],
                   low=None, high=None, n_bins=100,
                   colors=['g', 'r', 'o', 'b'],
                   data_name=None,
                   title=None,
                   fig_name=None, folder_name=None,
                   fontsize_label=default_fontsize['label'],
                   save_fig=True, ax=None, get_sc=False,
                   pos_text_LHC=None, **kwargs):
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

    # low, high and units into a list of size 2
    low = el_to_list(low, 2)
    high = el_to_list(high, 2)

    units = el_to_list(units, 2)

    if ax is not None:
        save_fig = False

    fig, ax = get_fig_ax(ax)

    title = string.add_text(None, title, default=None)

    ax.set_title(title, fontsize=25)

    scs = [None] * len(dfs)
    for k, (data_name, df) in enumerate(dfs.items()):
        scs[k] = ax.scatter(df[branches[0]], df[branches[1]],
                            c=colors[k], label=data_name, **kwargs)
    if len(scs) == 1:
        scs = scs[0]

    ax.set_xlim([low[0], high[0]])
    ax.set_ylim([low[1], high[1]])

    # Label, color bar
    pt.set_label_ticks(ax)
    pt.set_text_LHCb(ax, pos=pos_text_LHC)

    set_label_2Dhist(ax, latex_branches, units, fontsize=fontsize_label)

    # Save the data
    if save_fig:
        pt.save_fig(fig, fig_name, folder_name,
                    string.add_text(string.list_into_string(branches, '_vs_'),
                                    string.list_into_string(data_name, '_'), '_'))

    if fig is not None:
        if get_sc:
            return fig, ax, scs
        else:
            return fig, ax
    else:
        if get_sc:
            return scs

##########################################################################
##################################### Automatic label plots ##############
##########################################################################


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
    
    centres = kwargs.get("centres")
    edges = kwargs.get("edges")
    if edges is not None or centres is not None:
        plot_function = plot_hist_counts
    else:
        plot_function = plot_hist
    

    # Retrieve particle name, and branch name and unit.
    latex_branch, unit = pt.get_latex_branches_units(branch)
    data_names = string.list_into_string(list(dfs.keys()))

    add_in_dic('fig_name', kwargs)
    add_in_dic('title', kwargs)
    kwargs['fig_name'] = pt._get_fig_name_given_BDT_cut(fig_name=kwargs['fig_name'], cut_BDT=cut_BDT,
                                                        branch=branch, data_name=data_names)
    kwargs['title'] = pt._get_title_given_BDT_cut(
        title=kwargs['title'], cut_BDT=cut_BDT)

    # Name of the folder = list of the names of the data
    pt._set_folder_name_from_data_name(kwargs, data_names)

    return plot_function(dfs, 
                     branch=branch, 
                     latex_branch=latex_branch, 
                     unit=unit, **kwargs)


def plot_divide_auto(dfs, branch, **kwargs):
    """Retrieve the latex name of the branch and unit. Set the folder name to the name of the datasets.
    Then, plot a "divide" histogram with :py:func:`plot_divide`.

    Parameters
    ----------

    dfs             : dict(str:pandas.Dataframe)
        Dictionnary {name of the dataframe : pandas dataframe}
    branch          : str
        name of the branch in the dataframe
    kwargs          :
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


def _core_plot_hist2d(branches, kwargs, suff=''):
    """ Fix few parameters before plotting
    a 2d histogram.
    
    * Get the latex branches and units
    * Change the figure name to ``branch1_vs_branch2_fit``
    * Set the ``folder_name`` to ``data_name``
    
    Parameters
    ----------
    branches: list(str, str)
        list of two branches that are going to be
        plotted
    kwargs: dict
        other parameters passed to the plotting function
    suff: str
        to be added at the end of the name of the figure
    
    Returns
    -------
    latex_branches: list(str, str)
        latex labels of each of the two branches
    unit: list(str, str)
        units of each of the two branches
    kwargs: dict
        other parameters passed to the plotting function,
        updated
    """
    
    latex_branches, units = pt.get_latex_branches_units(branches)
    add_in_dic('data_name', kwargs)
    add_in_dic('fig_name', kwargs)
    pt._set_folder_name_from_data_name(kwargs, kwargs['data_name'])
    if kwargs['fig_name'] is None:
        kwargs['fig_name'] = "_vs_".join(branches) + suff
    return latex_branches, units, kwargs

def plot_hist2d_auto(branches, *args, with_counts=False, **kwargs):
    """  Retrieve the latex name of the branch and unit.
    Then, plot a 2d histogram with :py:func:`plot_hist2d`.

    Parameters
    ----------
    df        : pandas.Dataframe
        Dataframe that contains the branches
    branches  : [str, str]
        names of the two branches
    with_counts: bool
        whether to use :py:func:`plot_hist2d_counts` (``True``)
        or ``plot_hist2d``. The latter is used
        to plot from a dataframe. The first is used to plot
        from the counts and edges of a histogram
    **kwargs  : dict
        arguments passed to the plotting function 
        (except ``branches``, ``latex_branches`` and ``units``)

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot (only if ``ax`` is not specified)
    ax : matplotlib.figure.Axes
        Axis of the plot (only if ``ax`` is not specified)
    """
    if with_counts:
        plot_function = plot_hist2d_counts
    else:
        plot_function = plot_hist2d

    latex_branches, units, kwargs = _core_plot_hist2d(branches, kwargs)
    

    return plot_function(
        branches, *args, latex_branches=latex_branches, units=units, **kwargs)


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
    
    return plot_scatter2d(
        dfs, branches, latex_branches=latex_branches, units=units, **kwargs)

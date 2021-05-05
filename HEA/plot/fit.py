"""
Plot a histogram, with the fitted PDF, the pull diagram and the fitted parameters.
"""

import HEA.plot.tools as pt
from HEA.plot.histogram import (
    plot_hist_alone, set_label_hist, get_bin_width,
    plot_hist_alone_from_hist,
    get_centres_edges,
    plot_hist2d_counts,
    _core_plot_hist2d,
)

import HEA.plot.histogram as h

from HEA.tools.da import add_in_dic, el_to_list, get_element_list
from HEA.tools import string, assertion
from HEA.tools import dist
from HEA.fit.params import get_ufloat_latex_from_param_latex_params
from HEA.fit import PDF
from HEA.config import default_fontsize
import HEA.plot.zfit as pz


#from zfit.core.parameter import Parameter

import numpy as np
from pandas import DataFrame
from uncertainties import ufloat


import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker

# Gives us nice LaTeX fonts in the plots
from matplotlib import rc, rcParams
rc('font', **{'family': 'serif', 'serif': ['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False


model_names_types = {
    'm': 'model',
    's': 'signal',
    'b': 'background',
    'n': None,
    '': None
}

# Alternative names for the models
name_PDF_dict = {
    'DoubleCB': 'Double CB',
    'SumPDF': 'Sum',
    'Exponential': 'Exponential',
    'Gauss': 'Gaussian',
    'CrystalBall': 'Crystal Ball',
    'Chebyshev': 'Chebyshev',
    'HORNS': 'HORNS',
    'HILL': 'HILL',
    None: None,
    'Chi2': r"$\chi^2$ PDF"
}

##########################################################################
#################################### Sub-plotting functions ##############
##########################################################################

def print_fit_info(edges, fit_counts, counts, pull, ndof, err=None):
    """ Show:
    
    * Number of bins and bin width
    * :math: Reduced :math:`\\chi^2`
    * Mean and std of the normalised residuals
    
    Parameters
    ----------
    centres: array-like
        centres of the bins
    fit_counts: array-like
        Number of counts as model by the fitted model
    counts: array-like
        counts in data
    pull: array-like
        pull diagram
    ndof: int
        number of d.o.f. in the model.
    """
    # Fit quality
    print(f"Number of bins: {len(edges)-1}")
    print(f"Width of the bins: {h.get_bin_width_from_edges(edges)}")
    print("")
    chi2 = dist.get_reduced_chi2(fit_counts, counts, ndof, err=err)
    print("Number of d.o.f. in the model: ", ndof)
    print('Reduced chi2: ', chi2)
    print("")
    print(f"Mean of the normalised residuals: {dist.get_mean(pull)}")
    print(f"Std of the normalised residuals: {dist.get_std(pull)}")

def plot_pull_diagram_from_hist(ax, fit_counts, counts,
                                edges, err=None, ndof=None,
                                y_line=3, bar_mode_pull=True,
                                fontsize=default_fontsize['label'], 
                                color='b', color_lines='r'):
    """
    Plot pull diagram of ``model`` compared to the data, given by (``counts``, ``centres``)


    Parameters
    ----------

    ax            : matplotlib.axes.Axes
        axis where to plot
    fit_counts: np.array(float)
        Number of counts as model by the fitted model
    counts        : np.array(float)
        counts of the bins given by centres in the histogram
    edges        : np.array(float)
        edges of the bins (if ``centres`` is not provided)
    err       : np.array(float)
        bin centres of the histogram
    ndof         : number of d.o.f. in the model
        Used to compute the some metrics of fit quality
    y_line : float
        Position of the lines at ``y = 0``, ``y = y_line`` and ``y = - y_line``
    bar_mode_pull: bool
        if ``True``, the pull diagram is plotted with bars instead of points + error bars
    fontsize      : float
        fontsize of the labels
    color         : str
        color of the pull diagram
    color_lines   : str
        color of the lines at ``y = 0``, ``y = y_line`` and ``y = - y_line``
    """

    centres = (edges[:-1] + edges[1:]) / 2.
    bin_widths = edges[1:] - edges[:-1]
    low = edges[0]
    high = edges[-1]
    
    if err is None:
        err = np.sqrt(counts)

    
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore divide-by-0 warning
        pull = np.divide(counts - fit_counts, err)
        

    # Plotting
    if bar_mode_pull:
        ax.bar(centres, pull, bin_widths, 
               color=color, edgecolor=None)
        ax.step(edges[1:], pull, color=color)
    else:
        ax.errorbar(centres, pull, yerr=np.ones(
            len(centres)), color=color, ls='', marker='.')
    ax.plot([low, high], [y_line, y_line], color='r', ls='--')
    ax.plot([low, high], [-y_line, -y_line], color='r', ls='--')
    ax.plot([low, high], [0, 0], color='r')

    # Symmetric pull diagram
    low_y, high_y = ax.get_ylim()
    maxi = max(4., -low_y, high_y)
    low_y = -maxi
    high_y = maxi

    ax.set_ylim([low_y, high_y])

    ## Label and ticks
    ax.set_ylabel('residuals / $\\sigma$', fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=20)
    ax.yaxis.set_major_locator(MultipleLocator(3.))
    ax.yaxis.set_minor_locator(MultipleLocator(1.))

    pt.show_grid(ax, which='minor', axis='y')

    ax.set_xlim([low, high])
    return pull
 

# FITTED CURVES ==========================================================


def plot_fitted_curve_from_hist(ax, x, fit_counts,
                                PDF_name=None,
                                model_name=None, model_type=None,
                                color='b',
                                linestyle='-',linewidth=2., 
                                alpha=1, 
                                mode=False, edges=None,
                                **kwargs):
    """
    Plot a fitted curve given by ``model``

    Parameters
    ----------
    ax            : matplotlib.axes.Axes
        axis where to plot the label
    x             : numpy.numpy(float)
        points of the x-axis where to model has been evaluated
    fit_counts: zfit.pdf.BasePDF
        Model evaluated at x
    PDF_name : str
        name of the PDF - used in the legend.
    model_name : str
        name of the models - used in the legend.
        If ``None``, the legend is not shown
    model_type  : str
        type of the model

        * ``'m'`` : model (sum) ; should always be the FIRST ONE !!
        * ``'s'`` : signal
        * ``'b'`` : background
        * ``'n'`` : write nothing
        
        used in the legend to indicate if it is a signal or a background component
    
    color         : str
        color of the line
    linewidth    : float
        width of the curve line
    linestyle     : str
        style of the line
    alpha         : float, between 0 and 1
        opacity of the curve
    edges : array-like
        bin edges. Used for non-uniform bin width
    **kwargs:
        passed to :py:func`ax.plot()`
    """
    
    if model_name is None:
        label = None
    else:
        label = string.add_text(name_PDF_dict[PDF_name], model_names_types[model_type], ' - ')
        label = string.add_text(label, model_name)  
    
    if mode=='fillbetween':
        return ax.fill_between(x, fit_counts, label=label,
                               color=color,
                               alpha=alpha, **kwargs)  
    elif mode=='bar':
        return plot_hist_alone_from_hist(ax, fit_counts, err=None,
                                           color=color,
                                           centres=x,
                                           bar_mode=True, alpha=alpha,
                                           label=label, 
                                           orientation='vertical',
                                           linestyle=linestyle,
                                           linewidth=linewidth,
                                           edges=edges,
                                          **kwargs)
        
    else:
        return ax.plot(x, fit_counts, linewidth=linewidth, color=color,
                       label=label,
                       ls=linestyle, alpha=alpha, **kwargs)




# RESULT FIT =============================================================


def plot_result_fit(ax, params, latex_params=None,
                    fontsize=default_fontsize['legend'], 
                    colWidths=[None, None, None, None], loc='upper right'):
    """
    Plot the results of the fit in a table

    Parameters
    ----------
    ax            : matplotlib.axes.Axes
        axis where to plot
    params        : dict
        dictionnary of the result of the fit.
        Associates to a fitted parameter (key)
        a dictionnary that contains its value (key: 'v')
        and its error(key: 'e')
    latex_params  :
        Dictionnary with the name of the params
        Also indicated the branchs to show in the table 
        among all the branchs in params
    fontsize      : float
        Fontsize of the text in the table
    colWidths     : [float, float, float, float]
        Width of the four columns of the table

        * first column: latex name of the parameter
        * second column: nominal value
        * third column: ``$\\pm$``
        * fourth column: error
    loc           : str
        location of the table
    """
    
    result_fit_table = []
    table_lines = {}
    
    for param_name in list(params.keys()):  # loop through the parameters      
        
        # if latex_params not None, it specifies the branchs we want to show
        if (latex_params is None) or (param_name in latex_params):
            # Retrieve value and error
            value_param = params[param_name]['v']
            error_param = params[param_name]['e']
            
            ufloat_value = ufloat(value_param, error_param)
            
            latex_param, ufloat_latex = get_ufloat_latex_from_param_latex_params(
                    param_name, 
                    ufloat_value, 
                    latex_params
                )
            
            latex_nominal, latex_error =  ufloat_latex.split('\\pm')
            
            line = [ latex_param, ":", latex_nominal, f'$\\pm~{latex_error}$']
            
            if latex_params is not None:
                # Table --> param_name   :   value_param +/- error_param
                table_lines[param_name] = line
            else:
                result_fit_table.append(line)
                
    # Create the table in the order given by ``latex_params``
    
    if latex_params is not None:
        for param_name in latex_params.keys():
            if param_name in table_lines:
                line = table_lines[param_name]
                result_fit_table.append(line)
    
        
    
    # Plot the table with the fitted parameters in the upper right part of the
    # plot
    
    index_auto_colWidths = []
    default_colWidths = [0.05, 0.01, 0.055, 0.1]

    new_colWidths = []
    for i in range(len(colWidths)):
        if colWidths[i] is None:
            index_auto_colWidths.append(i)
            new_colWidths.append(default_colWidths[i]) # dummy value
        else:
            new_colWidths.append(colWidths[i])

    table = ax.table(result_fit_table, loc=loc, edges='open', cellLoc='left',
                     colWidths=new_colWidths
                    )
    
    cells = table.get_celld()
    for i in range(len(result_fit_table)):
        cells[(i, 3)].PAD = 0
    table.AXESPAD = 0.01
    
    table.scale(1., 1.7)
    table.auto_set_font_size(False)
    
    table.set_fontsize(fontsize)
    table.auto_set_column_width(index_auto_colWidths)
    


##########################################################################
###################################### Plotting function #################
##########################################################################

def create_fig_plot_hist_fit(plot_pull):
    """ Create fig, axs for :py:func:`plot_hist_fit`
    
    Parameters
    ----------
    plot_pull: bool
        Do we need to plot pull diagram?
    
    Returns
    -------
    fig   : matplotlib.figure.Figure
        Figure of the plot (only if ``axis_mode`` is ``False``)
    ax   : list(matplotlib.figure.Axes)
        List of axes:
        
        * Axis of the histogram + fitted curves + table
        * Axis of the pull diagram (only if ``plot_pull`` is ``True``)
    """
    if plot_pull:
        fig = plt.figure(figsize=(12, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax = [plt.subplot(gs[i]) for i in range(2)]
    else:
        fig, ax = plt.subplots(figsize=(12, 10 * 2 / 3))
        ax = [ax]
    
    return fig, ax
    


def plot_hist_fit_counts(
    # Sample
    data, branch, latex_branch=None, unit=None,
    centres=None, edges=None, density=None,
    # models
    models=None, models_names=None, PDF_names=None, 
    models_types=None, ndof=0,
    data_label=None,
    # histogram style
    color='black', bar_mode=False,
    # models style
#     linewidth=2.5, 
    colors=None,
    stack=False,
    # pull
    bar_mode_pull=True,
    lim_pull=None, 
    color_pull=None,
    # plot style
    title=None, pos_text_LHC=None,
    # Show fitted parameters
    params=None, latex_params=None, 
    kwargs_res={},
    # Save the figure
    fig_name=None, folder_name=None, data_name=None,
    save_fig=True, 
    # Legend
    fontsize_leg=default_fontsize['legend'],
    loc_leg='upper left', show_leg=None,
    model_bar_mode=False, factor_max=1.1,
    ymin_to_0=True,
    **kwargs):
    
    """ Plot complete histogram with fitted curve, pull histogram 
    and results of the fits, from counts and bin edges
    (for the fitted sample but also the models). This function
    just plots what is given to it in the desired way.

    Parameters
    ----------
    data            : tuple(array-like, array-like)
        Couple ``(counts, err)``
    branch        : str
        name of the branch to plot, which was fitted to
    latex_branch  : str
        latex name of the branch, to be used in the xlabel of the plot
    unit            : str
        Unit of the physical quantity
    centres, edges  : array-like
        bin centres and edges
    color         : str
        color of the histogram of the fitted sample
    bar_mode     : bool

        * if True, plot the histogram of the fitted sample with bars
        * else, plot with points and error bars
    models        : 3-tuple or 1-tuple
        ``(x_model, y_models, y_model_pull)``, where
        ``x_model`` are the bin centres, ``y_models`` the
        list of counts describing the models, and ``y_model_pull``
        is is the counts of the total model with the same binning as
        for the sample, to compute the pull histogram.
        ``x_model`` and ``y_model_pull`` might not be specified (1-tuple).
        In this case, the binning is specified by ``counts``
        or ``edges`` as for the sample.
    models_names : str or list(str) or list(list(str))
        passed to :py:func:`plot_fitted_curves_zfit`
    models_types  : str
        passed to :py:func:`plot_fitted_curves_zfit`
    linewidth     : str
        width of the fitted curve line
    colors        : str
        colors of the fitted curves
    title         : str
        title of the plot
    bar_mode_pull: bool
        if ``True``, the pull diagram is plotted with bars instead of points + error bars.
        if None, don't plot the pull diagram.
    
    params        : dict
        Fitted parameters, given by :py:func:`HEA.fit.fit.launch_fit`
    latex_params  :
        Dictionnary with the name of the params.
        Also indicated the branchs to show in the table among all the branchs in params
    fig_name      : str
        name of the saved file
    folder_name   : str
        name of the folder where to save the plot
    data_name     : str
        name of the data used to constitute the name of the saved file, if ``fig_name`` is not specified.
    save_fig      : str
        name of the figure to save
    pos_text_LHC    : dict, list or str
        passed to :py:func:`HEA.plot.tools.set_text_LHCb` as the ``pos`` argument.
    kwargs_res    : dict
        arguments passed to :py:func:`plot_result_fit`
    show_leg : bool
        is the legend shown? By default, yes.
    ndof : int
        Number of degrees of freedom in the model
    color_pull: str
        color of the pull histogram
    **kwargs:
        passed to :py:func:`plot_fitted_curves_zfit`

    Returns
    -------
    fig   : matplotlib.figure.Figure
        Figure of the plot (only if ``axis_mode`` is ``False``)
    ax[0] : matplotlib.figure.Axes
        Axis of the histogram + fitted curves + table
    ax[1] : matplotlib.figure.Axes
        Axis of the pull diagram (only if ``plot_pull`` is ``True``)
    """
    if show_leg is None:
        if assertion.is_list_tuple(models_names):
            show_leg = False
            for model_name in models_names:
                if model_name is not None:
                    show_leg = True
        else:
            show_leg = models_names is not None
        
    ## Retrieve the histogram ===========================================
    
    centres, edges = get_centres_edges(centres=centres, edges=edges)
    
    counts = data[0]
    err = data[1]
    

    # Retrieve the models ===============================================
    if len(models)==1:
        y_models = models[0]
        y_models = el_to_list(y_models)
        x_model = centres
    

    elif len(models)==3:
        x_model, y_models, y_model_pull = models
        y_models = el_to_list(y_models)

    n_models = len(y_models)

    if y_models[0] is None: # The first model is the sum of the others
        y_models[0] = np.array([sum(y) for y in zip(*tuple(y_models[1:]))])
        if len(colors) < n_models:
            colors = ['b'] + colors
        models_names = el_to_list(models_names, n_models)
        models = el_to_list(models, n_models)
        if len(models_names) < n_models:
            models_names = [None] + models_names

    if len(models)==1: # I
        y_model_pull = y_models[0]
        
    
    # Create the figure ====================================================
    plot_pull = (bar_mode_pull is not None) and (models is not None)
    fig, ax = create_fig_plot_hist_fit(plot_pull)

    if latex_branch is None:
        latex_branch = string._latex_format(branch)

    ax[0].set_title(title, fontsize=25)
    
    ## Plot the fitted distribution =========================================
    bin_width = h.get_bin_width_from_edges(edges)
    
    
    # plot 1D histogram of data
    # Histogram
    
    h.plot_hist_alone_from_hist(
        ax[0], counts=counts, err=err, 
        color=color,
        edges=edges,
        bar_mode=bar_mode, alpha=0.1,
        show_ncounts=False,
        label=data_label,
        orientation='vertical',
        show_xerr=(bin_width is None), # if bin_width is None, it is not constant.
                                
    )
    # Label
    
    set_label_hist(ax[0], latex_branch, unit, bin_width, fontsize=25,
                  density=density)

    # Ticks
    pt.set_label_ticks(ax[0])
    pt.set_text_LHCb(ax[0], pos=pos_text_LHC)

    
    ## Plot the models ===============================================
    # Plot fitted curves
    
    models_names = el_to_list(models_names, n_models)
    colors = el_to_list(colors)
    
    if models_types is None:
        models_types = 'n'
        models_types = "".join(el_to_list(models_types, n_models))
    
    PDF_names = el_to_list(PDF_names, n_models)
    
    if stack:
        model_bar_mode = False
    start = 1 if y_models[0] is None else 0   
    
    ## Allow that a kwarg = list
    
    list_kwargs = []
    for i in range(start, n_models):
        ukwargs = {}
        for key, value in kwargs.items():
            if assertion.is_list_tuple(value):
                ukwargs[key] = value[i]
            else:
                ukwargs[key] = value
            
        list_kwargs.append(ukwargs)
    
    
    for i in range(start, n_models):
        if PDF_names is None:
            PDF_name = None
        else:
            PDF_name = PDF_names[i]

        if stack: 
#             model_mode = 'fillbetween'
            model_mode = 'bar'
        else:
            model_mode = 'bar' if model_bar_mode else None
        if stack and i!=0: 
            model_counts = np.array([sum(y) for y in zip(*tuple(y_models[i:]))])
            plot_fitted_curve_from_hist(
                ax[0], x_model, 
                model_counts,
                PDF_name=PDF_name,
                model_name=models_names[i], 
                model_type=models_types[i],
                color=colors[i], 
                mode=model_mode, edges=edges,
                **list_kwargs[i]
            )
            

        elif y_models[i] is not None: # plot the line of the full model (i=0)
            plot_fitted_curve_from_hist(
                ax[0], x_model, y_models[i],
                PDF_name=PDF_name,
                model_name=models_names[i],
                model_type=models_types[i],
                color=colors[i], 
                mode=model_mode, edges=edges,
                **list_kwargs[i]
            )
            
    pt.change_range_axis(ax[0], factor_max=factor_max, min_to_0=ymin_to_0)
    
    if show_leg:
        ax[0].legend(fontsize=fontsize_leg, loc=loc_leg)
    
    ## Plot the pull histogram ==========================================
    
    if plot_pull:
        if color_pull is None:
            color_pull = colors if not assertion.is_list_tuple(colors) else colors[0]
        
        pull =  plot_pull_diagram_from_hist(
                ax=ax[1], 
                fit_counts=y_model_pull, counts=counts,
                edges=edges, err=err, 
                color=color_pull,
                bar_mode_pull=bar_mode_pull)
        
        print_fit_info(edges, y_model_pull, counts, pull, ndof, err=err)

        if lim_pull is not None:
            if isinstance(lim_pull, float) or isinstance(lim_pull, int):
                lim_pull = (-lim_pull, lim_pull)
            ax[1].set_ylim(lim_pull)
        
        
    ## Plot the fitted parameters ==========================================
    if params is not None:
        plot_result_fit(ax[0], params, latex_params=latex_params,
                        **kwargs_res)

    # print characteristics of the fit and histogram
    
    
    ## Save the results ==========================================
    plt.tight_layout()
    if save_fig:
        pt.save_fig(fig, fig_name, folder_name, f'{branch}_{data_name}_fit')

    if plot_pull:
        return fig, ax[0], ax[1]
    else:
        return fig, ax[0]

def plot_hist_fit2d_counts(branches, counts, fit_counts, xedges, yedges, 
                    err=None, fig_name=None, pull_lim=(-5, 5), **kwargs):
    """ Produce 3 separate histograms:
    
    * Counts in the sample
    * Fitted counts
    * Pull
    
    Parameters
    ----------
    counts: 2d array
        counts in the sample
    err : 2d array
        error on ``counts``
    fit_counts: 2d array
        fitted counts
    xedges, yedges: array-like
        Bin edges
    **kwargs:
        passed to `plot_hist2d_counts`
    """
    
    if err is None:
        err = np.sqrt(counts)
    
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore divide-by-0 warning
        pull = np.divide(counts - fit_counts, err)
    
    vmin = min(counts.min(), fit_counts.min())
    vmax = max(counts.max(), fit_counts.max())
    
    plot_hist2d_counts(branches, counts, xedges, yedges,
                       fig_name=string.add_text(fig_name, 'obs'),
                       vmin=vmin, vmax=vmax,
                       **kwargs)
    plot_hist2d_counts(branches, fit_counts, xedges, yedges, 
                       fig_name=string.add_text(fig_name, 'fit'),
                       vmin=vmin, vmax=vmax,
                       **kwargs)
    plot_hist2d_counts(branches, pull, xedges, yedges, 
                       fig_name=string.add_text(fig_name, 'pull'),
                       vmin=pull_lim[0], vmax=pull_lim[1],
                       **kwargs)
    
    
    


##########################################################################
##################################### Automatic label plots ##############
##########################################################################

def _core_plot_hist_fit_auto(branch, cut_BDT, kwargs):
    """ Core of the function :py:func:`plot_hist_fit_auto`
    
    Parameters
    ----------
    branch: str
        name of the branch
    kwargs: dict
        will be passed to :py:func:`plot_hist_fit`
        
    Returns
    -------
    latex_branch: str
        name of the branch in latex
    unit : str
        unit of the branch
    kwargs: dict
        will be passed to :py:func:`plot_hist_fit`
    """
    latex_branch, unit = pt.get_latex_branches_units(branch)

    # Title and name of the file with BDT
    add_in_dic('fig_name', kwargs)
    add_in_dic('title', kwargs)
    add_in_dic('data_name', kwargs)

    kwargs['fig_name'] = pt._get_fig_name_given_BDT_cut(fig_name=kwargs['fig_name'], cut_BDT=cut_BDT,
                                                        branch=branch,
                                                        data_name=string.add_text(kwargs['data_name'],
                                                                                  'fit', '_', None))

    kwargs['title'] = pt._get_title_given_BDT_cut(
        title=kwargs['title'], cut_BDT=cut_BDT)

    # Name of the folder = name of the data
    add_in_dic('folder_name', kwargs)

    if kwargs['folder_name'] is None and kwargs['data_name'] is not None:
        kwargs['folder_name'] = kwargs['data_name']
    
    return latex_branch, unit, kwargs
        
        
def plot_hist_fit_auto(data, branch, cut_BDT=None, **kwargs):
    """ Retrieve the latex name of the branch and unit. Set the folder name to the name of the datasets.
    Then, plot with :py:func:`plot_hist_fit`.

    Parameters
    ----------

    df            : pandas.Dataframe
        dataframe that contains the branch to plot
    branch : str
        branch (for instance: ``'B0_M'``), in dataframe
    cut_BDT         : float or str
        ``BDT > cut_BDT`` cut. Used in the name of saved figure.
    **kwargs : dict
        arguments passed in :py:func:`plot_hist_fit` (except ``branch``, ``latex_branch``, ``unit``)

    Returns
    -------
    fig   : matplotlib.figure.Figure
        Figure of the plot (only if ``axis_mode`` is ``False``)
    ax[0] : matplotlib.figure.Axes
        Axis of the histogram + fitted curves + table
    ax[1] : matplotlib.figure.Axes
        Axis of the pull diagram (only if ``plot_pull`` is ``True``)
    """
    centres = kwargs.get("centres")
    edges = kwargs.get("edges")
    
    if centres is not None or edges is not None:
        plot_function = plot_hist_fit_counts
    else:
        plot_function = pz.plot_hist_fit
    
    latex_branch, unit, kwargs = _core_plot_hist_fit_auto(branch, cut_BDT, kwargs)
    
    return plot_function(
        data, branch, latex_branch=latex_branch, unit=unit, **kwargs)

def plot_hist_fit2d_auto(branches, *args, with_counts=False, **kwargs):
    """ Retrieve the latex name and set the folder to the name of the datasets,
    and the file name to ``{branch1}_vs_{branch2}_fit``
    
    Parameters
    ----------
    branches, *args, **kwargs:
        passed to :py:func:`plot_hist_fit2d_counts`
        or :py:func:`plot_hist_fit2d`
    with_counts: bool
        whether to use :py:func:`plot_hist_fit2d_counts` (if ``True``) or
        or :py:func:`HEA.plot.zfit.plot_hist_fit2d`
    """
    
    if with_counts:
        plot_function = plot_hist_fit2d_counts
    else:
        plot_function = pz.plot_hist_fit2d
    
    latex_branches, units, kwargs = _core_plot_hist2d(branches, kwargs, suff='_fit')
    
    return plot_function(
        branches, *args, latex_branches=latex_branches, units=units, **kwargs
    )
    

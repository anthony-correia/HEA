"""
Plot a histogram, with the fitted PDF, the pull diagram and the fitted parameters.
"""

import HEA.plot.tools as pt
from HEA.plot.histogram import plot_hist_alone, set_label_hist, get_bin_width
from HEA.tools.da import add_in_dic, el_to_list, get_element_list
from HEA.tools import string, assertion
from HEA.tools import dist
from HEA.fit.params import get_ufloat_latex_from_param_latex_params
from HEA.fit import PDF
from HEA.config import default_fontsize


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
name_PDF = {
    'DoubleCB': 'Double CB',
    'SumPDF': 'Sum',
    'Exponential': 'Exponential',
    'Gauss': 'Gaussian',
    'CrystalBall': 'Crystal Ball',
    'HORNS': 'HORNS'
}

##########################################################################
########################################## Scaling PDF ###################
##########################################################################


def _get_plot_scaling(counts, low, high, n_bins):
    """Return plot_scaling, the factor to scale the curve fit to unormalised histogram
    Parameters
    ----------
    counts : np.array or list
        number of counts in the histogram of the fitted data
    low    : float
        low limit of the distribution
    high   : float
        high limit of the distribution
    n_bins : int
        number of bins
    """
    return counts.sum() * (high - low) / n_bins


def frac_model(x, model, frac=1.):
    """ Return the list of the values of the pdf of the model
    evaluated at ``x``.
    Multiply by ``frac`` each of these values.

    Parameters
    ----------
    x       : numpy.array(float)
        Array of numbers where to evaluate the PDF
    model: zfit.pdf.BasePDF
        Model (PDF)
    frac    : float
        Parameter which is multiplied to the result

    Returns
    -------

    np.array(float)
        list of the values of the pdf of the model evaluated in x,
        multiplied by ``frac``

    """

    return (model.pdf(x) * frac).numpy()

##########################################################################
#################################### Sub-plotting functions ##############
##########################################################################


# PULL DIAGRAM ===========================================================


def print_fit_info(centres, fit_counts, counts, pull, ndof):
    """ Sohow
    
    * Number of bins and bin width
    * :math: Reduced `\\chi^2`
    * Mean and std of the normalised residuals
    
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
    print(f"Number of bins: {len(centres)}")
    print(f"Width of the bins: {centres[1]-centres[0]}")
    print("")
    chi2 = dist.get_reduced_chi2(fit_counts, counts, ndof)
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
    low = edges[0]
    high = edges[-1]
    
    if err is None:
        err = np.sqrt(counts)

    
    with np.errstate(divide='ignore', invalid='ignore'):  # ignore divide-by-0 warning
        pull = np.divide(counts - fit_counts, err)

    # Plotting
    if bar_mode_pull:
        ax.bar(centres, pull, centres[1] -
               centres[0], color=color, edgecolor=None)
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

def _get_frac_or_yield_model(models):
    """ return the ``frac`` of a composite PDF specified with the ``frac`` argument. If the model is a sum of extended PDFs, just return the total number of events in the model

    Parameters
    ----------
    models       : list(zfit.pdf.BasePDF) or list(list(zfit.pdf.BasePDF))
        list of zFit models, whose first element is the sum of the others, weighted by ``frac`` and ``1 - frac`` or extended.

    Returns
    -------
    total_yield : float
        Yield of the model (returned if the composing PDFs are extended)
    frac        : float
        Relative yield of the sub-models (returned if the composing PDFs are not extended)
    mode_frac   : bool
        if ``True``, ``frac`` is returned
        else, ``n_tot`` is returned

    NB: this functions assumes that there is only 1 ``frac`` (I don't need more that 1 ``frac`` yet)
    """
    
    from zfit.core.parameter import ComposedParameter
    from zfit.core.parameter import Parameter as SimpleParameter
    from zfit.models.functor import SumPDF

    # Get the composite PDF
    model = models[0]
    assert isinstance(model, SumPDF)

    mode_frac = False
    parameters = list(model.params.values())

    # The parameters of the composite PDF should be all composedParameters as they are from the composing PDFs
    # Except if there is the ``frac`` parameter, which is indeed not a
    # composed parameted for the composite PDF.
    i = 0
    while not mode_frac and i < len(parameters):
        # if one of the parameter is not a ComposedParameter, this it is a frac
        # parameter
        mode_frac = not isinstance(parameters[i], ComposedParameter)
        if mode_frac:
            # If it is not a ComposedParameter, it should be a SimpleParameter
            assert isinstance(parameters[i], SimpleParameter)
        i += 1

    if mode_frac:
        # parameters[i-1] is a SimpleParameter, i.e., frac
        frac = float(parameters[i - 1])
        return frac, mode_frac
    else:
        n_tot = 0
        # We sum up the yields of the composing PDFs
        for sub_model in model.models:
            assert sub_model.is_extended
            n_tot += float(sub_model.get_yield().value())
        return n_tot, mode_frac


def get_PDF_name(model):
    """ return the name of the ``model``

    Parameters
    ----------
    model: zfit.pdf.BasePDF
        Model (PDF)

    Returns
    -------
    label_mode: str
        name of the model (specified by the dictionnary ``name_PDF``), e.g., ``'Gaussian'``, ``'Crystall Ball'``, ...
    """
    # get the name of the model, removing  '_extended' when the PDF is extended
    marker = model.name.find('_')
    if marker == -1:
        marker = None

    model_name = model.name[:marker]
    assert model_name in name_PDF, f"{model_name} is not defined in {list(name_PDF.keys())}"
    label_model = name_PDF[model_name]

    return label_model


def plot_fitted_curve_from_hist(ax, x, fit_counts,
                                PDF_name=None,
                                model_name=None, model_type=None,
                                color='b', 
                                linestyle='-',linewidth=2.5, 
                                alpha=1, fillbetween=False,
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
    **kwargs:
        passed to :py:func`ax.plot()`
    """
    
    if model_name is None:
        label = None
    else:
        label = string.add_text(PDF_name, model_names_types[model_type], ' - ')
        label = string.add_text(label, model_name)  
    
    if fillbetween:
        return ax.fill_between(x, fit_counts, label=label,
                               color=color,
                               alpha=alpha, **kwargs)  
    else:
        return ax.plot(x, fit_counts, linewidth=linewidth, color=color,
                       label=label,
                       ls=linestyle, alpha=alpha, **kwargs)  


def plot_single_model(ax, x, model, plot_scaling,
                       model_name=None,
                       frac=1., **kwargs):
    """ Plot the models recursively
    with a label for the curve ``"{name of the PDF (e.g., Gaussian, ...)} - {type of the model, e.g., signal ...} {Name of the model, e.g., "B0->Dst Ds"}"`` (if ``model_name`` is specified)
    ax           : matplotlib.axes.Axes
        axis where to plot
    x             : numpy.numpy(float)
        points of the x-axis where to evaluate the pdf of the model to plot
    model        : zfit.pdf.BasePDF
        just one zfit model
    plot_scaling : float
        scaling to get the scale of the curve right
        
    model_name : str
        name of the models - used in the legend.
        If ``None``, the legend is not shown
    frac        : float
        frac is multiplied to the PDF to get the correct scale due to composite PDFs
    """
    assert not assertion.is_list_tuple(model)
    
    if model_name is None:
        PDF_name = None
    else:
        PDF_name = get_PDF_name(model)
        
    y = frac_model(x, model, frac=frac) * plot_scaling    
    plot_fitted_curve_from_hist(ax, x, y, 
                                model_name=model_name,
                                PDF_name=PDF_name,
                                **kwargs)


def _plot_models(ax, x, models, plot_scaling, models_types=None, models_names=None,
                 frac=1., PDF_level=0, colors=['b', 'g', 'gold', 'magenta', 'orange'],
                 linestyles=['-', '--', ':', '-.'], linewidth=2.5):
    """ Plot the models recursively
    with a label for each curve ``"{name of the PDF (e.g., Gaussian, ...)} - {type of the model, e.g., signal ...} {Name of the model, e.g., "B0->Dst Ds"}"`` (if the corresponding model_name is specified)
    ax           : matplotlib.axes.Axes
        axis where to plot
    x             : numpy.numpy(float)
        points of the x-axis where to evaluate the pdf of the model to plot
    models       : list(zfit.pdf.BasePDF) or list(list(zfit.pdf.BasePDF))

        * just one PDF (e.g., ``[model_PDF]``)
        * a list of PDFs, whose first PDF is the composite PDF
        and the other ones are their components (e.g., ``[model_PDG, signal_PDF, background_PDF]``)
        *  list of list of PDFs, if composite of composite of PDFs (e.g., ``[model_PDG, [signal_PDF, signal_compo1_PDF, signal_compo2_PDF], background_PDF]``)
        * etc. (recursive)
    plot_scaling : float
        scaling to get the scale of the curve right
    models_types  : str
        type of each mode (one character for each model or for a list of models):

        * ``'m'`` : model (sum) ; should always be the FIRST ONE !!
        * ``'s'`` : signal
        * ``'b'`` : background
        used in the legend to indicate if it is a signal or a background component
    models_names : str or list(str) or list(list(str))
        name of the models - used in the legend

        * list of the same size as ``models_names`` with the name of each PDF
        * If there is only one string for a list of models, it corresponds to the name of the first composite PDFs. The other PDFs are plotted but they aren't legended
    frac        : float
        frac is multiplied to the PDF to get the correct scale due to composite PDFs
    colors      : list(str)
        list of colors for each curve, same structure as models_names
    linestyles  : list(str)
        list of linestyles at each level of composite PDFs, as specified by the ``PDF_level`` argument
    PDF_level   : int

        * 0 is first sumPDF
        * 1 if component of this sumPDF
        * 2 if component of a sumPDF component of sumPDF
        * etc.
    linewidth  : float
        width of the plotted lines
    """

    if assertion.is_list_tuple(models):  # if there are several PDFs to plot
        # if there are more than 1 model:
        if len(models) > 1:
            frac_or_yield, mode_frac = _get_frac_or_yield_model(models)
        else:  # if there is only one model, there is no frac and no yield to compute. It has already been computed
            mode_frac = None

        # So far with this function, we can use to specify ``frac`` for
        # composite PDF only if the model is made of 2 composing PDFs.
        if mode_frac:
            assert len(models) == 3

        for k, model in enumerate(models):
            if k == 1:  # models = [sumPDF, compositePDF1, compositePDF2, ...]
                PDF_level += 1
            # Compute frac
            applied_frac = frac  # frac already specified

            if mode_frac is not None:
                if mode_frac:  # in this case, frac_or_yield = frac and there is 2 composite PDFs
                    if k == 1:
                        applied_frac = frac * frac_or_yield
                    elif k == 2:
                        applied_frac = frac * (1 - frac_or_yield)
                else:  # in this case, frac_or_yield = yield, the yield of the model
                    # frac_or_yield is yield
                    total_yield = frac_or_yield
                    if k >= 1:
                        # we get the composing model
                        main_model = get_element_list(model, 0)
                        # and compute its relative yield
                        applied_frac = frac * \
                            float(main_model.get_yield().value()) / total_yield

            # labels
            if len(models_types) > 1:
                model_type = models_types[k]
            else:
                model_type = models_types

            # color
            color = get_element_list(colors, k)

            if not isinstance(models_names, list):
                # if the name of the subsubmodel is not specified, put it to
                # None
                if k == 0:
                    model_name = models_names
                else:
                    model_name = None
            else:
                model_name = models_names[k]

            _plot_models(ax, x, model, plot_scaling, model_type, model_name, applied_frac, PDF_level, color,
                         linestyles, linewidth)

    else:  # if there is only one PDF to plot
        if PDF_level >= 2:
            alpha = 0.5
        else:
            alpha = 1
        plot_single_model(ax, x, models, plot_scaling, model_type=models_types, model_name=models_names,
                           frac=frac, color=colors,
                           linestyle=linestyles[PDF_level], linewidth=linewidth, alpha=alpha)


def plot_fitted_curves(ax, models, plot_scaling, low, high,
                       models_names=None, models_types=None,
                       fontsize_leg=default_fontsize['legend'],
                       loc_leg='upper left', show_leg=None,
                       **kwargs):
    """Plot fitted curve given by ``models``, with labels given by ``models_names``

    Parameters
    ----------
    ax              : axis where to plot
    models       : zfit.pdf.BasePDF or list(zfit.pdf.BasePDF) or list(list(zfit.pdf.BasePDF)) or ...

        * just one PDF (e.g., ``[model_PDF]`` or ``model_PDF``)
        * a list of PDFs, whose first PDF is the composite PDF and the other ones are their components (e.g., ``[model_PDG, signal_PDF, background_PDF]``)
        * list of list of PDFs, if composite of composite of PDFs (e.g., ``[model_PDG, [signal_PDF, signal_compo1_PDF, signal_compo2_PDF], background_PDF]``)
        * etc. (recursive)
    low             : float
        low limit of the plot (x-axis)
    high            : float
        high limit of the plot (x-axis)
    models_names : str or list(str) or list(list(str))
        name of the models - used in the legend.
        List of the same size as ``models_names`` with the name of each PDF.
        If there is only one string for a list of models, it corresponds to the name of the first composite PDFs.
        The other PDFs are plotted but they aren't legended.

    models_types  : str
        type of each mode (one character for each model or for a list of models):

        * ``'m'`` : model (sum) ; should always be the FIRST ONE !!
        * ``'s'`` : signal
        * ``'b'`` : background

        used in the legend to indicate if it is a signal or a background component.
        If ``None``, it is put to ``['m', 's', 'b', 'b', ...]``.

    fontsize_leg : float
        fontsize of the legend
    loc_leg         : str
        location of the legend
    show_leg     : bool
        if ``True``, show the legend,
        if None, show the legend only if there are more than 1 model
    **kwgs        : dict
        passed to :py:func:`plot_models`

    """
    models = el_to_list(models, 1)
    
    models_names = el_to_list(models_names, len(models))

    x = np.linspace(low, high, 1000)

    # Plot the models
    if models_types is None:
        models_types = 'm'
        if len(models_names) >= 2:
            models_types += 's'
            models_types += 'b' * (len(models_names) - 2)

    _plot_models(ax, x, models, plot_scaling,
                 models_types=models_types, models_names=models_names, **kwargs)

    if show_leg is None:
        show_leg = models_names is not None
    if show_leg:
        ax.legend(fontsize=fontsize_leg, loc=loc_leg)

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
    """ Create fig, axs for py:func:`plot_hist_fit`
    
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
    
        

def plot_hist_fit(df, branch, latex_branch=None, unit=None, weights=None,
                  obs=None, n_bins=50, low_hist=None, high_hist=None,
                  color='black', bar_mode=False,
                  models=None, models_names=None, models_types=None,
                  linewidth=2.5, colors=None,
                  title=None,
                  bar_mode_pull=True,
                  params=None, latex_params=None, 
                  kwargs_res={},
                  fig_name=None, folder_name=None, data_name=None,
                  save_fig=True, pos_text_LHC=None,
                  **kwargs):
    """ Plot complete histogram with fitted curve, pull histogram and results of the fits. Save it in the plot folder.

    Parameters
    ----------
    df            : pandas.Dataframe
        dataframe that contains the branch to plot
    branch        : str
        name of the branch to plot and that was fitted
    latex_branch  : str
        latex name of the branch, to be used in the xlabel of the plot
    unit            : str
        Unit of the physical quantity
    weights         : numpy.array
        weights passed to ``plt.hist``
    obs           : zfit.Space
        Space used for the fit
    n_bins        : int
        number of desired bins of the histogram
    low_hist      : float
        lower range value for the histogram (if not specified, use the value contained in ``obs``)
    high_hist     : float
        lower range value for the histogram (if not specified, use the value contained in ``obs``)
    color         : str
        color of the histogram
    bar_mode     : bool

        * if True, plot with bars
        * else, plot with points and error bars
    models        : zfit.pdf.BasePDF or list(zfit.pdf.BasePDF) or list(list(zfit.pdf.BasePDF)) or ...
        passed to :py:func:`plot_fitted_curves`
    models_names : str or list(str) or list(list(str))
        passed to :py:func:`plot_fitted_curves`
    models_types  : str
        passed to :py:func:`plot_fitted_curves`
    linewidth     : str
        width of the fitted curve line
    colors        : str
        colors of the fitted curves
    title         : str
        title of the plot
    bar_mode_pull: bool
        if ``True``, the pull diagram is plotted with bars instead of points + error bars.
        if None, don't plot the pull diagram.
    params        : dict[zfit.zfitParameter, float]
        Result ``result.params`` of the minimisation of the loss function (given by :py:func:`HEA.fit.fit.launch_fit`)
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
    **kwargs:
        passed to py:func:`plot_fitted_curves`

    Returns
    -------
    fig   : matplotlib.figure.Figure
        Figure of the plot (only if ``axis_mode`` is ``False``)
    ax[0] : matplotlib.figure.Axes
        Axis of the histogram + fitted curves + table
    ax[1] : matplotlib.figure.Axes
        Axis of the pull diagram (only if ``plot_pull`` is ``True``)
    """
    
    # Create figure
    plot_pull = (bar_mode_pull is not None)
    fig, ax = create_fig_plot_hist_fit(plot_pull)

    # Retrieve low,high (of x-axis)
    low = float(obs.limits[0])
    high = float(obs.limits[1])

    if low_hist is None:
        low_hist = low
    if high_hist is None:
        high_hist = high

    if latex_branch is None:
        latex_branch = string._latex_format(branch)

    ax[0].set_title(title, fontsize=25)

    # plot 1D histogram of data
    # Histogram
    counts, edges, centres, err = plot_hist_alone(
        ax=ax[0], 
        data=df[branch], weights=weights,
        n_bins=n_bins,
        low=low_hist, high=high_hist, 
        color=color, bar_mode=bar_mode, alpha=0.1)

    # Label
    bin_width = get_bin_width(low_hist, high_hist, n_bins)
    set_label_hist(ax[0], latex_branch, unit, bin_width, fontsize=25)

    # Ticks
    pt.set_label_ticks(ax[0])
    pt.set_text_LHCb(ax[0], pos=pos_text_LHC)

    # Plot fitted curve
    if isinstance(models, list):
        model = models[0]  # the first model is the "global" one
    else:
        model = models

    plot_scaling = _get_plot_scaling(counts, low_hist, high_hist, n_bins)
    plot_fitted_curves(ax[0], models, plot_scaling, low, high, 
                       models_names=models_names, models_types=models_types,
                       linewidth=2.5, colors=colors, **kwargs)

    pt.change_range_axis(ax[0], factor_max=1.1)

    color_pull = colors if not isinstance(colors, list) else colors[0]
    # Plot pull histogram
    if plot_pull:
        fit_counts = frac_model(centres, model, plot_scaling)
        pull =  plot_pull_diagram_from_hist(
                ax=ax[1], 
                fit_counts=fit_counts, counts=counts,
                edges=edges, err=err, 
                color=color_pull,
                bar_mode_pull=bar_mode_pull)
        
        ndof = PDF.get_n_dof_model(model)
        print_fit_info(centres, fit_counts, counts, pull, ndof)

    # Plot the fitted parameters of the fit
    if params is not None:
        plot_result_fit(ax[0], params, latex_params=latex_params,
                        **kwargs_res)

    # print characteristics of the fit and histogram
    
    
    # Save result
    plt.tight_layout()
    if save_fig:
        pt.save_fig(fig, fig_name, folder_name, f'{branch}_{data_name}_fit')

    if plot_pull:
        return fig, ax[0], ax[1]
    else:
        return fig, ax[0]


def plot_hist_fit_var(data, branch, latex_branch=None, unit=None, **kwargs):
    """ plot data with his fit

    Parameters
    ----------
    data      : pandas.Series or list(pandas.Series)
        dataset to plot
    branch    : str
        name of the branch, for the name of the file
    latex_branch  : str
        name of the branch, for the label of the x-axis
    unit      : str
        unit of the branch
    **kwargs  : dict
        parameters passed to :py:func:`HEA.plot.tools.plot_hist_fit`

    Returns
    -------
    fig   : matplotlib.figure.Figure
        Figure of the plot (only if ``axis_mode`` is ``False``)
    ax[0] : matplotlib.figure.Axes
        Axis of the histogram + fitted curves + table
    ax[1] : matplotlib.figure.Axes
        Axis of the pull diagram (only if `plot_pull` is ``True``)
    """
    df = DataFrame()
    df[branch] = data

    return plot_hist_fit(df, branch, latex_branch, unit, **kwargs)


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
        
        
def plot_hist_fit_auto(df, branch, cut_BDT=None, **kwargs):
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
    
    latex_branch, unit, kwargs = _core_plot_hist_fit_auto(branch, cut_BDT, kwargs)
    
    return plot_hist_fit(
        df, branch, latex_branch=latex_branch, unit=unit, **kwargs)

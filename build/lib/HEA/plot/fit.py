"""
Anthony Correia
02/01/21
- Compute the number of d.o.f. of a model
- Compute the reduced chi2 of a model
- plot the pull diagram ot a fit
- plot the histogram, with the fitted PDF and the fitted parameters and the pull diagram
"""

import HEA.plot.tools as pt
from .histogram import plot_hist_alone, set_label_hist, get_bin_width
from HEA.tools.da import add_in_dic, el_to_list, get_element_list
from HEA.tools import string, assertion
from HEA.config import default_fontsize
from HEA.fit import PDF


from zfit.core.parameter import ComposedParameter
from zfit.core.parameter import Parameter as SimpleParameter
from zfit.models.functor import SumPDF

#from zfit.core.parameter import Parameter

import numpy as np

from pandas import DataFrame

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator
import matplotlib.ticker as ticker

#Gives us nice LaTeX fonts in the plots
from matplotlib import rc, rcParams
rc('font',**{'family':'serif','serif':['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False


model_names_types = {
    'm' : 'model',
    's' : 'signal',
    'b' : 'background'
}

# Alternative names for the models
name_PDF = {
    'DoubleCB': 'Double CB',
    'SumPDF'  : 'Sum',
    'Exponential': 'Exponential',
    'Gauss' : 'Gaussian',
    'CrystalBall': 'Crystal Ball'
}

#################################################################################################
########################################## Scaling PDF ##########################################
#################################################################################################


def get_plot_scaling(counts, low, high, n_bins):
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
    return counts.sum() *(high-low) / n_bins 
    
def frac_model(x, model, frac=1.):
    """ Return the list of the values of the pdf of the model evaluated at x.
    Multiply by frac each of these values.
    
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
        list of the values of the pdf of the model evaluated in x
                    multiplied by frac
    
    """
    
    return (model.pdf(x)*frac).numpy()

#################################################################################################
#################################### Sub-plotting functions #####################################
#################################################################################################
    

## PULL DIAGRAM ===================================================================================
def plot_pull_diagram(ax, model, counts, edges=None, centres=None, err=None, low=None, high=None, y_line=3, bar_mode_pull=True,
                      plot_scaling=None, fontsize=default_fontsize['label'], color='b', color_lines='r', show_chi2=False):
    """
    Plot pull diagram of 'model' compared to the data, given by (counts, centres)
    
    
    Parameters
    ----------
    
    ax            : matplotlib.axes.Axes
        axis where to plot
    model: zfit.pdf.BasePDF
        Model (PDF)
    counts        : np.array(float)
        counts of the bins given by centres in the histogram
    centres       : np.array(float)
        bin centres of the histogram (if `edges` is not provided)
    edges        : np.array(float)
        edges of the bins (if `centres` is not provided)
    err       : np.array(float)
        bin centres of the histogram    
    low    : float
        low limit of the distribution
    high   : float
        high limit of the distribution
    fontsize      : float
        fontsize of the labels
    color         : str
        color of the pull diagram
    color_lines   : str
        color of the lines at y = 0, y = y_line and y= - y_line (default is red)
    show_chi2     : bool
        if True, show the chi2 in the label of the x-axis of the pull diagram
    """
    
    assert (edges is not None) or (centres is not None), "The array of the edges or of the centres of the bins should be provided"
    
    if edges is None and centres is not None:
        edges = centres - (centres[1] - centres[0])/2
        edges = np.append(edges, edges[-1])
    
    elif edges is not None and centres is None:
        centres = (edges[:-1] + edges[1:])/2.
        
    if err is None:
        err = np.sqrt(counts)
    
    ## Computing
    if low is None:
        low = centres[0] - (centres[1]-centres[0])/2
    if high is None:
        high = centres[-1] + (centres[1]-centres[0])/2
    
    if plot_scaling is None:
        n_bins = len(centres)
        plot_scaling = counts.sum() * (high-low) / n_bins
        
    fit = model.pdf(centres).numpy()*plot_scaling
    with np.errstate(divide='ignore', invalid='ignore'): # ignore divide-by-0 warning
        pull = np.divide(counts-fit,err)
    
    ## Plotting
    if bar_mode_pull:
        ax.bar(centres, pull, centres[1]-centres[0], color=color, edgecolor=None)
        ax.step(edges[1:], pull, color=color)
    else:
        ax.errorbar(centres,pull, yerr = np.ones(len(centres)),color=color, ls='', marker='.')
    ax.plot([low,high],[y_line, y_line],color='r',ls='--')
    ax.plot([low,high],[-y_line, -y_line],color='r',ls='--')
    ax.plot([low,high],[0,0],color='r')
    
    ## Symmetric pull diagram
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
    
    ax.set_xlim([low,high]) 
    
    ndof = PDF.get_n_dof_model(model)
    
    ## Fit quality
    print(f"Number of bins: {len(centres)}")
    print(f"Width of the bins: {centres[1]-centres[0]}")
    print("")
    chi2 = PDF.get_reduced_chi2(fit, counts, ndof)
    print("Number of d.o.f. in the model: ", ndof)
    print('Reduced chi2: ', chi2)
    print("")
    print(f"Mean of the normalised residuals: {PDF.get_mean(pull)}")
    print(f"Std of the normalised residuals: {PDF.get_std(pull)}")
    
    if show_chi2:
        ax.set_xlabel(f'(reduced $\\chi^2$={chi2:.2f})', fontsize=fontsize)

## FITTED CURVES ===================================================================================
        
def plot_fitted_curve(ax, model, plot_scaling, frac=None, line_width=2.5,
                      color='b',linestyle='-', low=None, high=None, label=None, x=None, alpha=1):
    """
    Plot a fitted curve given by 'model'
    
    Parameters
    ----------
    ax            : matplotlib.axes.Axes
        axis where to plot the label
    model: zfit.pdf.BasePDF
        Model (PDF)
    frac          : float between 0 and 1
        multiplicative factor corresponding to yield ratio of the sub-models of the model
        frac(sub_model) = yield(sub_model)/yield(model)
    plot_scaling  : float
        scaling to get the scale of the curve right compared to the histogram
    line_width    : float
        width of the curve line
    color         : str
        color of the line
    linestyle     : str
        style of the line
    low           : float
        low limit of the plot (x-axis)
    high          : float
        high limit of the plot (x-axis)
    label         : str
        label of the curve (for the legend)
    x             : numpy.numpy(float)
        points of the x-axis where to evaluate the pdf of the model to plot
        If not given, it is computed to be 1000 points between `low` and `high`
    alpha         : float, between 0 and 1
        opacity of the curve
    """
    
    if x is None:
        assert (low is not None) and (high is not None), "If 'x' is None, low and high must be specified"
        x = np.linspace(low, high, 1000)
    
    y = frac_model(x, model, frac=frac)* plot_scaling
    ax.plot(x, y, linewidth=line_width, color=color, ls=linestyle, label=label, alpha=alpha)




def get_frac_or_yield_model(models):
    """ return the `frac` of a composite PDF specified with the `frac` argument. If the model is a sum of extended PDFs, just return the total number of events in the model
    Parameters
    ----------
    models       : list(zfit.pdf.BasePDF) or list(list(zfit.pdf.BasePDF))
        list of zFit models, whose first element is the sum of the others, weighted by frac or extended.
    
    Returns
    -------
    total_yield : float
        Yield of the model (returned if the composing PDFs are extended)
    frac        : float
        Relative yield of the sub-models (returned if the composing PDFs are not extended)
    mode_frac   : bool
        if True, `frac` was returned
        else, `n_tot` was returned
    
    NB: this functions assumes that there is only 1 frac (I don't need more that 1 fracs yet)
    """
    
    # Get the composite PDF
    model = models[0]
    assert isinstance(model, SumPDF)
    
    mode_frac = False
    parameters = list(model.params.values())
    
    # The parameters of the composite PDF should be all composedParameters as they are from the composing PDFs
    # Except if there is the `frac` parameter, which is indeed not a composed parameted for the composite PDF.
    i = 0
    while not mode_frac and i<len(parameters):
        # if one of the parameter is not a ComposedParameter, this it is a frac parameter
        mode_frac = not isinstance(parameters[i], ComposedParameter)
        if mode_frac: 
            # If it is not a ComposedParameter, it should be a SimpleParameter
            assert isinstance(parameters[i], SimpleParameter)
        i+= 1
    
    if mode_frac:
        # parameters[i-1] is a SimpleParameter, i.e., frac
        frac = float(parameters[i-1])
        return frac, mode_frac
    else:
        n_tot = 0
        # We sum up the yields of the composing PDFs
        for sub_model in model.models:
            assert sub_model.is_extended
            n_tot += float(sub_model.get_yield().value())
        return n_tot, mode_frac

        

def get_model_name(model):
    """ return the name of the model
    
    Parameters
    ----------
    model: zfit.pdf.BasePDF
        Model (PDF)
    
    Returns
    -------
    label_mode: str
        name of the model (specified by the dictionnary name_PDF)
    """
    # get the name of the model, removing  '_extended' when the PDF is extended
    marker = model.name.find('_') 
    if marker == -1:
        marker = None
    
    model_name = model.name[:marker]
    assert model_name in name_PDF, f"{model_name} is not defined in {list(name_PDF.keys())}"
    label_model = name_PDF[model_name]
    
    return label_model


def _plot_single_model(ax, x, model, plot_scaling, 
                       model_type=None, model_name=None, 
                       frac=1.,
                       color='b', linestyle='-', line_width=2.5, alpha=1):
    """ Plot the models recursively
    with a label for the curve `"{name of the PDF (e.g., Gaussian, ...)} - {type of the model, e.g., signal ...} {Name of the model, e.g., "B0->Dst Ds"}"` (if `model_name` is specified)
    ax           : matplotlib.axes.Axes
        axis where to plot
    x             : numpy.numpy(float)
        points of the x-axis where to evaluate the pdf of the model to plot
    model        : zfit.pdf.BasePDF
        just one zfit model
    plot_scaling : float
        scaling to get the scale of the curve right
    model_type  : str
        type of the model
        - 'm' : model (sum) --> should always be the FIRST ONE !!
        - 's' : signal
        - 'b' : background
        used in the legend to indicate if it is a signal or a background component
    model_name : str
        name of the models - used in the legend
        if None, the legend is not shown
    frac        : float
        frac is multiplied to the PDF to get the correct scale due to composite PDFs
    color      : str
        list of colors for each curve, same structure as models_names
    linestyle  : str
        line style of the curve
    PDF_level   : int
        - 0 is first sumPDF
        - 1 if component of this sumPDF
        - 2 if component of a sumPDF component of sumPDF
        - etc.
    line_width  : float
        width of the plotted lines
    """
    assert not assertion.is_list_tuple(model)

    # Label
    if model_name is not None:
        label_model = f'{get_model_name(model)} - {model_names_types[model_type]}'
        label_model = string.add_text(label_model, model_name)
    else:
        label_model=None

    plot_fitted_curve(ax, model, plot_scaling, frac=frac, line_width=line_width, color=color, 
                      linestyle = linestyle, label=label_model, x=x, alpha=alpha)
    
def _plot_models(ax, x, models, plot_scaling, models_types=None, models_names=None, 
                frac=1., PDF_level=0, colors=['b', 'g', 'gold', 'magenta', 'orange'], 
                linestyles=['-', '--', ':', '-.'], line_width=2.5):
    """ Plot the models recursively
    with a label for each curve `"{name of the PDF (e.g., Gaussian, ...)} - {type of the model, e.g., signal ...} {Name of the model, e.g., "B0->Dst Ds"}"` (if the corresponding model_name is specified)
    ax           : matplotlib.axes.Axes
        axis where to plot
    x             : numpy.numpy(float)
        points of the x-axis where to evaluate the pdf of the model to plot
    models       : list(zfit.pdf.BasePDF) or list(list(zfit.pdf.BasePDF))
        - just one PDF (e.g., [model_PDF])
        - a list of PDFs, whose first PDF is the composite PDF and the other ones are their components
            (e.g., [model_PDG, signal_PDF, background_PDF])
        -  list of list of PDFs, if composite of composite of PDFs
        (e.g., [model_PDG, [signal_PDF, signal_compo1_PDF, signal_compo2_PDF], background_PDF])
        - ... (recursive)
    plot_scaling : float
        scaling to get the scale of the curve right
    models_types  : str
        type of each mode (one character for each model or for a list of models):
        - 'm' : model (sum) --> should always be the FIRST ONE !!
        - 's' : signal
        - 'b' : background
        used in the legend to indicate if it is a signal or a background component
    models_names : str or list(str) or list(list(str))
        name of the models - used in the legend
        - list of the same size as models_names with the name of each PDF
        - If there is only one string for a list of models, it corresponds to the name of the first composite PDFs. The other PDFs are plotted but they aren't legended
    frac        : float
        frac is multiplied to the PDF to get the correct scale due to composite PDFs
    colors      : list(str)
        list of colors for each curve, same structure as models_names
    linestyles  : list(str)
        list of linestyles at each level of composite PDFs, as specified by the `PDF_level` argument
    PDF_level   : int
        - 0 is first sumPDF
        - 1 if component of this sumPDF
        - 2 if component of a sumPDF component of sumPDF
        - etc.
    line_width  : float
        width of the plotted lines
    """             
    
    if assertion.is_list_tuple(models): # if there are several PDFs to plot
        # if there are more than 1 model:
        if len(models)>1:
            frac_or_yield, mode_frac = get_frac_or_yield_model(models)
        else: # if there is only one model, there is no frac and no yield to compute. It has already been computed
            mode_frac = None
        
        # So far with this function, we can use to specify `frac` for composite PDF only if the model is made of 2 composing PDFs.
        if mode_frac:
            assert len(models)==3

        for k, model in enumerate(models):
            if k==1: # models = [sumPDF, compositePDF1, compositePDF2, ...]
                PDF_level+=1
            # Compute frac
            applied_frac = frac # frac already specified
            
            if mode_frac is not None:
                if mode_frac: # in this case, frac_or_yield = frac and there is 2 composite PDFs
                    if k == 1:
                        applied_frac = frac * frac_or_yield
                    elif k == 2 : 
                        applied_frac = frac * (1 - frac_or_yield)
                else: # in this case, frac_or_yield = yield, the yield of the model
                    #frac_or_yield is yield
                    total_yield = frac_or_yield
                    if k>=1:
                        # we get the composing model
                        main_model = get_element_list(model, 0) 
                        # and compute its relative yield
                        applied_frac = frac * float(main_model.get_yield().value()) / total_yield 
            
            # labels
            if len(models_types)>1:
                model_type = models_types[k]
            else:
                model_type = models_types
            
            # color
            color = get_element_list(colors, k)
            
            if not isinstance(models_names, list): 
                # if the name of the subsubmodel is not specified, put it to None
                if k==0:
                    model_name = models_names
                else:
                    model_name = None
            else:
                model_name = models_names[k]


            _plot_models(ax, x, model, plot_scaling, model_type, model_name, applied_frac, PDF_level, color, 
                        linestyles, line_width)
    
    else: # if there is only one PDF to plot
        if PDF_level>=2:
            alpha = 0.5
        else:
            alpha = 1
        _plot_single_model(ax, x, models, plot_scaling, model_type=models_types, model_name=models_names, 
                frac=frac, color=colors, 
                linestyle=linestyles[PDF_level], line_width=line_width, alpha=alpha)

        
def plot_fitted_curves(ax, models, plot_scaling, low, high, 
                       models_names=None, models_types=None,
                       fontsize_legend=default_fontsize['legend'],
                       loc_leg='upper left', show_legend=None,
                       **kwgs):
    """Plot fitted curve given by `models`, with labels given by `models_names`
    
    Parameters
    ----------
    ax              : axis where to plot
    models       : zfit.pdf.BasePDF or list(zfit.pdf.BasePDF) or list(list(zfit.pdf.BasePDF)) or ...
        - just one PDF (e.g., [model_PDF] or model_PDF)
        - a list of PDFs, whose first PDF is the composite PDF and the other ones are their components
            (e.g., [model_PDG, signal_PDF, background_PDF])
        -  list of list of PDFs, if composite of composite of PDFs
        (e.g., [model_PDG, [signal_PDF, signal_compo1_PDF, signal_compo2_PDF], background_PDF])
        - ... (recursive)
    low             : float
        low limit of the plot (x-axis)
    high            : float
        high limit of the plot (x-axis)
    models_names : str or list(str) or list(list(str))
        name of the models - used in the legend
        - list of the same size as models_names with the name of each PDF
        - If there is only one string for a list of models, it corresponds to the name of the first composite PDFs. The other PDFs are plotted but they aren't legended
    models_types  : str
        type of each mode (one character for each model or for a list of models):
        - 'm' : model (sum) --> should always be the FIRST ONE !!
        - 's' : signal
        - 'b' : background
        used in the legend to indicate if it is a signal or a background component
        If None, it is put to ['m', 's', 'b', 'b', ...]
    fontsize_legend : float
        fontsize of the legend
    loc_leg         : str
        location of the legend
    show_legend     : bool
        if True, show the legend
        if None, show the legend only if there are more than 1 model
    **kwgs        : dict
        passed to plot_models
    
    """
    models = el_to_list(models, 1)
    if show_legend is None:
        show_legend = models_names is not None
    models_names = el_to_list(models_names, len(models))

    x = np.linspace(low, high, 1000)
    
    # Plot the models
    if models_types is None:
        models_types = 'm'
        if len(models_names) >= 2:
            models_types += 's'
            models_types += 'b'*(len(models_names)-2)
    
    _plot_models(ax, x, models, plot_scaling, models_types=models_types, models_names=models_names, **kwgs)

    if show_legend:
        ax.legend(fontsize=fontsize_legend, loc=loc_leg)

## RESULT FIT ===================================================================================
        
        
        
def plot_result_fit(ax, params, latex_params=None, fontsize=default_fontsize['legend'], colWidths=[0.06, 0.01, 0.05, 0.06], loc='upper right'):
    """
    Plot the results of the fit in a table
    
    Parameters
    ----------
    ax            : matplotlib.axes.Axes
        axis where to plot
    params        : dict[zfit.zfitParameter, float]
        Result 'result.params' of the minimisation of the loss function (given by `fit.fit.launch_fit`)
    latex_params  : 
        Dictionnary with the name of the params 
        Also indicated the branchs to show in the table among all the branchs in params
    fontsize      : float
        Fontsize of the text in the table
    colWidths     : [float, float, float, float]
        Width of the four columns of the table
        - first column: latex name of the parameter
        - second column: nominal value
        - third column: `$\pm$`
        - fourth column: error
    loc           : str
        location of the table
    """
    #result_fit = ""
    result_fit = []
    for p in list(params.keys()): # loop through the parameters
        
        name_param = p.name
        
        # if name_param as ';', remove everything from ';' onwards
        index = name_param.find(';')
        if index!=-1:
            name_param = name_param[:index]
        
        # if latex_params not None, it specifies the branchs we want to show
        if (latex_params is None) or (name_param in latex_params): 
            # Retrieve value and error
            value_param = params[p]['value']
            error_param = params[p]['minuit_hesse']['error']
            
            # Retrieve alt name
            if latex_params is not None:
                surname_param = latex_params[name_param]
            else:
                surname_param = string._latex_format(name_param)

            # Table --> name_param   :   value_param +/- error_param
            size_int_part = len(str(int(value_param)))
            if size_int_part>=4:
                value_param_text = f"{value_param:.0f}"
            else:
                value_param_text = f"{value_param:.4f}"
            result_fit.append([surname_param,":",value_param_text,f'$\pm$ {error_param:.2g}']) 

    # Plot the table with the fitted parameters in the upper right part of the plot  
    table = ax.table(result_fit,loc=loc, edges='open', cellLoc='left',
              colWidths=colWidths)                 
    table.auto_set_font_size(False)
    table.set_fontsize(fontsize)
    table.scale(2, 2) 



#################################################################################################
###################################### Plotting function ########################################
#################################################################################################   
    
def plot_hist_fit(df, branch, latex_branch=None, unit=None, weights=None,
                  obs=None, n_bins=50, low_hist=None, high_hist=None,
                  color='black', bar_mode=True,
                  models=None,  models_names=None, models_types=None, 
                  linewidth=2.5, colors=None,
                  title=None, 
                  plot_pull=True, bar_mode_pull=True,
                  show_leg=None, fontsize_leg=default_fontsize['legend'], loc_leg='upper left',
                  show_chi2=False,
                  params=None, latex_params=None, colWidths=[0.04,0.01,0.06,0.06], fontsize_res=default_fontsize['legend'],
                  loc_res='upper right', 
                  fig_name=None, folder_name=None, data_name=None, 
                  save_fig=True, pos_text_LHC=None):
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
        weights passed to plt.hist
    obs           : zfit.Space
        Space used for the fit
    n_bins        : int
        number of desired bins of the histogram
    low_hist      : float
        lower range value for the histogram (if not specified, use the value contained in `obs`)
    high_hist     : float
        lower range value for the histogram (if not specified, use the value contained in `obs`)
    color         : str
        color of the histogram
    bar_mode     : bool
        - if True, plot with bars
        - else, plot with points and error bars
    models        : zfit.pdf.BasePDF or list(zfit.pdf.BasePDF) or list(list(zfit.pdf.BasePDF)) or ...
        - just one PDF (e.g., [model_PDF] or model_PDF)
        - a list of PDFs, whose first PDF is the composite PDF and the other ones are their components
            (e.g., [model_PDG, signal_PDF, background_PDF])
        -  list of list of PDFs, if composite of composite of PDFs
        (e.g., [model_PDG, [signal_PDF, signal_compo1_PDF, signal_compo2_PDF], background_PDF])
        - ... (recursive)
    models_names : str or list(str) or list(list(str))
        name of the models - used in the legend
        - list of the same size as models_names with the name of each PDF
        - If there is only one string for a list of models, it corresponds to the name of the first composite PDFs. The other PDFs are plotted but they aren't legended
    models_types  : str
        type of each mode (one character for each model or for a list of models):
        - 'm' : model (sum) --> should always be the FIRST ONE !!
        - 's' : signal
        - 'b' : background
        used in the legend to indicate if it is a signal or a background component    
    linewidth     : str
        width of the fitted curve line
    colors        : str
        colors of the fitted curves
    title         : str
        title of the plot
    plot_pull     : bool
        if True, plot the pull diagram
    bar_mode_pull: bool
        if True, the pull diagram is plotted with bars instead of points + error abrs
    show_leg      : bool
        if True, show the legend
    fontsize_leg  : float
        fontsize of the legend
    loc_leg       : str
        position of the legend, loc argument specified by loc in plt.legend
    show_chi2     : bool
        if True, show the chi2 in the label of the x-axis of the pull diagram
    
    params        : dict[zfit.zfitParameter, float]
        Result 'result.params' of the minimisation of the loss function (given by `fit.fit.launch_fit`)
    latex_params  : 
        Dictionnary with the name of the params 
        Also indicated the branchs to show in the table among all the branchs in params
    colWidths     : [float, float, float, float]
        Width of the four columns of the table showing the result of the fit
        - first column: latex name of the parameter
        - second column: nominal value
        - third column: `$\pm$`
        - fourth column: error
    fontsize_res   : float
        fontsize of the text in the result table
    loc_res       : str
        position of the result table, loc argument specified in in plt.table 
     fig_name      : str
        name of the saved file
    folder_name   : str
        name of the folder where to save the plot
    data_name     : str
        name of the data used to constitute the name of the saved file, if fig_name is not specified. 
    save_fig      : str
        name of the figure to save
    pos_text_LHC    : dict, list or str
        passed to `plot.tools.set_text_LHCb()` as the `pos` argument
        Three possibilities
        - dictionnary with these keys
            - `'x'`: position of the text along the x-axis
            - `'y'`: position of the text along the y-axis
            - `'ha'`: horizontal alignment
            - `fontsize`: fontsize of the text
            - `text` : text to plot
        - list: [x, y, ha]
        - str: alignment 'left' or 'right'.
            - if 'left', x = 0.02 and y = 0.95
            - if 'right', x= 0.98 and y 0.95.
        These values are also the default values for the dictionnary input mode.
        These parameters are passed to `ax.text()`.
    
    Returns
    -------
    fig   : matplotlib.figure.Figure
        Figure of the plot (only if axis_mode=False)
    ax[0] : matplotlib.figure.Axes
        Axis of the histogram + fitted curves + table
    ax[1] : matplotlib.figure.Axes
        Axis of the pull diagram (only if plot_pull is True)
    
    """
    
    ## Create figure
    if plot_pull:
        fig = plt.figure(figsize=(12,10))
        gs = gridspec.GridSpec(2,1,height_ratios=[3,1])
        ax = [plt.subplot(gs[i]) for i in range(2)]
    else:
        fig, ax = plt.subplots(figsize=(8,6))
        ax = [ax]    
    
    
    ## Retrieve low,high (of x-axis)
    low = float(obs.limits[0])
    high = float(obs.limits[1])
    
    if low_hist is None:
        low_hist = low
    if high_hist is None:
        high_hist = high
    
    if latex_branch is None:
        latex_branch = string._latex_format(branch)
    
    
    ax[0].set_title(title, fontsize=25)
    
    ## plot 1D histogram of data
    # Histogram 
    counts, edges, centres, err = plot_hist_alone(ax[0], df[branch], n_bins,
                                               low_hist, high_hist, color, bar_mode, alpha = 0.1, weights=weights)
    
    
    # Label
    bin_width = get_bin_width(low_hist, high_hist, n_bins)
    set_label_hist(ax[0], latex_branch, unit, bin_width, fontsize=25)
    
    # Ticks
    pt.set_label_ticks(ax[0])
    pt.set_text_LHCb(ax[0], pos=pos_text_LHC)
    
    ## Plot fitted curve 
    if isinstance(models,list):
        model = models[0] # the first model is the "global" one
    else:
        model = models
    
        
    plot_scaling = get_plot_scaling(counts, low_hist, high_hist, n_bins)
    plot_fitted_curves(ax[0], models, plot_scaling, low, high, models_names=models_names, models_types=models_types,
                       line_width=2.5, colors=colors, fontsize_legend=fontsize_leg, loc_leg=loc_leg, show_legend=show_leg)
    
    pt.change_range_axis(ax[0], factor_max=1.1)    
    
    color_pull = colors if not isinstance(colors, list) else colors[0]
    ## Plot pull histogram
    if plot_pull:
        plot_pull_diagram(ax[1], model, counts, edges, centres, err, color=color_pull,
                          low=low, high=high, plot_scaling=plot_scaling, show_chi2=show_chi2, bar_mode_pull=bar_mode_pull)
    
    
    ## Plot the fitted parameters of the fit
    if params is not None:
        plot_result_fit(ax[0], params, latex_params=latex_params, fontsize=fontsize_res, colWidths=colWidths, loc=loc_res)
    
    # Save result
    plt.tight_layout()
    if save_fig:
        pt.save_fig(fig, fig_name,folder_name,f'{branch}_{data_name}_fit')
        
    if plot_pull:
        return fig, ax[0], ax[1]
    else:
        return fig, ax[0]

def plot_hist_fit_var (data, branch, latex_branch=None, unit=None, **kwargs):
    """ plot data with his fit
    data      : pandas.Series or list(pandas.Series)
        dataset to plot
    branch    : str
        name of the branch, for the name of the file
    latex_branch  : str
        name of the branch, for the label of the x-axis
    unit      : str
        unit of the branch
    **kwargs  : parameters passed to plot_hist_fit
    """
    df = DataFrame()
    df[branch] = data
    
    return plot_hist_fit(df, branch, latex_branch, unit, **kwargs) 
    



#################################################################################################
##################################### Automatic label plots #####################################
#################################################################################################  

def plot_hist_fit_auto(df, branch, cut_BDT=None, **kwargs):
    """ Retrieve the latex name of the branch and unit. Set the folder name to the name of the datasets.
    Then, plot 2d histogram with plot_hist_fit.
    
    Parameters
    ----------
    
    df            : pandas.Dataframe
        dataframe that contains the branch to plot
    branch : str
        branch (for instance: 'B0_M'), in dataframe
    cut_BDT         : float or str
        `BDT > cut_BDT` cut. Used in the name of saved figure.
    **kwargs : dict
        arguments passed in plot_hist_fit (except branch, latex_branch, unit)
    
    Returns
    -------
    fig   : matplotlib.figure.Figure
        Figure of the plot (only if axis_mode=False)
    ax[0] : matplotlib.figure.Axes
        Axis of the histogram + fitted curves + table
    ax[1] : matplotlib.figure.Axes
        Axis of the pull diagram (only if plot_pull is True)
    """
    
    ## Retrieve particle name, and branch name and unit.
#     particle, var = retrieve_particle_branch(branch)
    
#     latex_branch = branchs_params[var]['name']
#     unit = branchs_params[var]['unit']
#     name_particle = particle_names[particle]
    
    
    
    latex_branch, unit = pt.get_latex_branches_units(branch)
       
    # Title and name of the file with BDT    
    add_in_dic('fig_name', kwargs)
    add_in_dic('title', kwargs)
    add_in_dic('data_name', kwargs)
    
    kwargs['fig_name'] = pt._get_fig_name_given_BDT_cut(fig_name=kwargs['fig_name'], cut_BDT=cut_BDT, 
                                                        branch=branch, 
                                                        data_name=string.add_text(kwargs['data_name'],
                                                                                  'fit', '_', None))
    
    kwargs['title'] = pt._get_title_given_BDT_cut(title=kwargs['title'], cut_BDT=cut_BDT)
    
    
    # Name of the folder = name of the data
    add_in_dic('folder_name', kwargs)
    
    if kwargs['folder_name'] is None and kwargs['data_name'] is not None:
        kwargs['folder_name'] = kwargs['data_name']
    
    return plot_hist_fit(df, branch, latex_branch=latex_branch, unit=unit, **kwargs)
import numpy as np
import matplotlib.pyplot as plt

# Gives us nice LaTeX fonts in the plots
from matplotlib import rc, rcParams
rc('font', **{'family': 'serif', 'serif': ['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False

import HEA.plot.fit as pf
import HEA.plot.histogram as h
from HEA.tools import el_to_list
import HEA.plot.tools as pt
from HEA.tools import string
import HEA.fit.root as fr
from HEA.config import default_fontsize

from ROOT import RooFit

from HEA.tools.assertion import is_list_tuple
from HEA.tools.da import add_in_dic

def rooData_into_hist(data, models=None):
    """ Get the histogram of data and models,
    ready to be plotted with matplotlib.
    
    Parameters
    ----------
    data: RooDataSet
        Data that was fitted
    models: None or RooAbsPdf or list(RooAbsPdf)
        list of pdfs
    
    Returns
    -------
    x: array-like
        bin centres
    y: array-like
        counts
    y_err: array-like
        count errors
    model_hists: list()
    """
    # Get the rooVariable associated with the data
    rooVariable = data.get().first()
    
    # Get the low and high value and number of bins of this rooVariable
    low = rooVariable.getMin()
    high = rooVariable.getMax()
    n_bins = rooVariable.getBins()
    
    # Generate the histogram of data and pdfs   
    frame = rooVariable.frame()
    data.plotOn(frame)
    
    # Use Donal's function to get the histogram
    data_hist = frame.getObject(0)
    x, y, y_err = listsfromhist(data_hist, overunder=False, normpeak=False, xscale=1.0)
    
    # Do the same to get the models
    if models is not None:
        models = el_to_list(models)
        model_hists = []
        
        models[0].plotOn(frame)
        model_hist = frame.getObject(1)
        model_hists.append(model_hist)
        
        i = 2
        for model in models[1:]:
            models[0].plotOn(frame, RooFit.Components(model.GetName()))
            model_hist = frame.getObject(i)
            model_hists.append(model_hist)
            i+=1
    else:
        x_model = None
        y_models = None
        
    # Now, get the values for the models
    if models is None:
        return np.array(x), np.array(y), np.array(y_err[0])
    else:
        return np.array(x), np.array(y), np.array(y_err[0]), model_hists
    
#Create data array for plotting from the RooFit RooHist
def listsfromhist(hist, overunder=False, normpeak=False, xscale=1.0):
    """ By Donal Hill!
    see https://gitlab.cern.ch/lhcb-b2oc/analyses/Bd2DstDsst_Angular/-/blob/master/python/fitting/listsfromhist.py
    """
    from ROOT import TH1
    x, y, x_err, y_err = [ ], [ ], [ ], [ ]
    if isinstance(hist, TH1):
        bin_min = 0 if overunder else 1
        bin_max = hist.GetNbinsX()
        if overunder: bin_max += 1
        for n in range(bin_min, bin_max+1):
            x.append(hist.GetXaxis().GetBinCenter(n))
            y.append(hist.GetBinContent(n))
            x_err.append(hist.GetXaxis().GetBinWidth(n)*0.5)
            y_err.append(hist.GetBinError(n))
    else:
        npts = hist.GetN()
        data = { }
        xscale = 1.
        for v in ['X','Y','EXlow','EXhigh','EYlow','EYhigh']: #, 'EX', 'EY']:
            try:
                arr = getattr(hist, 'Get'+v)()
                scale = xscale if 'X' in v else 1.0
                tmp = [ arr[n]*scale for n in range(npts) ]
                data[v] = tmp
            except IndexError:
                pass
        x, y = data['X'], data['Y']
        if 'EXlow' in data and 'EXhigh' in data:
            x_err = [ data['EXlow'], data['EXhigh'] ]
        elif 'EX' in data:
            x_err = data['EX']

        if 'EYlow' in data and 'EYhigh' in data:
            y_err = [ data['EYlow'], data['EYhigh'] ]
        elif 'EY' in data:
            y_err = data['EY']

    if normpeak:
        ymax = max(y)
        y = [ yy/ymax for yy in y ]
        try:
            y_err = [
                    [ down/ymax for down in y_err[0] ],
                    [ up/ymax for up in y_err[1] ]
                    ]
        except:
            y_err = [ err/ymax for err in y_err ]

    #return x, y, x_err, y_err
    return x, y, y_err


def plot_hist_fit_root(data, branch, models=None,
                       PDF_names=None,
                       models_types=None,
                       recompute_err=False,
                       **kwargs):
    """ Plot complete histogram with fitted curve, pull histogram and results of the fits. Save it in the plot folder.

    Parameters
    ----------
    data          : RooDataSet
        Data to plot
    models_names : str or list(str) or list(list(str))
        passed to :py:func:`plot_fitted_curves`,
        Labels of the models
    models_types  : str
        passed to :py:func:`plot_fitted_curves`
        (Model, signal, background or nothing)
    *args, **kwargs:
        passed to py:func:`HEA.plot.fit.plot_hist_fit`

    Returns
    -------
    fig   : matplotlib.figure.Figure
        Figure of the plot (only if ``axis_mode`` is ``False``)
    ax[0] : matplotlib.figure.Axes
        Axis of the histogram + fitted curves + table
    ax[1] : matplotlib.figure.Axes
        Axis of the pull diagram (only if ``plot_pull`` is ``True``)
    """
        
    ## Get the fitted histogram & models =====================================
    rooVariable = data.get().first()
    if branch is None:
        branch = rooVariable.GetName()
    
    
    if models is None:
        x, y, y_err = rooData_into_hist(data, models)
        
    else:
        models = el_to_list(models)        
        x, y, y_err, model_hists = rooData_into_hist(data, models)
        edges = h.get_edges_from_centres(x)
        model_hists = el_to_list(model_hists, len(model_hists))
        n_models = len(model_hists)
        
        
        y_models = []

        x_model = np.linspace(start=edges[0], stop=edges[-1], num=1000)
        for model_hist in model_hists:
            y_models.append(fr.evaluate_pdf_root(x_model, model_hist))
        
        # Model types
        if models_types=='auto':
            models_types = []
            for model in models:
                models_types.append(model.GetTitle())
        
        # PDF names (gauss, horn, hill, CB, ...)
        if PDF_names != 'auto':
            PDF_names = el_to_list(PDF_names, n_models)
        else:
            PDF_names = []
            for model in models:
                PDF_names.append(model.GetName())
                        
    if recompute_err:
        y_err = np.sqrt(np.abs(y))
    
    
    if models is not None:
        if isinstance(models, list):
            model = models[0]
        else:
            model = models


        y_model_pull = fr.evaluate_pdf_root(x, model_hists[0])
        ndof = fr.get_n_dof_model_root(model, rooVariable)
        # Plot fitted curve
    else:
        ndof = 0
        y_models = None
        
    return pf.plot_hist_fit_counts(
        data=[y, y_err], branch=branch,
        models=[x_model, y_models, y_model_pull], models_types=models_types,
        PDF_names=PDF_names, ndof=ndof,
        centres=x, edges=edges,
        **kwargs
    )
    
    

# def plot_hist_fit_root(data, models=None, latex_branch=None, unit=None,
#                       weights=None, PDF_names=None,
#                        models_types=None,
#                       color='black', bar_mode=False,
#                       models_names=None,
#                       linewidth=2.5, colors=None,
#                       bar_mode_pull=True, 
#                       params=None, latex_params=None, 
#                       loc_res='upper right', 
#                       fig_name=None, folder_name=None, data_name=None,
#                        fontsize_leg=default_fontsize['legend'],
#                        loc_leg='upper left', show_leg=None,
#                       save_fig=True, pos_text_LHC=None, 
#                        kwargs_res={}, stack=False,
#                        ndof=None, branch=None, data_label=None,
#                        lim_pull=None, recompute_err=False, 
#                        **kwargs):
#     """ Plot complete histogram with fitted curve, pull histogram and results of the fits. Save it in the plot folder.

#     Parameters
#     ----------
#     data          : RooDataSet
#         Data to plot
#     models        : RooAbsPdf or list(RooAbsPdf)
#         List of PDFs to plot
#     latex_branch  : str
#         latex name of the branch, to be used in the xlabel of the plot
#     unit            : str
#         Unit of the physical quantity
#     weights         : numpy.array
#         weights passed to ``plt.hist``
#     color         : str
#         color of the histogram
#     bar_mode     : bool

#         * if True, plot with bars
#         * else, plot with points and error bars
    
#     models_names : str or list(str) or list(list(str))
#         passed to :py:func:`plot_fitted_curves`
#     models_types  : str
#         passed to :py:func:`plot_fitted_curves`
#     linewidth     : str
#         width of the fitted curve line
#     colors        : str
#         colors of the fitted curves
#     bar_mode_pull: bool or None
#         if ``True``, the pull diagram is plotted with bars instead of points + error bars.
#         if None, don't plot the pull diagram.
#     show_leg      : bool
#         if ``True``, show the legend
#     fontsize_leg  : float
#         fontsize of the legend
#     loc_leg       : str
#         position of the legend, ``loc`` argument in ``plt.legend``
#     params        : dict[zfit.zfitParameter, float]
#         Result ``result.params`` of the minimisation of the loss function (given by :py:func:`HEA.fit.fit.launch_fit`)
#     latex_params  :
#         Dictionnary with the name of the params.
#         Also indicated the branchs to show in the table among all the branchs in params
#     fig_name      : str
#         name of the saved file
#     folder_name   : str
#         name of the folder where to save the plot
#     data_name     : str
#         name of the data used to constitute the name of the saved file, if ``fig_name`` is not specified.
#     save_fig      : str
#         name of the figure to save
#     pos_text_LHC    : dict, list or str
#         passed to :py:func:`HEA.plot.tools.set_text_LHCb` as the ``pos`` argument.
#     kwargs_res    : dict
#         arguments passed to :py:func:`plot_result_fit`
#     **kwargs:
#         passed to py:func:`plot_fitted_curves`

#     Returns
#     -------
#     fig   : matplotlib.figure.Figure
#         Figure of the plot (only if ``axis_mode`` is ``False``)
#     ax[0] : matplotlib.figure.Axes
#         Axis of the histogram + fitted curves + table
#     ax[1] : matplotlib.figure.Axes
#         Axis of the pull diagram (only if ``plot_pull`` is ``True``)
#     """
        

#     given_counts = is_list_tuple(data)
#     models_given_by_counts = False
#     colors = el_to_list(colors)
    
#     if stack:
#         model_bar_mode = False
    
#     # Get the name of the branch
#     if given_counts:
#         assert branch is not None
#     else:
#         rooVariable = data.get().first()
#         if branch is None:
#             branch = rooVariable.GetName()
    
    
#     # Get the histograms
#     if is_list_tuple(data):
#         x = data[0]
#         y = data[1]
#         y_err = data[2]
        
#         models_given_by_counts = is_list_tuple(models)
#         assert models_given_by_counts
# #         if models_given_by_counts:

#         if len(models)==1:
#             y_models = models[0]
#             y_models = el_to_list(y_models)
#             x_model = x
#             model_bar_mode = True
            
            
#         elif len(models)==3:
#             x_model, y_models, y_model_pull = models
#             y_models = el_to_list(y_models)
        
#         n_models = len(y_models)
        
#         if y_models[0] is None:
#             y_models[0] = np.array([sum(y) for y in zip(*tuple(y_models[1:]))])
#             if len(colors) < n_models:
#                 colors = ['b'] + colors
#                 models_names = el_to_list(models_names, n_models)
#                 models = el_to_list(models, n_models)
#             if len(models_names) < n_models:
#                 models_names = [None] + models_names
            
#         if len(models)==1:
#             y_model_pull = y_models[0]
        
        
        
#     elif models is None:
#         x, y, y_err = rooData_into_hist(data, models)
#     else:
#         models = el_to_list(models)        
#         x, y, y_err, model_hists = rooData_into_hist(data, models)
#         model_hists = el_to_list(model_hists, len(model_hists))
#         n_models = len(model_hists)
        
#         if models_types is None:
#             for model in models:
#                 models_types = model.GetTitle()
    
#     if recompute_err:
#         y_err = np.sqrt(np.abs(y))
    
#     edges = h.get_edges_from_centres(x)
#     low = edges[0]
#     high = edges[-1]
    
#     plot_pull = (bar_mode_pull is not None) and (models is not None)
#     fig, ax = pf.create_fig_plot_hist_fit(plot_pull)
    
#     if latex_branch is None:
#         latex_branch = string._latex_format(branch)
#     # Plot the histogram
#     h.plot_hist_alone_from_hist(ax[0], counts=y, err=y_err, 
#                               color=color,
#                               centres=x,
#                               bar_mode=False, alpha=0.1,
#                               show_ncounts=False,
#                               weights=weights, label=data_label,
#                               orientation='vertical')
    
#     # Label
#     bin_width = x[1] - x[0]
#     h.set_label_hist(ax[0], latex_branch, unit, bin_width, fontsize=25)
#     # Ticks
#     pt.set_label_ticks(ax[0])
#     pt.set_text_LHCb(ax[0], pos=pos_text_LHC)
    
    
#     # Plot fitted curve
#     if models is not None:
#         models_names = el_to_list(models_names, n_models)
        
#         if models_types is None:
#             models_types = 'n'
#         models_types = el_to_list(models_types, n_models)
        
#         if PDF_names != 'auto':
#             PDF_names = el_to_list(PDF_names, n_models)

        
#         if not models_given_by_counts:
#             y_models = []
            
#             x_model = np.linspace(start=edges[0], stop=edges[-1], num=1000)
#             for model_hist in model_hists:
#                 y_models.append(fr.evaluate_pdf_root(x_model, model_hist))
        
#         start = 0
#         if y_models[0] is None:
#             start = 1
        
#         for i in range(start, n_models):
            
#             if PDF_names is None:
#                 PDF_name = None
#             elif PDF_names=="auto":
#                 PDF_name = model.GetName()
#             else:
#                 PDF_name = PDF_names[i]
            
#             if stack: 
#                 model_mode = 'bar' if model_bar_mode else 'fillbetween'
#             else:
#                 model_mode = None
            
#             if stack and i!=0: 
#                 model_counts = np.array([sum(y) for y in zip(*tuple(y_models[i:]))])
#                 pf.plot_fitted_curve_from_hist(
#                     ax[0], x_model, 
#                     model_counts,
#                     PDF_name=PDF_name,
#                     model_name=models_names[i], 
#                     model_type=models_types[i],
#                     color=colors[i], 
#                     alpha=1,
#                     mode=model_mode,
#                     **kwargs
#                 )
                    
#             elif y_models[i] is not None: # plot the line of the full model (i=0)
                
                    
#                 pf.plot_fitted_curve_from_hist(
#                     ax[0], x_model, y_models[i],
#                     PDF_name=PDF_name,
#                     model_name=models_names[i],
#                     model_type=models_types[i],
#                     color=colors[i], 
#                     linestyle='-', linewidth=1.,
#                     alpha=1, 
#                     mode=model_mode,
#                     **kwargs
#                 )
            
#     pt.change_range_axis(ax[0], factor_max=1.1)
    
    
#     if plot_pull:
#         if not models_given_by_counts:
#             if isinstance(models, list):
#                 model = models[0]
#             else:
#                 model = models
        
#             y_model_pull = fr.evaluate_pdf_root(x, model_hists[0])
            
#         color_pull = colors if not isinstance(colors, list) else colors[0]
        
#         pull =  pf.plot_pull_diagram_from_hist(
#                 ax=ax[1], 
#                 fit_counts=y_model_pull, counts=y,
#                 edges=edges, err=y_err, 
#                 color=color_pull,
#                 bar_mode_pull=bar_mode_pull)
        
#         if models_given_by_counts:
#             ndof = 0
#         elif ndof is None:
#             ndof = fr.get_n_dof_model_root(model, rooVariable)
#         pf.print_fit_info(centres=x, fit_counts=y_model_pull, counts=y, pull=pull, ndof=ndof, err=y_err)
#         ax[1].set_xlim(low, high)
        
#         if lim_pull is not None:
#             if isinstance(lim_pull, float) or isinstance(lim_pull, int):
#                 lim_pull = (-lim_pull,lim_pull)
#             ax[1].set_ylim(lim_pull)
        
        
#     # Plot the fitted parameters of the fit
#     if params is not None:
#         pf.plot_result_fit(ax[0], params, latex_params=latex_params,
#                         **kwargs_res)

#     # Legend
#     if show_leg is None
#         show_leg = models_names is not None
#     if show_leg:
#         ax[0].legend(fontsize=fontsize_leg, loc=loc_leg)
    
#     ax[0].set_xlim(low, high)
    
    
#     # Save result
#     plt.tight_layout()
#     if save_fig:
#         pt.save_fig(fig, fig_name, folder_name, f'{string.add_text(branch, data_name)}_fit')

#     if plot_pull:
#         return fig, ax[0], ax[1]
#     else:
#         return fig, ax[0]
    
    
def plot_hist_fit_root_auto(data, cut_BDT=None, **kwargs):
    """Retrieve the latex name of the branch and unit. 
    Set the folder name to the name of the datasets.
    Then, plot with :py:func:`plot_hist_fit_root`.
    
    Parameters
    ----------
    data          : RooDataSet
        Data to plot
    **kwargs      : passed to :py:func:`plot_hist_fit_root`
    
    Returns
    -------
    see :py:func:`plot_hist_fit_root`
    """
    add_in_dic('branch', kwargs, None)
    
    given_counts = is_list_tuple(data)
    
    if given_counts:
        assert kwargs['branch'] is not None
        branch = kwargs['branch']
    else:
        rooVariable = data.get().first()
        if kwargs['branch'] is None:
            branch = rooVariable.GetName()
        else:
            branch = kwargs['branch']
    
    latex_branch, unit, kwargs = pf._core_plot_hist_fit_auto(branch, cut_BDT, kwargs)
    kwargs['']
    del kwargs['title']
    
    return plot_hist_fit_root(data, 
                              latex_branch=latex_branch, unit=unit,
                              **kwargs)
    
    
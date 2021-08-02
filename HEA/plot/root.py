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
            models[0].Print()
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
        passed to :py:func:`HEA.plot.fit.plot_hist_fit`

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
    del kwargs['title']
    
    return plot_hist_fit_root(data, 
                              latex_branch=latex_branch, unit=unit,
                              **kwargs)
    
    
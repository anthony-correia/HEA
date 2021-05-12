"""
Functions for plotting with zfit.
"""

##########################################################################
########################################## Scaling PDF ###################
##########################################################################

from operator import mul
from functools import reduce

from HEA.tools.df_into_hist import (
    dataframe_into_hist1D,
    dataframe_into_hist2D,
    dataframe_into_histdD
)
import HEA.plot.fit as pf

from HEA.tools.da import (
    el_to_list, get_element_list,
)

from HEA.tools import assertion
from HEA.fit import PDF
 
import numpy as np

def _get_plot_scaling(counts, low, high, n_bins):
    """Return plot_scaling, the factor to scale the curve fit to unormalised histogram
    
    Parameters
    ----------
    counts : np.array or list
        number of counts in the histogram of the fitted data
    low    : float or list(float)
        low limit of the distribution
    high   : float or list(float)
        high limit of the distribution
    n_bins : int or list(int)
        number of bins
    
    Returns
    -------
    plot_scaling: float
        parameter to multiply the PDF by to get 
        the right yield    
    """
    high = el_to_list(high)
    low = el_to_list(low)
    n_bins = el_to_list(n_bins)
    prod_space = reduce(mul, [high[i] - low[i] for i in range(len(low))])
    prod_n_bins = reduce(mul, n_bins)
    
    return counts.sum() * prod_space / prod_n_bins


def partial_pdf(model, x, branch):
    """
    Evaluate a multi-dimensional PDF
    by integrating out the dimensions
    that are not in ``branch``
    
    Parameters
    ----------
     model: zfit.pdf.BasePDF
        Multi-dimensional model (PDF)
    x       : array-like
        Array of numbers where to evaluate the PDF
    branch: str or list(str)
        branches where the PDF is evaluated.
        If the PDF is multi-dimensional, 
        the other dimensions are integrated out.
    
    Returns
    -------
    y : array-like
        PDF evaluated at x
    """
    
    branches = el_to_list(branch)
    
    import zfit
    
    other_obs_list = []
    branch_obs_list = []
    
    for i in range(model._N_OBS):
        sub_obs = zfit.Space(
            model.obs[i], 
            limits=(model.space.limits[0].flatten()[i],
                   model.space.limits[1].flatten()[i])
        )
        
        if model.obs[i] in branches:
            branch_obs_list.append(sub_obs)
        else:
            other_obs_list.append(sub_obs)

    other_obs = reduce(mul, other_obs_list)
    branch_obs = reduce(mul, branch_obs_list)
    x_adapted = zfit.Data.from_numpy(array=np.array(x), obs=branch_obs)

    return model.partial_integrate(x_adapted, limits=other_obs)

def frac_model(x, model, frac=1., branch=None):
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
    branch: str or list(str)
        branches where the PDF is evaluated.
        If the PDF is multi-dimensional, 
        the other dimensions are integrated out.

    Returns
    -------

    np.array(float)
        list of the values of the pdf of the model evaluated in x,
        multiplied by ``frac``

    """
    print("Dim of the pdf:", model.n_obs)
    partial_needed = (model.n_obs > 1) and not (assertion.is_list_tuple(branch) and model.n_obs==len(branch))

    if partial_needed:
        y = partial_pdf(model, x, branch)
    else:
        y = model.pdf(x)
    return (y * frac).numpy()

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

    PDF_name = model.name[:marker]
#     assert model_name in name_PDF, f"{model_name} is not defined in {list(name_PDF.keys())}"
#     label_model = name_PDF[model_name]

#     return label_model
    return PDF_name

def get_limits_from_obs(obs, branch=None):
    """
    Get the low and high limits from a zfit space
    
    Parameters
    ----------
    obs           : zfit.Space
        Space used for the fit
    ranch        : str
        name of the branch to plot, which was fitted to
    """
    low = obs.rect_limits[0].flatten()
    high = obs.rect_limits[1].flatten()
    if len(low)==1 or branch is None:
        return low[0], high[0]
    else:
        branch_index = obs.obs.index(branch)
        return low[branch_index], high[branch_index]

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
    
##########################################################################
#################################### Sub-plotting functions ##############
##########################################################################

def _get_single_model_zfit(x, model, plot_scaling,
                           model_name=None,  
                           branch=None,
                           frac=1., **kwargs):
    """ Plot the models recursively
    with a label for the curve ``"{name of the PDF (e.g., Gaussian, ...)} - {type of the model, e.g., signal ...} {Name of the model, e.g., "B0->Dst Ds"}"`` (if ``model_name`` is specified)
    
    Parameters
    ----------
    x             : numpy.numpy(float)
        points of the x-axis where to evaluate the pdf of the model to plot
    model        : zfit.pdf.BasePDFfrac
        just a zfit model
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
        
    y = frac_model(x, model, frac=frac, branch=branch) * plot_scaling    
    return y, model_name, PDF_name, kwargs


def _get_models_zfit(x, models, plot_scaling, 
                     models_types=None, models_names=None, frac=1.,
                     PDF_level=0, colors=['b', 'g', 'gold', 'magenta', 'orange'],
                     linestyles=['-', '--', ':', '-.'], 
                     branch=None,
                     list_models=[], list_models_names=[], list_models_types="",
                     list_PDF_names=[], list_kwargs={},
                    ):
    """ Plot the models recursively
    with a label for each curve ``"{name of the PDF (e.g., Gaussian, ...)} - {type of the model, e.g., signal ...} {Name of the model, e.g., "B0->Dst Ds"}"`` (if the corresponding model_name is specified)
    
    
    Parameters
    ----------
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
    
    branch: str
        branch to plot (used for multidimensional PDFs)
    list_models  :list(array-like)
        list of the counts of the models. 
        Built recursively.
    list_models_names : list(str)
        list of the names of the models. 
        Built recursively.
    list_models_types: str
        ``models_types`` as described in :py:func:`get_fitted_curves_zfit`. 
        Built recursively.
    list_PDF_names: list(str)
        List of the names to identify the PDF ("gauss", ...). 
        Built recursively.
    list_kwargs: dict
        associates to a parameter a list, whose each element 
        characterises a model. 
        Built recursively.
    
    
    Returns
    -------
    list_models  :list(array-like)
        list of the counts of the models. 
    list_models_names : list(str)
        list of the names of the models. 
    list_models_types: str
        ``models_types`` as described in :py:func:`get_fitted_curves_zfit`. 
    list_PDF_names: list(str)
        List of the names to identify the PDF ("gauss", ...). 
    list_kwargs: dict
        associates to a parameter a list, whose each element 
        characterises a model. 
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
            
            # Get the updated lists
            list_models, list_models_names, list_models_types, list_PDF_names, list_kwargs = \
                _get_models_zfit(
                    x=x, models=model, plot_scaling=plot_scaling, 
                    models_types=model_type, models_names=model_name, 
                    frac=applied_frac, PDF_level=PDF_level, 
                    colors=color,
                    linestyles=linestyles, 
                    branch=branch,
                    list_models=list_models, list_models_names=list_models_names,
                    list_models_types=list_models_types,
                    list_PDF_names=list_PDF_names, list_kwargs=list_kwargs)

            
            

    else:  # if there is only one PDF to plot
        if PDF_level >= 2:
            alpha = 0.5
        else:
            alpha = 1
        
        y, model_name, PDF_name, plot_kwargs = \
            _get_single_model_zfit(x=x, model=models, 
                                   plot_scaling=plot_scaling, 
                                   model_name=models_names,
                                   frac=frac, colors=colors,
                                   linestyle=linestyles[PDF_level], 
                                   alpha=alpha,
                                   branch=branch
                                  )
        # update the lists
        list_models.append(y)
        list_models_names.append(model_name)
        list_PDF_names.append(PDF_name)
        list_models_types += models_types

        for key, value in plot_kwargs.items():
            if key not in list_kwargs:
                list_kwargs[key] = []
            list_kwargs[key].append(value)
    
    return list_models, list_models_names, list_models_types, list_PDF_names, list_kwargs
        
        


def get_fitted_curves_zfit(x, models, plot_scaling, low, high,
                       models_names=None, models_types=None,
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
    **kwargs        : dict
        passed to :py:func:`_get_single_model_zfit`

    """
    models = el_to_list(models, 1)
    
    models_names = el_to_list(models_names, len(models))

    if models_types is None:
        models_types = 'm'
        if len(models_names) >= 2:
            models_types += 's'
            models_types += 'b' * (len(models_names) - 2)

    return _get_models_zfit(x, models=models, plot_scaling=plot_scaling,
                 models_types=models_types, models_names=models_names,
                 list_models=[], list_models_names=[], list_PDF_names=[], list_kwargs={},
                 list_models_types="",
                 **kwargs)

    
# def compute_chi2_model_data(data, model, n_bins):
#     """ Compute the reduced chi2 between data and a zfit model
    
#     Parameters
#     ----------
#     data: array-like
#         data that the model has been fitted to
#     model: zfit model
#         PDF fitted to data
#     n_bins: int
#         number of bins which is 
#     """

def plot_hist_fit(df, branch, weights=None,
                  obs=None, n_bins=50, low_hist=None, high_hist=None,
                  models=None, models_names=None, models_types=None,
                  colors=None, auto_PDF_names=True, auto_models_types=True,
                  **kwargs):
    """ Plot complete histogram with fitted curve,
    pull histogram and results of the fits, where zfit models are used.

    Parameters
    ----------
    df            : pandas.Dataframe
        dataframe that contains the branch to plot
    branch        : str
        name of the branch to plot, which was fitted to
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
    models        : zfit.pdf.BasePDF or list(zfit.pdf.BasePDF) or list(list(zfit.pdf.BasePDF)) or ...
        passed to :py:func:`get_fitted_curves_zfit`
    models_names : str or list(str) or list(list(str))
        passed to :py:func:`get_fitted_curves_zfit`
    models_types  : str
        passed to :py:func:`get_fitted_curves_zfit`
    colors        : str
        colors of the fitted curves
    **kwargs:
        passed to :py:func:`HEA.plot.fit.plot_hist_fit_counts`

    Returns
    -------
    fig   : matplotlib.figure.Figure
        Figure of the plot (only if ``axis_mode`` is ``False``)
    ax[0] : matplotlib.figure.Axes
        Axis of the histogram + fitted curves + table
    ax[1] : matplotlib.figure.Axes
        Axis of the pull diagram (only if ``plot_pull`` is ``True``)
    """
    
    ## Retrieve low,high (of x-axis) ===========================
    low, high = get_limits_from_obs(obs, branch)
    
    if low_hist is None:
        low_hist = low
    
    if high_hist is None:
        high_hist = high


    # Get 1D histogram of the fitted sample ===================
    # Histogram
    data, edges, density = dataframe_into_hist1D(
        df, branch=branch,
        low=low_hist, high=high_hist, n_bins=n_bins, 
        weights=weights, density=False)
    counts, err = data
    centres = (edges[1:] + edges[:-1])/2
    
    # Get fitted curves =======================================
    if isinstance(models, list):
        model = models[0]  # the first model is the "global" one
    else:
        model = models

    plot_scaling = _get_plot_scaling(counts, low_hist, high_hist, n_bins)
    
    x_model = np.linspace(low, high, 1000)
    
    list_models, list_models_names, list_models_types, list_PDF_names, list_kwargs = \
        get_fitted_curves_zfit(x=x_model,
            models=models, plot_scaling=plot_scaling, 
            low=low, high=high, 
            models_names=models_names, models_types=models_types,
            colors=colors, branch=branch
        )

    # Get model for pull histogram ==============================
    if models is not None:
        fit_counts_pull = frac_model(centres, model, plot_scaling, branch=branch)
        ndof = PDF.get_n_dof_model(model)
    else:
        ndof = 0
        fit_counts_pull = None
    
    if not auto_PDF_names:
        list_PDF_names = None
    if not auto_models_types:
        list_models_types = None
        
    return pf.plot_hist_fit_counts(
        data=[counts, err], branch=branch,
        models=[x_model, list_models, fit_counts_pull], 
        models_types=list_models_types,
        PDF_names=list_PDF_names, ndof=ndof,
        models_names=list_models_names,
        edges=edges, **kwargs,
        **list_kwargs
    )
        

def get_counts_fit_counts_2d(
    branches, df, model, obs,
    n_bins=20):
    """
    Get the counts, edges and fit counts
    
    Parameters
    ----------
    branches: list(str, str)
        two branches to plot
    df: pd.DataFrame or list(array-like, array-like)
        Dataframes associated with the two branches.
        If it is just a dataframe, it should contain
        the two columns given by ``branches``
    model: zfit.BasePDF
        zfit PDF
    obs: zfit.Space or tuple(2-tuple, 2-tuple)
        tuple (2 low values, 2 high values)
        (1 value for each branch)
    n_bins: int or list(int, int)
        number of bins

    Returns
    -------
    counts, err: array-like
        Fitted sample, binned,
        and uncertainty on these counts
    fit_counts: array-like
        Fitted pdf, binned
    xedges, yedges: array-like
        Common edges
   
    """
    lows = [None, None]
    highs = [None, None]
    n_bins = el_to_list(n_bins, 2)
    for i in range(2):
        if assertion.is_list_tuple(obs):
            lows[i] = obs[i][0]
            highs[i] = obs[i][1]
        else:
            lows[i], highs[i] = get_limits_from_obs(obs, branches[i])
    
    counts, xedges, yedges = dataframe_into_hist2D(
        branches, df, 
        low=lows, high=highs, 
        n_bins=n_bins
    )
    
    err = np.sqrt(counts)
    xcentres = (xedges[1:] + xedges[:-1]) / 2
    ycentres = (yedges[1:] + yedges[:-1]) / 2
    # mesh where to evaluate the PDF
    mesh = np.array(np.meshgrid(xcentres, ycentres)).T
    plot_scaling = _get_plot_scaling(
        counts, n_bins=n_bins, 
        low=[xedges[0], yedges[0]], 
        high=[xedges[-1], yedges[-1]]
    )
    
    fit_counts = frac_model(mesh, model, frac=plot_scaling, branch=branches).reshape(len(xcentres), len(ycentres)).T
    # fit_counts = plot_scaling * fit_counts

    return counts, err, fit_counts, xedges, yedges

def get_counts_fit_counts_dD(
    branches, df, model, obs,
    n_bins=20):
    """
    Get the counts, edges and fit counts
    
    Parameters
    ----------
    branches: list(str, str)
        two branches to plot
    df: pd.DataFrame or list(array-like, array-like)
        Dataframes associated with the two branches.
        If it is just a dataframe, it should contain
        the two columns given by ``branches``
    model: zfit.BasePDF
        zfit PDF
    obs: zfit.Space or tuple(2-tuple, 2-tuple)
        tuple (2 low values, 2 high values)
        (1 value for each branch)
    n_bins: int or list(int, int)
        number of bins

    Returns
    -------
    counts, err: array-like
        Fitted sample, binned,
        and uncertainty on these counts
    fit_counts: array-like
        Fitted pdf, binned
    xedges, yedges: array-like
        Common edges
   
    """
    dim = len(branches) # dimension

    lows = [None for i in range(dim)]
    highs = [None for i in range(dim)]
    n_bins = el_to_list(n_bins, 2)
    for i in range(dim):
        if assertion.is_list_tuple(obs):
            lows[i] = obs[i][0]
            highs[i] = obs[i][1]
        else:
            lows[i], highs[i] = get_limits_from_obs(obs, branches[i])
    
    counts, edges = dataframe_into_histdD(
        branches, df, 
        low=lows, high=highs, 
        n_bins=n_bins
    )
    
    err = np.sqrt(counts)
    centres = []
    for subedges in edges:
        centres.append((subedges[1:] + subedges[:-1]) / 2)
    # mesh where to evaluate the PDF
    mesh = np.array(np.meshgrid(*tuple(centres))).T
    plot_scaling = _get_plot_scaling(
        counts, n_bins=n_bins, 
        low=lows, 
        high=highs
    )

    list_n_bins = []
    for subcentres in centres:
        list_n_bins.append(len(subcentres))
    
    

    fit_counts = frac_model(mesh, model, frac=plot_scaling, branch=branches).reshape(*tuple(list_n_bins)).T
    # fit_counts = plot_scaling * fit_counts

    return counts, err, fit_counts, edges

def plot_hist_fit2d(branches, df, model, obs,
                n_bins=20,
                **kwargs):
    """ Produce 3 separate histograms:
    
    * Counts in the sample
    * Fitted counts
    * Pull 
    
    Parameters
    ----------
    branches: list(str, str)
        two branches to plot
    df: pd.DataFrame or list(array-like, array-like)
        Dataframes associated with the two branches.
        If it is just a dataframe, it should contain
        the two columns given by ``branches``
    model: zfit.BasePDF
        zfit PDF
    obs: zfit.Space or tuple(2-tuple, 2-tuple)
        tuple (2 low values, 2 high values)
        (1 value for each branch)
    n_bins: int or list(int, int)
        number of bins
    **kwargs : dict
        passed to :py:func:`HEA.plot.fit.plot_hist_fit2d_counts`
    
    Returns
    -------
    fig   : matplotlib.figure.Figure
        Figure of the plot (only if ``axis_mode`` is ``False``)
    ax[0] : matplotlib.figure.Axes
        Axis of the histogram + fitted curves + table
    ax[1] : matplotlib.figure.Axes
        Axis of the pull diagram (only if ``plot_pull`` is ``True``)
    """
    counts, err, fit_counts, edges = get_counts_fit_counts_dD(
        branches=branches, df=df, model=model, obs=obs,
        n_bins=20,
    )

    xedges = edges[0]
    yedges = edges[1]
    
    return pf.plot_hist_fit2d_counts(branches=branches, counts=counts, fit_counts=fit_counts,
                           xedges=xedges, yedges=yedges, err=err,
                           **kwargs)
                                 
    
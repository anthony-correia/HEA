"""
Tool functions for distributions.
"""

import numpy as np
import pandas as pd

## CHI2 =============================================================================

def get_chi2(fit_counts, counts, err=None):
    """ Get the :math:`\\chi^2` of a fit

    Parameters
    ----------
    fit_counts : np.array or list
        fitted number of counts
    counts     : np.array or list
        number of counts in the histogram of the fitted data

    Returns
    -------
    returns : float
        :math:`\\chi^2` of the fit
    """
    if err is None:
        square_err = np.abs(counts)
    else:
        square_err = np.square(err)
        
    diff = np.square(fit_counts - counts)
    #n_bins = len(counts)
    diff = np.divide(diff, square_err,
                     out=np.zeros_like(diff), where=counts != 0)
    chi2 = np.sum(diff)  # sigma_i^2 = mu_i
    return chi2


def get_reduced_chi2(fit_counts, counts, n_dof=0, err=None):
    """ Get the reduced :math:`\\chi^2` of a fit

    Parameters
    ----------
    fit_counts : np.array or list
        fitted number of counts
    counts     : np.array or list
        number of counts in the histogram of the fitted data
    n_dof      : int
        number of d.o.f. in the model

    Returns
    -------
    returns    : float
        reduced :math:`\\chi^2` of the fit
    """
    n_bins = len(counts)

    return get_chi2(fit_counts, counts, err=err) / (n_bins - n_dof)


def get_mean(pull):
    """ Get the mean of an array, excluding non-valid values.

    Parameters
    ----------
    pull : numpy.array
        Array

    Returns
    -------
    mean_pull: float
        mean of the array
    """
    return np.mean(pull[np.isfinite(pull)])


def get_std(pull):
    """ Get the standard deviation of an array, excluding non-valid values.

    Parameters
    ----------
    pull : numpy.array
        Array

    Returns
    -------
    mean_pull: float
        Standard deviation of the array
    """
    return np.std(pull[np.isfinite(pull)])



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
    return float((high - low) / n_bins)

def weighted_qcut(data, weights, q):
    """Return weighted quantile cuts from a given series, values.
    Adapted from
    https://stackoverflow.com/questions/45528029/python-how-to-create-weighted-quantiles-in-pandas
    """
    assert isinstance(q, int)
    quantiles = np.linspace(0, 1, q + 1)

    index_sort = data.argsort().values
    sorted_data = np.array(data)[index_sort]

    if weights is None:
        sorted_weights = np.ones(len(data))
    else:
        sorted_weights = np.array(weights)[index_sort]

    order = sorted_weights.cumsum()

    df_p = pd.DataFrame()
    df_p['data'] = sorted_data
    df_p['weights'] = sorted_weights
    df_p['weighted_qcut'] = pd.cut(order / order[-1], quantiles)

    df_p_bins = df_p.groupby('weighted_qcut').agg({
        'weights': 'sum',
        'data': ['min', 'max']
    })

    edges = df_p_bins['data']['min'].values
    edges = np.append(edges, df_p_bins['data']['max'].values[-1])
    counts = df_p_bins['weights'].values.flatten()
    return edges, counts


def get_count_err(data, n_bins, low=None, high=None, weights=None,
                  cumulative=False, density=False,
                  quantile_bin=False,
                  **kwargs):
    """ get counts and error for each bin

    Parameters
    ----------
    data          : pandas.Series
        data to plot
    n_bins        : int or array-like
        number of bins or bin edges
    low           : float
        low limit of the distribution
    high          : float
        high limit of the distribution
    weights       : pandas.Series, numpy.array
        weights of each element in data
    cumulative    : bool
        if ``True``, return the cumulated event counts.
    **kwargs      : dict
        passed to ``np.histogram``

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
    
    if low is None or high is None:
        range_v = None
    else:
        range_v = (low, high)
    
    if quantile_bin:
        assert isinstance(n_bins, int)
        data_cut = data
        weights_cut = weights
        
        if low is not None:
            cut = data_cut > low
            data_cut = data_cut[cut]
            if weights is not None:
                weights_cut = weights_cut[cut]
        if high is not None:
            cut = data_cut < high
            data_cut = data_cut[cut]
            if weights is not None:
                weights_cut = weights_cut[cut]
        
        edges, counts = weighted_qcut(data_cut, 
                                      weights_cut, q=n_bins)
        
#         if weights is None:
#             out = pd.qcut(data_cut, q=n_bins)
#         else:
#             out = weighted_qcut(data_cut, weights, 
#                                 q=n_bins, **kwargs)
            
#         histogram = out.value_counts(sort=False)
        
#         # edges
#         # (haven't find any other way than a loop)
#         edges = []
#         for interval in histogram.index:
#             edges.append(interval.left)
#         edges.append(interval.right)
#         edges = np.array(edges)
        
        if low is not None:
            edges[0] = low
        if high is not None:
            edges[-1] = high
        
        n_bins = edges
        
        # counts
#         counts = np.array(histogram.values)

    counts, edges = np.histogram(data, range=range_v, 
                                 bins=n_bins, weights=weights, 
                                 **kwargs)
    
    
    centres = (edges[:-1] + edges[1:]) / 2.
    err = np.sqrt(np.abs(counts))
    
    if density:
        if quantile_bin:
            bin_width = None
        else:
            if not isinstance(n_bins, int):
                number_bins = len(n_bins) - 1
            else:
                number_bins = n_bins
            bin_width = get_bin_width(low, high, number_bins)
        counts, err = get_density_counts_err(counts, 
                                             bin_width, err, 
                                             density=density)
        
    if cumulative:
        counts = np.cumsum(counts)
        err = np.cumsum(err)
    
    return counts, edges, centres, err

def get_count_err_ratio(data1, data2, 
                        n_bins, low, high,
                        normalise=True,
                        **kwargs):
    """ Get ``data1[bin i]/data2[bin i]`` histogram,
    with error and bin centres.
    
    Parameters
    ----------
    data1:  array-like
        Data 1
    data2: array-like
        Data 2
    n_bins: int
        number of bins in the histogram
    low: float
        low value of the histogram
    high: float
        high value of the histogram
    normalise: bool
        if True, normalise the counts before
        dividing them
    **kwargs: dict[str:2-list]
        passed to :py:func:`np.histogram`
    
    
    Returns
    -------
    counts : array-like
        counts on ``data1[bin i]/data2[bin i]``
    bin_centres: array-like
        bin centres of the histograms
    err: array-like
        error on ``data1[bin i]/data2[bin i]``
    
    """
    
    # Separate the kwargs into arguments for the first histogram and arguments for the 2nd.
    kwargs1 = {}
    kwargs2 = {}
    for key, kwarg in kwargs.items():
        kwargs1[key] = kwarg[0]
        kwargs2[key] = kwarg[1]
    
    
    counts1, bin_edges = np.histogram(
        data1, 
        n_bins, 
        range=(low, high), 
        **kwargs1
    )
    counts2, _ = np.histogram(
        data2, 
        n_bins, 
        range=(low, high), 
        **kwargs2
    )
    
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
#     # Remove bins with 0 counts or NaN values
#     keep1 = (counts1!=0) & (~np.isnan(counts1))
#     keep2 = (counts2!=0) & (~np.isnan(counts2))
    
#     counts1 = counts1[keep1 & keep2]
#     counts2 = counts2[keep1 & keep2]
    
#     bin_centres = bin_centres[keep1 & keep2]
        
    # Get error

    err1 = np.sqrt(np.abs(counts1))
    err2 = np.sqrt(np.abs(counts2))
    

    # Division
    division , err = divide_counts(counts1, counts2, err1=err1, err2=err2, normalise=normalise)
    return division, bin_centres, err

def divide_counts(counts1, counts2, err1=None, err2=None, normalise=False):
    """ Divide counts1/counts1 and compute the error
    
    Parameters
    ----------
    counts1, counts2: array-like
        number of counts in the bins
    err1, err2 : array-like
        error in the bin counts
    normalise: bool
        if ``True``, the counts are normalised before
        taking their ratio
    
    Returns
    -------
    ratio_counts: array-like
        ``counts1 / counts2``
    ratio_err: 
        Uncertainty in ``ratio_counts``
    
    """
    
    if err1 is None:
        err1 = np.sqrt(counts1)
    if err2 is None:
        err2 = np.sqrt(counts2)
    
    if normalise:
        division = counts1 * counts2.sum() / (counts2 * counts1.sum())
    else:
        division = counts1 / counts2
        
    err = division * np.sqrt((err1 / counts1)**2 + (err2 / counts2)**2)
    return division, err


def get_density_counts_err(counts, bin_width=None, err=None, density=True):
    """ Return the correct normalised array of counts and error on counts,
    by normalising them by (``bin_width * counts.sum()``)
    
    Parameters
    ----------
    counts : np.array
        counts on a histogam
    bin_width: float or None
        bin width of the histogram
        if ``bin_width`` is ``None``, the histogram is not normalised by ``bin_width``
    err : np.array
        count errors
    density: bool
        do we need to normalise
    
    Returns
    -------
    normalised_counts : np.array
        if ``density`` is ``True``, return the normalised ``counts`` array.
    normalised_err    : np.array
        if ``density`` is ``True``, return the normalised ``err`` array.
    """
    
    if bin_width is None:
        bin_width = 1
    
    if density:
        if density=="candidates":
            n_candidates = counts.sum()
            diviser = n_candidates
        elif density=="bin_width":
            diviser = bin_width
        elif density=="both" or density==True:
            n_candidates = counts.sum()
            diviser = n_candidates * bin_width
        
        counts = counts / diviser
        err   = err / diviser if err is not None else None

    if err is None:
        return counts
    else:
        return counts, err

def get_chi2_2samp(data1, data2, n_bins=20, 
                   low=None, high=None, 
                   weights1=None, weights2=None,
                  **kwargs):
    """ Compute the chi2 distance between two histograms
    
    Parameters
    ----------
    data1: array-like
        data n°1
    data2: array-like
        data n°2
    n_bins: int
        number of bins
    low: float
        low value of the range
    high: float
        high of the range
    **kwargs: 
        passed to :py:func:`get_chi2_2counts`
    
    Returns
    -------
    chi2: float
        chi2 between the two sample: 
        :math:`\\chi^2 = \\sum_{\\text{bin i}} \\frac{(\\#1[i] - \\#2[i])^2}{\\#1[i] + \\#2[i]}`
    """
    
    if low is None:
        low = min(min(data1), min(data2))
    if high is None:
        high = max(max(data1), max(data2))
    
    counts1, _, _, err1 = get_count_err(data1, n_bins, low, high, weights=weights1)
    counts2, _, _, err2 = get_count_err(data2, n_bins, low, high, weights=weights2)
    
    return get_chi2_2counts(counts1, counts2, err1=None, err2=None, **kwargs)
    
    

def get_chi2_2counts(counts1, counts2, err1=None, err2=None, normalise=True, div_bins=True):
    """
    Parameters
    ----------
    counts1, counts2: array-like
        number of counts in the bins
    err1, err2 : array-like
        error in the bin counts
    normalise: bool
        if ``True``, the counts are normalised before
        taking their ratio
    
    Returns
    -------
    chi2: float
        chi2 between the two sample: 
        :math:`\\chi^2 = \\sum_{\\text{bin i}} \\frac{(\\#1[i] - \\#2[i])^2}{\\#1[i] + \\#2[i]}`
    """
    
    if err1 is None:
        err1 = np.sqrt(counts1)
    if err2 is None:
        err2 = np.sqrt(counts2)
    
    
    if normalise:
        counts1, err1 = get_density_counts_err(counts1, err=err1, density=True)
        counts2, err2 = get_density_counts_err(counts2, err=err2, density=True)
    
    diff = np.square(counts1 - counts2)
    
    chi2_terms = np.divide(diff, np.square(err1) + np.square(err2),
                     out=np.zeros_like(diff), where=err1+err2!=0)
    
    
    chi2 = chi2_terms.sum()
    
    if div_bins:
        chi2 = chi2 / counts1.size
    
    return chi2
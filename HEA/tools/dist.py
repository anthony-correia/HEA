"""
Tool functions for distributions.
"""

from HEA.tools.string import list_into_string
import numpy as np


def get_chi2(fit_counts, counts):
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
    diff = np.square(fit_counts - counts)

    #n_bins = len(counts)
    diff = np.divide(diff, np.abs(counts),
                     out=np.zeros_like(diff), where=counts != 0)
    chi2 = np.sum(diff)  # sigma_i^2 = mu_i
    return chi2


def get_reduced_chi2(fit_counts, counts, n_dof):
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
    n_bins = np.abs(len(counts))
    return get_chi2(fit_counts, counts) / (n_bins - n_dof)


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


def get_count_err(data, n_bins, low, high, weights=None, **kwargs):
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
    counts, edges = np.histogram(data, range=(
        low, high), bins=n_bins, weights=weights, 
                                 **kwargs)
    centres = (edges[:-1] + edges[1:]) / 2.
    err = np.sqrt(np.abs(counts))

    return counts, edges, centres, err

def get_count_err_ratio(data1, data2, 
                        n_bins, low, high, 
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
    division = counts1 * counts2.sum() / (counts2 * counts1.sum())
    err = division * np.sqrt((err1 / counts1)**2 + (err2 / counts2)**2)
    return division, bin_centres, err


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
    
    n_candidates = counts.sum()
    if density:
        counts = counts / (n_candidates * bin_width)
        err   = err / (n_candidates * bin_width) if err is not None else None
    else:
        counts = counts
        err = err
    
    if err is None:
        return counts
    else:
        return counts, err

def get_chi2_2samp(data1, data2, n_bins=20, low=None, high=None, weights1=None, weights2=None):
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
    
    
    
    counts1, err1 = get_density_counts_err(counts1, err=err1, density=True)
    counts2, err2 = get_density_counts_err(counts2, err=err2, density=True)
    
    diff = np.square(counts1 - counts2)
    
    chi2_terms = np.divide(diff, np.square(err1) + np.square(err2),
                     out=np.zeros_like(diff), where=err1+err2!=0)
    
    
    return chi2_terms.sum() / len(counts1)
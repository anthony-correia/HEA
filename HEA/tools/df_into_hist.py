"""
Functions to turn a dataframe into a histogram
"""

from pandas import DataFrame, Series
from HEA.tools.da import el_to_list
from HEA.tools.assertion import is_list_tuple
from HEA.tools.dist import (
    get_count_err,
    get_bin_width
)

import numpy as np


def _redefine_low_high(low, high, data):
    """ if low or high is not None, return global min (``low``) or max (``high``) of all the data in data, respectively.

    Parameters
    ----------
    low    : float or None
        low value of the range
    high   : float or None
        high value of the range
    data   : pandas.Series or list(pandas.Series)
        for which we want to define the ``low``/``high`` value

    Returns
    -------
    low  : float
        low  if the parameter ``low`` was not None, else minimum of all the data in ``data``
    high : float
        high if the parameter ``high`` was not None, else maximum of all the data in ``data``
    """
    # Transform data into a list of data (if it is not already the case)
    l_data = [data] if isinstance(data, Series) else data

    define_low = low is None
    define_high = high is None

    if define_low or define_high:
        if define_low:
            low = np.inf
        if define_high:
            high = - np.inf
        for el_data in l_data:
            if define_low:
                low = min(low, el_data.min())
            if define_high:
                high = max(high, el_data.max())

    return low, high

def dataframe_into_hist1D(dfs, low, high, n_bins, weights=None,
                         density=None, cumulative=False, 
                          quantile_bin=False, branch=None, **kwargs):
    """ Turn a dataframe into a histogram
    
    Parameters
    ----------
    dfs : dict(str:array-like/pd.DataFrame) or array-like or pd.DataFrame
        Dictionnary {name of the dataframe : associated data}
    branch : str
        name of the branch in the dataframe
    low : float
        low value of the distribution
    high : float
        high value of the distribution
    n_bins : int
        Desired number of bins of the histogram
    weights  : numpy.array
        weights passed to  :py:func:`np.histogram`
    density : bool
        if True, divide the numbers of counts in the histogram by the total number of counts
    cumulative : bool
        if ``True``, return the cumulated event counts.
    quantile_bin : bool
        whether to use equal-yield bins
    branch: str
        branch to plot (used in the case ``dfs`` has DataFrames as values)
    **kwargs :
        passed to :py:func:`HEA.tools.dist.get_count_err`
    
    Returns
    -------
    dfs : dict or tuple
        Associates a name of dataframe its tuple
        ``(counts, err)``
        or directly a the tuple if ``dfs`` was 
        directly the dataframe / array-like
    edges : array-like
        edges of the histogram
    """
    
    weights = el_to_list(weights, len(dfs))
    dfs = dfs.copy()
    dfs_not_dict = not isinstance(dfs, dict)
    if not isinstance(dfs, dict):
        dfs = {"e": dfs}
    for i, (data_name, df) in enumerate(dfs.items()):
        
        if weights is not None:
            if isinstance(weights[i], str):
                weights[i] = df[weights[i]]

            if isinstance(df, DataFrame):
                dfs[data_name] = df[branch]
                
    # First loop to determine the low and high value
    low, high = _redefine_low_high(
        low, high, list(dfs.values()))
    bin_width = get_bin_width(low, high, n_bins)
    
    if density is None:
        if quantile_bin:
            density = False
        elif len(dfs) > 1:
            density = 'candidates'
        else:
            density = "bin_width"
        
    else:
        assert not ((density=="bin_width" or density=="both") and quantile_bin)
    
    dfs_counts = {}
    bins = n_bins
    
    weights = el_to_list(weights, len(dfs))
    for k, (data_name, df) in enumerate(dfs.items()):
        counts, edges, centres, err = get_count_err(
            data=df, n_bins=bins, 
            low=low, high=high, 
            weights=weights[k],
            density=density,
            cumulative=cumulative,
            quantile_bin=quantile_bin,
            **kwargs) 
        
        dfs_counts[data_name] = [counts, err]
        bins = edges
    
    if dfs_not_dict:
        dfs_counts = dfs_counts["e"]
    
    return dfs_counts, edges, density

def dataframe_into_histdD(branches, df, low=None, high=None, n_bins=20, 
                          normalise=False):
    """ Turn a dataframe into a nd histogram
    
    Parameters
    ----------
    df                : pandas.Dataframe or list(array-like)
        Dataframe that contains the 2 branches to plot
        or two same-sized arrays, one for each variable
    branches          : list[str]
        names of the branches
    n_bins            : int or list(int)
        number of bins
    low               : float or list(float)
        low  value(s) of the branches
    high              : float or list(float)
        high value(s) of the branches
    normalise: bool
        Normalised?

    Returns
    -------
    counts: nd array-like
        bin counts
    edges: array-like
        Bin edges
    """

    dim = len(branches) # dimension
    
    # Formatting data input
    list_samples = []
    if isinstance(df, DataFrame):
        for branch in branches:
            list_samples.append(df[branch])
    elif is_list_tuple(df):
        list_samples = df
    
    # low, high and units into a list of size 2
    low = el_to_list(low, dim)
    high = el_to_list(high, dim)

   
    range_list = []
    for i in range(dim):
        low[i], high[i] = _redefine_low_high(
            low[i], high[i], list_samples[i])
        range_list.append((low[i], high[i]))

    print("Number of bins:", n_bins)
    counts, edges = \
        np.histogramdd(np.array(list_samples).T, 
                       range=range_list,
                       bins=n_bins,
                      )
    counts = counts.T 
    
    if normalise:
        counts = counts / counts.sum()

    return counts, edges


def dataframe_into_hist2D(branches, df, low=None, high=None, n_bins=20, 
                          normalise=False):
    """ Turn a dataframe into a 2d histogram
    
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

    Returns
    -------
    counts: 2d array-like
        bin counts
    xedges, yedges: array-like
        Bin edges
    """
    
    # # low, high and units into a list of size 2
    # low = el_to_list(low, 2)
    # high = el_to_list(high, 2)
    
    # # Formatting data input
    # list_samples = []
    # if isinstance(df, DataFrame):
    #     list_samples.append(df[branches[0]])
    #     list_samples.append(df[branches[1]])
    # elif is_list_tuple(df):
    #     list_samples = df
    
    # # low, high and units into a list of size 2
    # low = el_to_list(low, 2)
    # high = el_to_list(high, 2)

    # for i in range(2):
    #     low[i], high[i] = pt._redefine_low_high(
    #         low[i], high[i], list_samples[i])
    
    # counts, xedges, yedges = \
    #     np.histogram2d(list_samples[0], list_samples[1], 
    #                    range=((low[0], high[0]), (low[1], high[1])),
    #                    bins=n_bins,
    #                   )
    # counts = counts.T 
    
    # if normalise:
    #     counts = counts / counts.sum()

    counts, edges = dataframe_into_histdD(branches, df, low=low, high=high, n_bins=n_bins, 
                          normalise=normalise)
    return counts, edges[0], edges[1]

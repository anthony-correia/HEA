"""
Compare original and reweighted MC distribution 
to LHCb distributions.
"""

from HEA.tools import get_chi2_2samp
import pandas as pd

def get_score(MC, data, 
              columns, 
              score_function=get_chi2_2samp,
              MC_weights=None, data_weights=None,
              MC_reweights=None,
              column_ranges=None, 
              **params):
    """ Compare tow datasets in a list of columns, 
    using the ``score_function`` (e.g., )
    
    Parameters
    ----------
    MC : pd.DataFrame
        Simulated data that needs reweighting
    data : pd.DataFrame
        True data
    score_function: python function
        Function used to compute the likelihood score
        between two distributions. 
        For instance:
        
        * :py:func:`HEA.tools.dist.get_chi2_2samp`
        * :py:func:`hep_ml.metrics_utils.ks_2samp_weighted`
    
    MC_weights: array-like
        original MC weigths (e.g., sWeights)
    data_weights: array-like
        Data weigths (e.g., sWeights)
    MC_reweights: array-like
        MC weigths applied to reweight MC
    column_ranges: dict
        Dictionnary whose keys are column names,
        and values are dictionnary with 
        'low' and 'high' keys to specify
        the low and high value of the distribution.
    **params: dict
        passed to ``score_function``
        (e.g., ``n_bins`` for ``get_chi2_2samp``)
    
    Returns
    -------
    result_scores_df: pd.DataFrame
        Dataframe containing the score obtained from
        the comparison to the weighted LHCb data
        of each distribution of:
        
        * The ``'original'`` MC
        * The ``'reweighted_MC'``
        
        Access a score using
        ```result_scores_df[column]['original']`` or
        ```result_scores_df[column]['reweighted_MC']``
    """
    result_scores_dict = {}
    
    for column in columns:
        
        column_scores_dict = {}
        
        if column_ranges is not None:
            if column in column_ranges:
                low, high = column_ranges[column]
            else:
                low = None
                high = None
            
            params['low'] = low
            params['high'] = high
        
        column_scores_dict['original'] = score_function(
            MC[column], data[column],
            weights1=MC_weights, weights2=data_weights,
            **params # n_bins 
        )
        column_scores_dict['reweighted_MC'] = \
            score_function(MC[column], data[column], 
                           weights1=MC_reweights, 
                           weights2=data_weights,
                           **params)
        
        result_scores_dict[column] = pd.Series(column_scores_dict.values(), 
                                               index=column_scores_dict.keys())
    
    result_scores_df = pd.DataFrame(result_scores_dict)
    
    return result_scores_df
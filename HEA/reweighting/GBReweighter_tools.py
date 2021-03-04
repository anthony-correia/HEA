"""
Module with functions that I use with 
GBreweighter.
"""

from hep_ml.reweight import GBReweighter


from HEA.tools.dist import get_chi2_2samp
from scipy.stats import ks_2samp
from itertools import product
import numpy as np



def compute_ks_score(train_dist, test_dist):
    """ Return the KS distance between two distributions.
    To be used with :py:func:`get_best_params`
    
    Parameters
    ----------
    train_dist: array-like
        Training distribution
    test_dist: array-like
        Test distribution
    
    Returns
    -------
    ks_2samp: float
        KS distribution between 
        ``train_dist`` and ``test_dist``
        
    """
    return ks_2samp(train_dist, test_dist).statistic

def compute_pvalue(train_dist, test_dist):
    """ Return the p-value from the KS test between
    two distributions.
    To be used with :py:func:`get_best_params`
    
    Parameters
    ----------
    train_dist: array-like
        Training distribution
    test_dist: array-like
        Test distribution
    
    Returns
    -------
    pvalue: float
        p-value of the KS test between 
        ``train_dist`` and ``test_dist``
    """
    return ks_2samp(train_dist, test_dist).pvalue

def compute_chi2_sum(data1, data2, 
                     weights1=None, weights2=None, 
                     columns=None,
                     n_bins=15):
    """ Return the sum of the chi2 distances between
    two distributions.
    
    data1: pd.DataFrame
        Dataframe 1
    data1: pd.DataFrame
        Dataframe 2
    weights1: array-like
        Weigths for the dataframe 1
    weights2: array-like
        Weigths for the dataframe 2
    columns: 
        Column names where to compute the chi2
    
    Returns:
    chi2_sum: float
        Sum of the chi2 of all the distributions specified
        by columns.
        If ``columns``, 
        take the columns of the first dataframe.
    
    """
    
    
    if columns is None:
        columns = data1.columns
    chi2_sum = 0
    for column in columns:
                
        chi2_sum += get_chi2_2samp(
            data1[column], data2[column], 
            n_bins=n_bins,
            low=None, high=None,
            weights1=weights1,
            weights2=weights2
        )
        
    return chi2_sum

def grid_search(param_grid,
                original_train_df, target_train_df, 
                original_test_df, target_test_df,
                target_train_weight=None, 
                target_test_weight=None,
                original_train_weight=None,
                original_test_weight=None,                
                random_state=20,
                **fit_params):
    """ Train a ``GBReweighter`` in all the hyperparameters
    specified by ``param_grid``. Return the sum of chi2 in
    the columns of the test data as well as the p-values
    obtained from a KS test between the training and test
    MC weights.
    
    Also returns the hyperparameter that leading to the lowest
    chi2 sum.
    
    Parameters
    ----------
    param_grid: dict[str: list]
        Couples (name of the hyperparameter) and 
        (list of values it will take during the grid search)
    original_train_df: pd.DataFrame
        Training sample of the original data
    target_train_df: pd.DataFrame
        Training sample of the target data
    original_test_df: pd.DataFrame
        Test sample of the original data
    target_test_df: pd.DataFrame
         Test sample of the target data
    target_train_weight: array-like
        Weights for the training sample of the target data
    target_test_weight: array-like
        Weights for the test sample of the target data
    original_train_weight: array-like
        Weights for the training sample of the original data
    original_test_weight: array-like
        Weights for the test sample of the original data
    
    **fit_params: dict
        other parameters passed to 
        ``GBReweighter.fit()``
    
    Returns
    -------
    chi2_sum: dict
        Dictionnary that associated 
        to a list of hyperparameter values
        (in the same order as specified in ``param_grid``)
        the chi2 sum in all the columns
    p_values: dict
        Dictionnary that associated 
        to a list of hyperparameter values
        (in the same order as specified in ``param_grid``)
        the p-value obtained when comparing 
        the MC weight distribution obtained 
        from the ``GBReweighter`` in training and test samples.
    best_param_values: dict
        List of hyperparameter values
        that lead to the lowest chi2 sum when comparing
        reweighted original and target in the test samples.
    """
    
    
    best_param_values = {}

    param_values_lists = product(*param_grid.values())
    param_names = param_grid.keys()
    
    best_chi2_sum = np.inf
    
    chi2_sum = {}
    p_values = {}
    
    for param_values_list in param_values_lists:
        
        ## Unwrap the hyperparameters and train
        hyperparams = dict(zip(param_names, param_values_list))
        print("test of ")
        print(hyperparams)
        
        reweighter = GBReweighter(gb_args={'random_state': random_state}, 
                                  **hyperparams)
        
        reweighter = reweighter.fit(
            original=original_train_df,
            target=target_train_df,
            original_weight=original_train_weight,
            target_weight=target_train_weight,
            **fit_params
        )
        
        MC_test_weight = reweighter.predict_weights(
            original_test_df,
            original_weight=original_test_weight
        )
        
        MC_train_weight = reweighter.predict_weights(
            original_train_df,
            original_weight=original_train_weight
        )
        
        chi2_sum[tuple(param_values_list)] = compute_chi2_sum(
            data1=target_test_df,
            data2=original_test_df,
            weights1=target_test_weight,
            weights2=MC_test_weight,
        )
        
        p_values[tuple(param_values_list)] = compute_pvalue(
            MC_train_weight, 
            MC_test_weight
        )
        
        
        if chi2_sum[tuple(param_values_list)] < best_chi2_sum:
            best_chi2_sum = chi2_sum[tuple(param_values_list)]
            best_param_values = tuple(param_values_list)
        
    return chi2_sum, p_values, best_param_values  


def show_sorted_combined_scores(chi2_sum1, chi2_sum2,
                                p_values1, p_values2):
    """ Print the hyperparameter values, the p_values and combined chi2
    sorted by lower combined chi2, for two different datasets.
    
    Parameters
    ----------
    chi_sum1, chi2_sum2: dict
        Dictionnaries that associated 
        to a list of hyperparameter values
        (in the same order as specified in ``param_grid``)
        the chi2 sum in all the columns
    p_values1, p_values2: dict
        Dictionnary that associated 
        to a list of hyperparameter values
        (in the same order as specified in ``param_grid``)
        the p-value obtained when comparing 
        the MC weight distribution obtained 
        from the ``GBReweighter`` in training and test samples.
    """
    combined_chi2_sum = []
    two_p_values = []
    parameter_values_list = list(chi2_sum1.keys())

    ## Combine the scores
    for parameter_values in parameter_values_list:
        combined_chi2_sum.append((chi2_sum1[parameter_values] + chi2_sum1[parameter_values]) / 2)
        two_p_values.append((p_values1[parameter_values], p_values2[parameter_values]))

    ## Sort by chi2_sum the three lists
    index_sorting = np.argsort(combined_chi2_sum)
    sorted_parameter_values_list = np.array(parameter_values_list)[index_sorting]
    sorted_two_p_values = np.array(two_p_values)[index_sorting]
    sorted_combined_chi2_sum = np.array(combined_chi2_sum)[index_sorting]
    
    ## Print
    for sorted_parameter_values, two_p_values, combine_chi2_sum\
    in zip(sorted_parameter_values_list, sorted_two_p_values, sorted_combined_chi2_sum):
        print(sorted_parameter_values, two_p_values, combine_chi2_sum)
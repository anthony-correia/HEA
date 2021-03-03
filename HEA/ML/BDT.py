"""
Module for BDT:

* Prepare the signal and background sample (merge them, create a ``'y'`` variable for learning, ...)
* Train the BDT with the specified classifier (adaboost or gradientboosting)
* Apply the BDT to the data and save the result
"""


from HEA.config import loc

import HEA.plot.tools as pt
from HEA.tools.dir import create_directory
from HEA.tools import string

from HEA.tools.da import add_in_dic, show_dictionnary
from HEA.tools.serial import dump_pickle
from HEA.pandas_root import save_root

from HEA.definition import RVariable



import pickle
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import pandas.core.common as com
from pandas import Index

# sklearn
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.model_selection import train_test_split

from scipy.stats import ks_2samp


# Parameters of the plot
from matplotlib import rc, rcParams, use
rc('font', **{'family': 'serif', 'serif': ['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False
# use('Agg') #no plot.show() --> no display needed


##########################################################################
######################################## BDT training ####################
##########################################################################

# DATA PROCESSING ------------------------------------------------------


def concatenate(dfa_tot_sig, dfa_tot_bkg):
    """ Concatenate the signal and background dataframes
    and create a new variable ``y``,
    which is 1 if the candidate is signal,
    or 0 if it is background.

    Parameters
    ----------
    dfa_tot_sig : pandas.Dataframe
        Signal dataframe
    dfa_tot_bkg : pandas.Dataframe
        Background dataframe
    Returns
    -------
    X  : numpy.ndarray
        Array with signal and MC data concatenated
    y  : numpy.array
        new variable: array with 1 for the signal events, and 0 for background events
    df : pandas.Dataframe
        signal and background dataframes concatenated with with the new column ``'y'``
    """
    assert len(dfa_tot_sig.columns) == len(dfa_tot_bkg.columns)
    assert (dfa_tot_sig.columns == dfa_tot_bkg.columns).all()

    # Concatenated data
    X = np.concatenate((dfa_tot_sig, dfa_tot_bkg))
    # List of 1 for signal and 0 for background (mask)
    y = np.concatenate((np.ones(dfa_tot_sig.shape[0]),
                        np.zeros(dfa_tot_bkg.shape[0])))

    # Concatened dataframe of signal + background, with a new variable y:
    # y = 0 if background
    # y = 1 if signal
    df = pd.DataFrame(np.hstack((X, y.reshape(y.shape[0], -1))),
                      columns=list(dfa_tot_sig.columns) + ['y'])
    return X, y, df


def bg_sig(y):
    """Return the mask to get the background and the signal (in this order)

    Parameters
    ----------
    y  : numpy.array
        array with 1 for the signal events, and 0 for background events

    Returns
    -------
    signal: numpy.array
        array with ``True`` if signal event, else ``False``
    background: numpy.array
        array with ``True`` if background event, else ``False``
    """
    return (y < 0.5), (y > 0.5)


def get_train_test(X, y, test_size=0.5, random_state=15):
    """ Get the train and test arrays

    Parameters
    ----------
    X  : numpy.ndarray
        Array with signal and MC data concatenated
    y  : numpy.array
        Array with 1 for the signal events, and 0 for background events
    test_size: float between 0 and 1
        size of the test sample relatively to the full datasample
    random_state: float
        random state

    Returns
    -------
    X_train : numpy.ndarray
        Array with signal and MC data concatenated and shuffled for training
    X_text  : numpy.ndarray
        Array with signal and MC data concatenated and shuffled for test
    y_train : numpy.array
        Array with 1 for the signal events, and 0 for background events (shuffled) for training
    y_test  : numpy.array
        Array with 1 for the signal events, and 0 for background events (shuffled) for test
    """
    # Separate train/test data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test


def get_train_test_df(df, test_size=0.5, random_state=15):
    """ Get the train and test pandas dataframes

    Parameters
    ----------
    df : pandas.Dataframe
        dataframe
    test_size: float between 0 and 1
        size of the test sample relatively to the full datasample
    random_state: float
        random state

    Returns
    -------
    df_train : pandas.Dataframe
        dataframe using for training
    df_test : pandas.Dataframe
        dataframe using for test
    """
    # Separate train/test data
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state)
    return df_train, df_test


def BDT(X_train, y_train, classifier='adaboost', **hyperparams):
    """ Train the BDT and return the result

    Parameters
    ----------
    X               : numpy ndarray
        array with signal and background concatenated,
        The columns of X correspond to the variable the BDT will be trained with
    y               : numpy array
        array with 1 if the concatened event is signal, 0 if it is background
    classifier      : str
        Used classifier

        * ``'adaboost'``
        * ``'gradientboosting'``
        * ``'xgboost'`` (experimental)
    hyperparameters : dict
        used hyperparameters.
        Default:

        * ``n_estimators = 800``
        * ``learning_rate = 0.1``

    Returns
    -------
    xgb.XGBClassifier
        trained XGboost classifier, if ``classifier == 'xgboost'``
    sklearn.ensemble.AdaBoostClassifier
        trained adaboost classifier, if ``classifier == 'adaboost'``
    sklearn.ensemble.GradientBoostingClassifier
        trained gradient boosting classifier, if ``classifier == 'gradientboosting'``
    """

    weights = compute_sample_weight(class_weight='balanced', y=y_train)

    if hyperparams is None:
        hyperparams = {}

    add_in_dic('n_estimators', hyperparams, 800)
    # Learning rate shrinks the contribution of each tree by alpha
    add_in_dic('learning_rate', hyperparams, 0.1)
    show_dictionnary(hyperparams, "hyperparameters")

    # Define the BDT
    if classifier == 'adaboost':
        dt = DecisionTreeClassifier(max_depth=3, min_samples_leaf=0.05)
        # The minimum number of samples required to be at a leaf node
        # here, since it's a float, it is expressed in fraction of len(X_train)
        # We need min_samples_leaf samples before deciding to create a new leaf
        bdt = AdaBoostClassifier(
            dt, algorithm='SAMME', verbose=1, **hyperparams)

    elif classifier == 'gradientboosting':
        bdt = GradientBoostingClassifier(
            max_depth=1, min_samples_split=2, verbose=1, random_state=15, **hyperparams)

    elif classifier == 'xgboost':  # experimental
        import xgboost as xgb
        bdt = xgb.XGBClassifier(
            objective="binary:logistic", random_state=15, verbose=1, learning_rate=0.1)

    ## Learning (fit)
    bdt.fit(X_train, y_train, sample_weight=weights)

    return bdt

##########################################################################
################################### Analysis BDT training ################
##########################################################################





def apply_BDT(df_tot, df_train, bdt, BDT_name=None,
              save_BDT=False, kind_data='common'):
    """
    * Apply the BDT to the dataframe ``df_train`` which contains only the training variable.
    * Add the BDT output as a new variable in ``df_tot``.
    * Save ``df_tot`` in a root file ``{loc['root']}/{kind_data}_{ BDT_name}.root`` (branch ``'DecayTree'``)
    * In addition,  save the BDT output in a separated root file ``{loc['root']t/BDT_{BDT_name}.root`` (branch ``'BDT'``)
    * if ``save_BDT`` is ``True``, save the BDT in a root file ``{loc['pickle']}/bdt_{BDT_name}.pickle``

    Parameters
    ----------
    df_tot        : pandas.Dataframe
        dataframe that will be saved together with the BDT output
    df_train      : pandas.Dataframe
        dataframe with only the variables that have been used for the training
    bdt           : sklearn.ensemble.AdaBoostClassifier or sklearn.ensemble.GradientBoostingClassifier
        trained BDT classifier
    BDT_name      : str
        name of the BDT, used for the name of the saved files
    save_BDT      : bool
        if ``True``, save the BDT in a pickle file
    kind_data     : str
        name of the data where the BDT is applied to (e.g., ``'MC'``, ``'common'``, ...)
    """

    # Apply the BDT to the dataframe that contains only the variables used in
    # the training, in the right order
    df_tot['BDT'] = bdt.decision_function(df_train)

    file_name = string.add_text(kind_data, BDT_name, '_')

    df = pd.DataFrame()
    df['BDT'] = df_tot['BDT']

    save_root(df, 'BDT_' + file_name, 'DecayTree')
    save_root(df_tot, file_name, 'DecayTree')

    if save_BDT:
        dump_pickle(bdt, string.add_text('bdt', file_name, '_'))

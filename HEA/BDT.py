"""
Module for BDT:

* Plot the distributions of the chosen variables in signal and background (super-imposed)
* Prepare the signal and background sample (merge them, create a ``'y'`` variable for learning, ...)
* Train the BDT with the specified classifier (adaboost or gradientboosting)
* Plot the result of the tests (ROC curve, overtraining check with KS test)
* Apply the BDT to the data and save the result
"""


from HEA.config import loc

import HEA.plot.tools as pt
import HEA.plot.histogram as h
from HEA.tools.dir import create_directory
from HEA.tools import string

from HEA.tools.da import add_in_dic, show_dictionnary
from HEA.tools.serial import dump_pickle
from HEA.pandas_root import save_root

from HEA.definition import RVariable


import os.path as op
from os import makedirs

import pickle
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd
import pandas.core.common as com
from pandas.plotting import scatter_matrix
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
###################################### Plotting function #################
##########################################################################


# it's an adaptation of hist_frame of  pandas.plotting._core
def signal_background(data1, data2, column=None, range_column=None, grid=True,
                      xlabelsize=None, ylabelsize=None,
                      sharex=False,
                      sharey=False, figsize=None,
                      layout=None, n_bins=40, fig_name=None,
                      folder_name=None, colors=['red', 'green'], **kwds):
    """Draw histogram of the DataFrame's series comparing the distribution
    in ``data1`` to ``data2`` and save the result in
    ``{loc['plot']}/BDT/{folder_name}/1D_hist_{fig_name}``

    Parameters
    ----------
    data1        : pandas.Dataframe
        First dataset
    data2        : pandas.Dataframe
        Second dataset
    column       : str or list(str)
        If passed, will be used to limit data to a subset of columns
    grid         : bool
        Whether to show axis grid lines
    xlabelsize   : int
        if specified changes the x-axis label size
    ylabelsize   : int
        if specified changes the y-axis label size
    ax           : matplotlib.axes.Axes
    sharex       : bool
        if ``True``, the X axis will be shared amongst all subplots.
    sharey       : bool
        if ``True``, the Y axis will be shared amongst all subplots.
    figsize      : tuple
        the size of the figure to create in inches by default
    bins         : int,
        number of histogram bins to be used
    fig_name    : str
        name of the saved file
    folder_name  : str
        name of the folder where to save the plot
    colors       : [str, str]
        colors used for the two datasets
    **kwds       : dict
        other plotting keyword arguments, to be passed to the `ax.hist()` function

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot
    ax : matplotlib.figure.Axes
        Axis of the plot
    """
    if 'alpha' not in kwds:
        kwds['alpha'] = 0.5

    if column is not None:
        # column is not a list, convert it into a list.
        if not isinstance(column, (list, np.ndarray, Index)):
            column = [column]
        data1 = data1[column]
        data2 = data2[column]

    data1 = data1._get_numeric_data()  # select only numbers
    data2 = data2._get_numeric_data()  # seject only numbers
    naxes = len(data1.columns)  # number of axes = number of available columns

    max_nrows = 4
    # subplots
    fig, axes = plt.subplots(nrows=min(naxes, max_nrows), ncols=1 + naxes // max_nrows, squeeze=False,
                             sharex=sharex,
                             sharey=sharey,
                             figsize=figsize)

    _axes = axes.flat

    if range_column is None:
        range_column = [[None, None] for i in range(len(column))]
    # data.columns = the column labels of the DataFrame.
    for i, col in enumerate(data1.columns):
        # col = name of the column/variable
        ax = _axes[i]

        if range_column[i] is None:
            range_column[i] = [None, None]
        if range_column[i][0] is None:
            low = min(data1[col].min(), data2[col].min())
        else:
            low = range_column[i][0]
        if range_column[i][1] is None:
            high = max(data1[col].max(), data2[col].max())
        else:
            high = range_column[i][1]

        low, high = pt.redefine_low_high(
            range_column[i][0], range_column[i][1], [data1[col], data2[col]])
        _, _, _, _ = h.plot_hist_alone(ax, data1[col].dropna().values, n_bins, low, high, colors[1], mode_hist=True, alpha=0.5,
                                       density=True, label='background', label_ncounts=True)
        _, _, _, _ = h.plot_hist_alone(ax, data2[col].dropna().values, n_bins, low, high, colors[0], mode_hist=True, alpha=0.5,
                                       density=True, label='signal', label_ncounts=True)

        bin_width = (high - low) / n_bins
        latex_branch, unit = RVariable.get_latex_branch_unit_from_branch(col)
        h.set_label_hist(ax, latex_branch, unit,
                         bin_width=bin_width, density=False, fontsize=20)
        pt.fix_plot(ax, factor_ymax=1 + 0.3, show_leg=True,
                    fontsize_ticks=15., fontsize_leg=20.)
        pt.show_grid(ax, which='major')

    i += 1
    while i < len(_axes):
        ax = _axes[i]
        ax.axis('off')
        i += 1

    #fig.subplots_adjust(wspace=0.3, hspace=0.7)
    if fig_name is None:
        fig_name = string.list_into_string(column)

    plt.tight_layout()
    pt.save_fig(fig, f"1D_hist_{fig_name}", folder_name=f'BDT/{folder_name}')

    return fig, axes


def correlations(data, fig_name=None, folder_name=None, title=None, **kwds):
    """ Calculate pairwise correlation between features of the dataframe data
    and save the figure in ``{loc['plot']}/BDT/{folder_name}/corr_matrix_{fig_name}``

    Parameters
    ----------
    data         : pandas.Dataframe
        dataset
    fig_name     : str
        name of the saved file
    folder_name  : str
        name of the folder where to save the plot
    **kwds       : dict
        other plotting keyword arguments, to be passed to ``pandas.DataFrame.corr()``

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot
    ax : matplotlib.figure.Axes
        Axis of the plot
    """

    # simply call df.corr() to get a table of
    # correlation values if you do not need
    # the fancy plotting
    corrmat = data.corr(**kwds)  # correlation

    fig, ax1 = plt.subplots(ncols=1, figsize=(12, 10))  # 1 plot

    opts = {'cmap': plt.get_cmap("RdBu"),  # red blue color mode
            'vmin': -1, 'vmax': +1}  # correlation between -1 and 1
    heatmap1 = ax1.pcolor(corrmat, **opts)  # create a pseudo color plot
    plt.colorbar(heatmap1, ax=ax1)  # color bar

    title = string.add_text("Correlations", title, ' - ')
    ax1.set_title(title)

    labels = list(corrmat.columns.values)  # get the list of labels
    for i, label in enumerate(labels):
        latex_branch, _ = RVariable.get_latex_branch_unit_from_branch(label)
        labels[i] = latex_branch
    # shift location of ticks to center of the bins
    ax1.set_xticks(np.arange(len(labels)) + 0.5, minor=False)
    ax1.set_yticks(np.arange(len(labels)) + 0.5, minor=False)
    ax1.set_xticklabels(labels, minor=False, ha='right', rotation=70)
    ax1.set_yticklabels(labels, minor=False)

    plt.tight_layout()

    if fig_name is None:
        fig_name = string.list_into_string(column)

    pt.save_fig(fig, f"corr_matrix_{fig_name}",
                folder_name=f'BDT/{folder_name}')

    return fig, ax1


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


def classification_report_print(X_test, y_test, bdt, BDT_name=None):
    """ Test the bdt training with the testing sample.\
    Print and save the report in ``{loc['tables']}/BDT/{BDT_name}/classification_report.txt``.

    Parameters
    ----------
    X_text    : numpy.ndarray
        Array with signal and MC data concatenated and shuffled for test
    y_test    : numpy.array
        Array with 1 for the signal events, and 0 for background events (shuffled) for test
    bdt       : sklearn.ensemble.AdaBoostClassifier or sklearn.ensemble.GradientBoostingClassifier
        trained classifier
    BDT_name      : str
        name of the BDT, used for the path of the saved txt file.
    """
#     if xgboost:
#         y_predicted = xgbmodel.predict_proba(X)[:,1]
#     else:
    y_predicted = bdt.predict(X_test)

    classification_report_str = classification_report(y_test, y_predicted,
                                                      target_names=["background", "signal"])

    print(classification_report_str)
    ROC_AUC_score = roc_auc_score(y_test,  # real
                                  bdt.decision_function(X_test))
    # bdt.decision_function(X_test) = scores = returns a Numpy array, in which each element
    # represents whether a predicted sample for x_test by the classifier lies to the right
    # or left side of the Hyperplane and also how far from the HyperPlane.

    print("Area under ROC curve: %.4f" % (ROC_AUC_score))

    # Write the results -----
    fig_name = string.add_text('classification_report', BDT_name, '_')

    path = create_directory(f"{loc['tables']}/BDT/", BDT_name)
    with open(f"{path}/{fig_name}.txt", 'w') as f:
        f.write(classification_report_str)
        f.write("Area under ROC curve: %.4f" % (ROC_AUC_score))


def plot_roc(X_test, y_test, bdt, BDT_name=None):
    """ Plot and save the roc curve in ``{loc['plots']}/BDT/{BDT_name}/ROC.pdf``

    Parameters
    ----------
    X_test        : numpy.ndarray
        signal and background concatenated, testing sample
    y_test        : numpy.array
        signal and background concatenated, testing sample,
        0 if the events is background, 1 if it is signal
    bdt           : sklearn.ensemble.AdaBoostClassifier or sklearn.ensemble.GradientBoostingClassifier
        trained BDT
    BDT_name      : str
        name of the BDT, used for the name of the saved plot
    folder_name   : str
        name of the folder where to save the BDT

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot
    ax : matplotlib.figure.Axes
        Axis of the plot
    """

    # Get the results -----
    # result of the BDT of the test sample
    decisions = bdt.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, decisions)  # roc_curve
    # y_test: true results
    # decisions: result found by the BDT
    # fpr: Increasing false positive rates such that element i is the false positive rate of predictions with score >= thresholds[i].
    # tpr: Increasing true positive rates such that element i is the true positive rate of predictions with score >= thresholds[i].
    # thresholds: Decreasing thresholds on the decision function used to
    # compute fpr and tpr. thresholds[0] represents no instances being
    # predicted and is arbitrarily set to max(y_score) + 1
    fig, ax = plt.subplots(figsize=(8, 6))
    roc_auc = auc(fpr, tpr)

    # Plot the results -----
    ax.plot(fpr, tpr, lw=1, label='ROC (area = %0.2f)' % (roc_auc))
    ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Luck')
    ax.set_xlim([-0.05, 1.05])
    ax.set_ylim([-0.05, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=25)
    ax.set_ylabel('True Positive Rate', fontsize=25)
    title = 'Receiver operating characteristic'

    ax.legend(loc="lower right", fontsize=20.)
    pt.show_grid(ax)
    pt.fix_plot(ax, factor_ymax=1.1, show_leg=False,
                fontsize_ticks=20., ymin_to_0=False)
    # Save the results -----

    pt.save_fig(fig, "ROC", folder_name=f'BDT/{BDT_name}')

    return fig, ax


def compare_train_test(bdt, X_train, y_train, X_test, y_test, bins=30, BDT_name="",
                       colors=['red', 'green']):
    """ Plot and save the overtraining plot in ``{loc['plots']}/BDT/{folder_name}/overtraining_{BDT_name}.pdf``

    Parameters
    ----------
    bdt           : sklearn.ensemble.AdaBoostClassifier or sklearn.ensemble.GradientBoostingClassifier
        trained BDT classifier
    X_train : numpy.ndarray
        Array with signal and MC data concatenated and shuffled for training
    y_train : numpy.array
        Array with 1 for the signal events, and 0 for background events (shuffled) for training
    X_text  : numpy.ndarray
        Array with signal and MC data concatenated and shuffled for test
    y_test  : numpy.array
        Array with 1 for the signal events, and 0 for background events (shuffled) for test
    bins          : int
        number of bins of the plotted histograms
    BDT_name      : str
        name of the BDT, used for the folder where the figure is saved

    Returns
    -------
    fig              : matplotlib.figure.Figure
        Figure of the plot
    ax               : matplotlib.figure.Axes
        Axis of the plot
    s_2samp_sig      : float
        Kolmogorov-Smirnov statistic for the signal distributions
    ks_2samp_bkg     : float
        Kolmogorov-Smirnov statistic for the background distributions
    pvalue_2samp_sig : float
        p-value of the Kolmogorov-Smirnov test for the signal distributions
    pvalue_2samp_bkg : float
        p-value of the Kolmogorov-Smirnov test for the background distributions
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    ## decisions = [d(X_train_signal), d(X_train_background),d(X_test_signal), d(X_test_background)]
    decisions = []
    for X, y in ((X_train, y_train), (X_test, y_test)):
        d1 = bdt.decision_function(X[y > 0.5]).ravel()
        d2 = bdt.decision_function(X[y < 0.5]).ravel()
        decisions += [d1, d2]  # [signal, background]

    '''
    decisions[0]: train, background
    decisions[1]: train, signal
    decisions[2]: test, background
    decisions[3]: test, signal
    '''

    # Range of the full plot
    low = min(np.min(d) for d in decisions)
    high = max(np.max(d) for d in decisions)
    low_high = (low, high)

    # Plot for the train data the stepfilled histogram of background (y<0.5)
    # and signal (y>0.5)
    ax.hist(decisions[0],
            color=colors[0], alpha=0.5, range=low_high, bins=bins,
            histtype='stepfilled', density=True,
            label='S (train)')
    ax.hist(decisions[1],
            color=colors[1], alpha=0.5, range=low_high, bins=bins,
            histtype='stepfilled', density=True,
            label='B (train)')

    # Plot for the test data the points with uncertainty of background (y<0.5)
    # and signal (y>0.5)
    hist, bins = np.histogram(decisions[2],
                              bins=bins, range=low_high, density=True)
    scale = len(decisions[2]) / sum(hist)
    # Compute and rescale the error
    err = np.sqrt(hist * scale) / scale

    width = (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    ax.errorbar(center, hist, yerr=err, fmt='o', c=colors[0], label='S (test)')

    hist, bins = np.histogram(decisions[3],
                              bins=bins, range=low_high, density=True)
    # Compute and rescale the error
    scale = len(decisions[2]) / sum(hist)
    err = np.sqrt(hist * scale) / scale

    ax.errorbar(center, hist, yerr=err, fmt='o', c=colors[1], label='B (test)')

    ax.set_xlabel("BDT output", fontsize=25.)
    ax.set_ylabel("Arbitrary units", fontsize=25.)
    ax.legend(loc='best', fontsize=20.)
    pt.show_grid(ax)

    pt.fix_plot(ax, factor_ymax=1.1, show_leg=False,
                fontsize_ticks=20., ymin_to_0=False)

    pt.save_fig(fig, "overtraining", folder_name=f'BDT/{BDT_name}')

    ks_2samp_sig = ks_2samp(decisions[0], decisions[2]).statistic
    ks_2samp_bkg = ks_2samp(decisions[1], decisions[3]).statistic
    pvalue_2samp_sig = ks_2samp(decisions[0], decisions[2]).pvalue
    pvalue_2samp_bkg = ks_2samp(decisions[1], decisions[3]).pvalue
    print('Kolmogorov-Smirnov statistic')
    print(f"signal    : {ks_2samp_sig}")
    print(f"Background: {ks_2samp_bkg}")

    print('p-value')
    print(f"signal    : {pvalue_2samp_sig}")
    print(f"Background: {pvalue_2samp_bkg}")
    return fig, ax, ks_2samp_sig, ks_2samp_bkg, pvalue_2samp_sig, pvalue_2samp_bkg


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

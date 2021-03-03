"""
Check that a classifier training went well
"""


import HEA.plot.tools as pt
from HEA.tools.dir import create_directory
from HEA.tools import string
import matplotlib.pyplot as plt
import numpy as np


# sklearn
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc

from scipy.stats import ks_2samp


# Parameters of the plot
from matplotlib import rc, rcParams, use
rc('font', **{'family': 'serif', 'serif': ['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False
# use('Agg') #no plot.show() --> no display needed


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
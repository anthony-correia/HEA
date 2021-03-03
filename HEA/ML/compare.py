"""
Compare signal and MC distributions.
"""


import HEA.plot.tools as pt
import HEA.plot.histogram as h
from HEA.tools import string
from HEA.definition import RVariable


import matplotlib.pyplot as plt
import numpy as np
from pandas import Index

# Parameters of the plot
from matplotlib import rc, rcParams, use
rc('font', **{'family': 'serif', 'serif': ['Roman']})
rc('text', usetex=True)
rcParams['axes.unicode_minus'] = False




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

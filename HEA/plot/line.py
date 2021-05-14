"""
Plot lines:

* plot y vs x
* plot y1, y2, y3, ... vs x (several curves with the same x)
"""


from HEA.tools.string import list_into_string, _remove_latex
import HEA.plot.tools as pt
from HEA.config import default_fontsize
from HEA.tools.da import el_to_list, flatten_2Dlist
from uncertainties import unumpy
import numpy as np
import matplotlib.pyplot as plt


##########################################################################
######################################## Tool function ###################
##########################################################################

def _el_or_list_to_2D_list(l, type_el=np.ndarray):
    """ Transform an element or a list into a 2D list.
    Parameters
    ----------
    l       : type_el, or list(type_el) or list(list(type_el))
        list to transform
    type_el : type
        e.g., 'numpy.ndarray', ...
    Returns
    -------
    list(list(type_el))
        2D list
    """
    if isinstance(l, type_el):
        list_2D = [[l]]
    elif isinstance(l[0], type_el):
        list_2D = [l]
    else:
        list_2D = l

    return list_2D


def add_value_labels(ax, lx, ly, labels, space_x=-10, space_y=5, labelsize=12):
    """ Annotate points with a text whose distance to the point is specified by (``space_x``, ``space_y``)

    Parameters
    ----------
    ax          : matplotlib.axes.Axes
        axis where to plot
    lx          : list(float)
        abcissa of the points
    ly          : list(float)
        ordinate of the points
    labels      : list(str)
        annotation of the specified points
    space_x     : float
        space in pixel from the point to the annotation text, projected in the x-axis
    space_y     : float
        space in pixel from the point to the annotation text, projected in the y-axis
    labelsize   : float
        fontsize of the annotation
    """
    assert len(lx) == len(ly)
    assert len(labels) == len(lx)

    # For each bar: Place a label
    for x, y, label in zip(lx, ly, labels):
        # Vertical alignment for positive values
        ha = 'center'
        va = 'center'
        if x != 0 and y != 0:
            ax.annotate(
                label,
                (x, y),  # xycoords='data',
                # Vertically shift label by ``space``
                xytext=(space_x, space_y),
                textcoords='offset pixels',  # Interpret ``xytext`` as offset in points
                va=va, ha=ha,
                size=labelsize)

##########################################################################
###################################### Plotting function #################
##########################################################################


def plot_line_alone(ax, x, ly, xlabel, labels=None,
                    colors=['b', 'g', 'r', 'y'],
                    fontsize=default_fontsize['label'],
                    markersize=1,
                    linewidth=1., linestyle='-', factor_ymax=1., marker='.',
                    elinewidth=None,
                    annotations=None,
                    fontsize_annot=default_fontsize['annotation'],
                    space_x=-15, space_y=5,
                    fontsize_leg=default_fontsize['legend'],
                    pos_text_LHC=None, 
                    ylabel='Value',
                    **kwgs):
    """ Plot the curve(s) in `ly` as a function of `x`, with `annotations`.

    Parameters
    ----------
    ax               : matplotlib.axes.Axes
        axis where to plot
    x                : list(float)
        abcissa of the points
    ly               : list(list(float))
        list of the ordinates of the points of the curves
    labels           : list(str)
        labels of the curves
    xlabel           : str
        label of the x-axis
    colors           : list(str)
        colors of each curve
    fontsize         : float
        fontsize of the labels
    markersize       : float
        size of the markers
    linewidth        : float
        linewidth of the plotted curves
    linestyle        : str
        linestyle of the plotted curves
    factor_ymax      : float
        multiplicative factor of ymax
    marker           : str
        marker style
    elinewidth       : float
        width of the error bars
    annotations      : list(str)
        list of the labels of the points - only if there is one curve (i.e., ``len(ly)==1``)
    fontsize_annot   : float
        fontsize of the annotations
    space_x          : float
        space in pixel from the point to the annotation text, projected in the x-axis
    space_y          : float
        space in pixel from the point to the annotation text, projected in the y-axis
    fontsize_leg : float
        fontsize of the legend    
    pos_text_LHC    : dict, list or str
        passed to :py:func:`HEA.plot.tools.set_text_LHCb` as the ``pos`` argument.
    ylabel: str
        ylabel if there are several curves.
    **kwgs: dict
        passed to ``ax.plot()`` or ``ax.errorbar``
    """
    
    
    colors = el_to_list(colors, len(ly))

    show_leg = False
    
    for i, y in enumerate(ly):
        label = labels[i] if len(ly) > 1 else None
        x = np.array(x)
        y = np.array(y)
        x_n = unumpy.nominal_values(x)
        y_n = unumpy.nominal_values(y)
        if (unumpy.std_devs(x)==0).all() and (unumpy.std_devs(y)==0).all():
            ax.plot(x_n, y_n, linestyle=linestyle, color=colors[i],
                    markersize=markersize, linewidth=linewidth,
                    label=label, marker=marker, **kwgs
                   )
        else:
            ax.errorbar(x_n, y_n,
                        xerr=unumpy.std_devs(x), yerr=unumpy.std_devs(y),
                        linestyle=linestyle, color=colors[i],
                        markersize=markersize, elinewidth=elinewidth,
                        linewidth=linewidth, label=label, marker=marker, 
                        **kwgs)

        if label is not None:
            show_leg = True

    ax.set_xlabel(xlabel, fontsize=fontsize)

    if len(ly) == 1:
        ax.set_ylabel(labels[0], fontsize=fontsize)
    else:
        ax.set_ylabel(ylabel, fontsize=fontsize)

    # Grid
    pt.show_grid(ax, which='major')
    pt.show_grid(ax, which='minor')

    # Ticks
    pt.fix_plot(ax, factor_ymax=factor_ymax, show_leg=False,
                ymin_to_0=False, pos_text_LHC=pos_text_LHC)

    if show_leg:
        ax.legend(fontsize=fontsize_leg, markerscale=12.)
        
    if annotations is not None:
        assert len(ly) == 1
        add_value_labels(ax, x_n, y_n, annotations,
                         labelsize=fontsize_annot, space_x=space_x, space_y=space_y)


def plot_lines(x, y, name_x, names_y, latex_name_x=None, latex_names_y=None,
               fig_name=None, folder_name=None,
               log_scale=None,
               save_fig=True,
               **kwgs):
    """ plot y or a list of y as a function of x. If they are different curves, their points should have the same abscissa.

    Parameters
    ----------
    x                : list(float)
        abcissa of the points
    y                : numpy.array(uncertainties.ufloat) or numpy.array(float) or list(numpy.array(uncertainties.ufloat)) or list(numpy.array(float)) or
        (list instead of numpy.array might work)
        ordinate of the points of the curve(s)
    name_x           : str
        name of the x variable, used for then name of the saved figure
    name_y           : str or list(str)
        name of each list in ``l_y``, used for then name of the saved figure
    latex_name_x : str
        latex name of the x variable, used to label the x-axis
    latex_names_y    : str or list(str)
        surname of each list in ``l_y`` - used for labelling each curve
    fig_name         : str
        name of the file to save
    folder_name      : str
        name of the folder where the image is saved
    factor_ymax      : float
        ymax is multiplied by factor_ymax
    log_scale        : 'both', 'x' ot 'y'
        specifies which axis will be set in log scale
    **kwgs           : dict
        passed to :py:func:`plot_line_alone`

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot
    ax : matplotlib.figure.Axes or list(matplotlib.figure.Axes)
        Axis of the plot or list of axes of the plot
    """

    if latex_name_x is None:
        latex_name_x = _remove_latex(name_x)

    groups_ly = _el_or_list_to_2D_list(y)
    groups_names_y = _el_or_list_to_2D_list(names_y, str)

    if latex_names_y is not None:
        groups_latex_names_y = _el_or_list_to_2D_list(latex_names_y, str)
    else:
        groups_latex_names_y = groups_names_y

    fig, axs = plt.subplots(len(groups_ly), 1, figsize=(8, 6 * len(groups_ly)))

    for k, ly in enumerate(groups_ly):
        if len(groups_ly) == 1:
            ax = axs
        else:
            ax = axs[k]

        # In the same groups_ly, we plot the curves in the same plot
        plot_line_alone(ax, x, ly, xlabel=latex_name_x, labels=groups_latex_names_y[k],
                        **kwgs)

        pt.set_log_scale(ax, axis=log_scale)

    plt.tight_layout()

    if save_fig:
        pt.save_fig(fig, fig_name, folder_name,
                    f'{name_x}_vs_{list_into_string(flatten_2Dlist(names_y))}')

    return fig, axs


def plot_lines_auto(x, y, name_x, names_y, **kwgs):
    """ Retrieve the latex name of the branch and unit associated with ``x``.
    Then, plot with :py:func:`plot_lines`
    
    Parameters
    ----------
    x       : list(float)
        passed to :py:func:`plot_lines`
    y       : numpy.array(uncertainties.ufloat) or numpy.array(float) or list(numpy.array(uncertainties.ufloat)) or list(numpy.array(float))
        passed to :py:func:`plot_lines`
    name_x  : str
        passed to :py:func:`plot_lines`
    name_y  : str or list(str)
        passed to :py:func:`plot_lines`
    **kwgs  : dict
        passed to :py:func:`plot_lines`
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure of the plot
    ax : matplotlib.figure.Axes or list(matplotlib.figure.Axes)
        Axis of the plot or list of axes of the plot
    """
    latex_branch, unit = pt.get_latex_branches_units(name_x)
    latex_name_x = pt.get_label_branch(latex_branch, unit)
    return plot_lines(x, y, name_x, names_y, 
                      latex_name_x=latex_name_x,
                      **kwgs)


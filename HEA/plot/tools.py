"""
* Save a plot
* Some functions to change how the plots look (grid, logscale, label of the ticks, text ['LHCb preliminary'], ...)which are used in :py:mod:`HEA.plot.histogram`, :py:mod:`HEA.plot.fit` and :py:mod:`HEA.line`
"""

from HEA.config import loc, default_fontsize, default_project

import numpy as np
from pandas.core.series import Series

import os.path as op
from copy import deepcopy

from HEA.tools.da import add_in_dic, el_to_list
from HEA.tools.dir import create_directory
from HEA.tools import string
from HEA.definition import RVariable
from HEA.tools import assertion





#################################################################################################
########################################### Saving ##############################################
################################################################################################# 

# SAVING FIGURE ================================================

def save_fig(fig, fig_name, folder_name=None, default_fig_name=None, directory=f"{loc['plots']}/"):
    """ Save fig in a file, in ``{directory}/{folder_name}/{fig_name or default_fig_name}.pdf``
    
    Parameters
    ----------
    fig                : matplotlib.figure.Figure
        figure to save
    fig_name         : str or None
        name of the file
    default_fig_name : str or None
        alternative name of the file if fig_name is None
    folder_name       : str or None
        name of the folder where the file will be saved
    directory         : str
         root path where to save the plot
    
    """
    folder_name = string._remove_space(folder_name)
    directory = create_directory(directory, folder_name)
    if fig_name is None:
        fig_name = default_fig_name

    fig_name = string._remove_space(string._remove_latex(fig_name)).replace('/','d')
    path = op.join(directory, fig_name)
    #Save the plot as a PDF document into our PLOTS folder (output/plots as defined in bd2dst3pi/locations.py)
    fig.savefig(path + '.pdf', dpi=600, bbox_inches="tight")
#     fig.savefig(path + '.jpg', dpi=600, bbox_inches="tight")
    print(f"Figure saved in {path}")

    
    
#################################################################################################
############################### Tool functions for plotting #####################################
################################################################################################# 

### Computation -------------------------------------------------------------------
def _redefine_low_high(low, high, data):
    """ if low or high is not None, return global min (``low``) or max (``high``) of all the data in data, respectively.
    
    Parameters
    ----------
    low    : float or None
        low value of the range
    high   : float or None
        high value of the range
    data   : pandas.Series or list of pandas.Series 
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
    
    return low,high

### Text formatting -------------------------------------------------------------------

def _get_fig_name_given_BDT_cut(fig_name=None, cut_BDT=None, branch="", data_name=None): # previously: get_fig_name_title_BDT
    """ Return the new name of the file and the new title given the cut on the BDT 
    
    Parameters
    ----------
    fig_name   : str
        initial name of the file
    cut_BDT     : float
        cut on the BDT (we keep ``BDT > {cut_BDT}``)
    branch    : float
        a name of branch (e.g., ``'B0_M'``)
    data_name   : str or None
        name of the plotted data
    
    Returns
    -------
    fig_name: str
        new fig_name
    
    Examples
    --------
    >>> _get_title_given_BDT_cut("fig_name", -0.1, 'B0_M', 'MC')
    "fig_BDT_name-0.1"
    >>> _get_title_given_BDT_cut(None, -0.1, 'B0_M', 'MC')
    "B0_M_MC_BDT-0.1"
    """
    
    assert (fig_name is not None) or (data_name is not None)
    
    if fig_name is None:
        fig_name = string.add_text(branch, data_name, '_')
    
    # Title with BDT
    if cut_BDT is not None:
        fig_name = string.add_text(fig_name, f'BDT{cut_BDT}')
    
    return fig_name

def _get_title_given_BDT_cut(title, cut_BDT): # previously: get_fig_name_title_BDT
    """ Return the the new title given the cut on the BDT 

    Parameters
    ----------
    title       : str
        initial title
    cut_BDT     : float
        cut on the BDT (we keep ``BDT > {cut_BDT}``)
    
    Returns
    -------
    title: str
        new title
    
    Examples
    --------
    >>> _get_title_given_BDT_cut("title", -0.1)
    "title - BDT > -0.1"
    """
    
    # Title with BDT
    if cut_BDT is not None:
        title = string.add_text(title, f"BDT $>$ {cut_BDT}", ' - ')
    
    return title

    
### Core of the plot -------------------------------------------------------------------
def show_grid(ax, which='major', axis='both'):
    """show grid
    
    Parameters
    ----------
    ax    : matplotlib.axes.Axes
        axis where to show the grid
    which : 'both', 'major' or 'minor'
        which grid to show
    """
    ax.grid(b=True, axis=axis, which=which, color='#666666', linestyle='-', alpha = 0.2)

# def change_y_range(ax, factor_ymax=1.1, ymin_to_0=True): # change_ymax
#     """ multiple ymax of the plot by factor
#     Parameters
#     ----------
#     ax           : matplotlib.axes.Axes
#         Axis to change
#     factor_ymax  : float
#         factor by which ymax is multiplied
#     ymin_to_0    : bool
#         if True, ymin is set at 0
    
    
#     """
#     ymin, ymax = ax.get_ylim()
    
#     if ymin_to_0:
#         ymin = 0
    
#     ax.set_ylim(ymin, ymax*factor_ymax)

def change_range_axis(ax, factor_max=1.1, min_to_0=True, axis='y'): # previously: change_max
    """ multiple the max range of an axis of a plot by ``factor``
    
    Parameters
    ----------
    ax        : matplotlib.axes.Axes
        axis where to plot
    factor_max: float
        factor by which xmax and/or ymax are/is multiplied
    min_to_0  : bool
        if True, the min of the range of the considered axis is set at 0
    axis      : 'x', 'y' or 'both'
        Axis where to change the range limits
    """
    if axis=='x' or axis=='both':
        xmin, xmax = ax.get_xlim()
        if min_to_0:
            xmin = 0
        ax.set_xlim(xmin, xmax * factor_max)
    
    if axis=='y' or axis=='both':
        ymin, ymax = ax.get_ylim()
        if min_to_0:
            ymin = 0
        ax.set_ylim(ymin,ymax*factor_max)
    
    
def set_label_ticks(ax, labelsize=default_fontsize['ticks'], axis='both'):
    """Set label ticks to size given by labelsize
    
    Parameters
    ----------
    ax        : matplotlib.axes.Axes
        axis where to plot
    labelsize : float
        fontsize of the ticks
    axis      : 'x', 'y' or 'both'
        Axis where to change the tick fontsize
    """
    ax.tick_params(axis=axis, which='both', labelsize=labelsize)


def set_log_scale(ax, axis='both'):
    """Set logscale to the specified axes
    
    Parameters
    ----------
    ax   : matplotlib.axes.Axes
        axis where to change the scale into a log one
    axis : 'both', 'x' or 'y'
        axis with a log scale
    """
    if axis == 'both' or axis == 'x':
        ax.set_xscale('log')
    if axis == 'both' or axis == 'y':
        ax.set_yscale('log')

def set_text_LHCb(ax, text=default_project['text_plot'], fontsize=default_fontsize['text'], pos=None):
    """ Put a text on a plot 
    
    Parameters
    ----------
    ax       : matplotlib.axes.Axes
        axis where to plot
    text     : str
        text to plot
    fontsize : float
        fontsize of the text
    pos      : dict, list or str
        Three possibilities
        
        - dictionnary with these keys
        
            - ``'x'``: position of the text along the x-axis
            - ``'y'``: position of the text along the y-axis
            - ``'ha'``: horizontal alignment
            - ``fontsize``: fontsize of the text
            - ``text`` : text to plot
            
        - list: ``[x, y, ha]``
        
        - str: alignment ``'left'`` or ``'right'``.
        
            - if 'left', ``x = 0.02`` and ``y = 0.95``
            - if 'right', ``x = 0.98`` and ``y = 0.95``.
            
        These values are also the default values for the dictionnary input mode.
        These parameters are passed to ``ax.text()``.
    
    Returns
    -------
    matplotlib.text.Text
        the text element that ``plt.text`` returns
    """
    if pos is not None:
        info = deepcopy(pos)
        if isinstance(pos, dict):
            ha = info['ha']
            if ha=='left' :
                x = 0.02 if 'x' not in info else info['x']
                y = 0.95 if 'y' not in info else info['y']
            elif ha=='right':
                x = 0.98 if 'x' not in info else info['x']
                y = 0.95 if 'y' not in info else info['y']
            
            add_in_dic('fontsize', pos, fontsize)
            add_in_dic('text', pos, text)
            
            fontsize = pos['fontsize']
            text = pos['text']
            
        elif isinstance(pos, str):
            if pos=='left':
                x = 0.02
                y = 0.95
                ha = 'left'
            elif pos=='right':
                x = 0.98
                y = 0.95
            ha = 'right'
        elif isinstance(pos, list):
            x = pos[0]
            y = pos[1]
            ha = pos[2]
    
        return ax.text(x, y, text, verticalalignment='top', horizontalalignment=ha, 
                    transform=ax.transAxes, fontsize=fontsize)

    
    
def fix_plot(ax, factor_ymax=1.1, show_leg=True, fontsize_ticks=default_fontsize['ticks'], 
             fontsize_leg=default_fontsize['legend'], loc_leg='best', ymin_to_0=True, pos_text_LHC=None, axis='y'):
    """ Some fixing of plot parameters (fontsize, ymax, legend)
    
    Parameters
    ----------
    ax              : matplotlib.axes.Axes
        axis where to plot
    factor_ymax     : float
        multiplicative factor of ymax
    show_leg        : bool
        True if the legend needs to be shown
    fontsize_ticks  : float
        fontsize of the ticks
    fontsize_leg    : float
        fontsize of the legend
    loc_leg         : str
        location of the legend
    ymin_to_0       : bool
        if ``True``, the min value of the y-axis is set at 0
    pos_text_LHC    : dict, list or str
        passed to :py:func:`HEA.plot.tools.set_text_LHCb` as the ``pos`` argument.
        
    axis            : 'y' or 'x'
        if ``axis`` is ``'x'``, ``ymin_to_0`` and ``factor_ymax`` acts on the x-axis
    """
    
    if factor_ymax is not None:
        change_range_axis(ax, factor_ymax, ymin_to_0, axis)
    
    set_label_ticks(ax)
    if show_leg:
        ax.legend(fontsize = fontsize_leg, loc=loc_leg)
    
    set_text_LHCb(ax, pos=pos_text_LHC)
    
def get_latex_branches_units(branches):
    """ Get the latex branches and units associated with ``branches``
    
    Parameters
    ----------
    branchs: str or list(str)
        branch or list of branches
    
    Returns
    -------
    latex_branchs: str or list(str)
        latex name of the branch or list of latex names of the branches
    units: str or list(str)
        unit of the branch or list of units of the branches
    """
    if assertion.is_list_tuple(branches):
        latex_branches = [None, None]
        units      = [None, None]
        for i in range(2):
            latex_branches[i], units[i] = RVariable.get_latex_branch_unit_from_branch(branches[i])
        return latex_branches, units
    else:
        return RVariable.get_latex_branch_unit_from_branch(branches)

def _set_folder_name_from_data_name(kwargs, data_names):
    """ Change the key `"folder_name"` of a dictionnary by the list of data names (in place)
    
    Parameters
    ----------
    kwargs: dict
        with the key `"folder_name"`
    data_names : str or list(str)
        name of the dataset(s)
    """
    add_in_dic('folder_name', kwargs)
    if kwargs['folder_name'] is None:
        if isinstance(data_names, str):
            str_data_names = data_names
        else:
            str_data_names = string.list_into_string(data_names)
        kwargs['folder_name'] = str_data_names

def _unit_between_brackets(unit, show_bracket=True): # previously: redefine_unit
        """Return the correct string to show the units in the labels of plots

        Parameters
        ----------
        unit         : str
            Unit in the label
        show_bracket : bool
            Is the bracket shown?

        Returns
        -------
            text with
            
            * a space before it
            * between the squared brackets ``'['`` if ``show_bracket`` is ``True``
        """
        if show_bracket:
            bracket = '['
        else:
            bracket = None
        return string.string_between_brackets(unit, bracket=bracket)           

def set_label_branch(ax, latex_branch, unit=None, data_name=None, fontsize=default_fontsize['label'], axis='x'):
        """ set the label branch ``(data_name) [unit]`` for the axis specified by ``'axis'``

        Parameters
        ----------
        ax            : matplotlib.axes.Axes
            axis where to show the grid
        latex_branch  : str
            latex name of the branch that was plotted
        unit          : str
            unit of the branch that was plotted
        data_name     : str or None
            Name of the data, in case in needs to be specified in the label of the axis between parentheses
        fontsize      : float
            fontsize of the label
        axis          : 'x' or 'y' or 'both'
            Axis where to set the label
        """
        unit_text = _unit_between_brackets(unit)
        data_name_text = string.string_between_brackets(data_name, bracket = '(')

        label = "%s%s%s" %(latex_branch, data_name_text, unit_text)
        if axis == 'x' or axis == 'both':
            ax.set_xlabel(label, fontsize=fontsize)
        if axis == 'y' or axis == 'both':
            ax.set_ylabel(label, fontsize=fontsize)
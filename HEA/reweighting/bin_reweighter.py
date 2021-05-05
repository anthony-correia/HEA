""" Reweighting with fitted spline
"""

import numpy as np
from scipy.interpolate import splrep, splev

from HEA.tools import dist
import HEA.plot.tools as pt
import HEA.plot.histogram as hist
from HEA.tools.string import str_number_into_latex
from HEA.tools.serial import (
    dump_pickle, dump_json, 
    retrieve_pickle, retrieve_json
)

MC_color = 'r'
reweighted_MC_color = 'b'
data_color = 'indigo'

class BinReweighter():
    """
    Reweighting MC to align it with data.
    
    Attributes
    ----------
    data : pd.DataFrame
        True data
    MC   : pd.DataFrame
        Simulated data to reweight
    n_bins : int or dict
        number of bins in the histograms
    name : str
        name of the bin reweighted, used to save the tck's.
    MC_weights : array-like
        weights to be applied to MC
    data_weights : array-like
        weights to be applied to data
    column_ranges : dict(str: list(float or None, float or None))
        Dictionnary that contains the ranges of some columns
    MC_color, data_color, reweighted_MC_color: str
        colors used in the plots
    folder_name : str
        Used to create the name of the folder 
        where to save the figures.
    column_tcks : dict
        Dictionnary that associates to a column the tck of the spline, 
        which was used to compute the weights.
    MC_color, data_color, reweighted_MC_color: str
            colors used in the plots
    data_label, MC_label, reweighted_MC_label: str
        Labels of data, MC and reweighted MC used in the plots
    normalise: bool
        Are the distributions normalised in the plots?
    """
    
    def __init__(self, data, MC, 
                 n_bins, name,
                 MC_weights=None, data_weights=None,
                 column_ranges={},
                 MC_color=MC_color,
                 reweighted_MC_color=reweighted_MC_color,
                 data_color=data_color,
                 folder_name=None,
                 data_label='Data',
                 MC_label="Original MC",
                 reweighted_MC_label="Reweighted MC",
                 normalise=True
                 ):
        """ Produce a BinReweighter
        
        Parameters
        ----------
        data : pd.DataFrame or dict[str: tuple]
            True data
        MC   : pd.DataFrame
            Simulated data to reweight
        n_bins : int
            number of bins in the histograms
        name : str
            name of the bin reweighted, used to save the tck's.
        MC_weights : array-like
            weights to be applied to MC
        data_weights : array-like
            weights to be applied to data
        column_ranges : dict(str: list(float or None, float or None))
            Dictionnary that contains the ranges of some columns
        MC_color, data_color, reweighted_MC_color: str
            colors used in the plots
        folder_name : str
            Used to create the name of the folder 
            where to save the figures.
        data_label, MC_label, reweighted_MC_label: str
            Labels of data, MC and reweighted MC used in the plots
        normalise: bool
            Are the distributions normalised in the plots?            
        edges_columns: dict
            associate a column with bin edges
        data_counts_columns, data_err_columns: dict
            associate a column with bin counts and errors
            in data
        MC_counts_columns, MC_err_columns: dict
            associate a column with bin counts and errors
             in MC
        
        Notes
        -----
        The arguments finishing with ``_columns`` allow
        to load data in the :py:class:``BinReweighter`` using
        bin counts and edges directly.
        
        In this case, the new reweighted distribution has to be
        loaded every time a branch is reweighted, with 
        :py:func:`BinReweighter.load_reweighted_MC_counts`
        """
        
        
        if folder_name is None:
            folder_name = name
        
        self.data = data
        self.MC = MC
        self.MC_weights = MC_weights
        self.data_weights = data_weights
        self.n_bins = n_bins
        self.name = name
        self.column_ranges = column_ranges
        self.MC_color = MC_color
        self.reweighted_MC_color = reweighted_MC_color
        self.data_color = data_color
        self._trained_columns = []
        self.folder_name = folder_name
        
        self.data_label = data_label
        self.MC_label = MC_label
        self.reweighted_MC_label = reweighted_MC_label
        
        self.normalise = normalise
        
        self.column_tcks = {}
        self.edges_columns = {}
        self.data_counts_columns = {}
        self.data_err_columns = {}
        self.MC_counts_columns = {}
        self.MC_err_columns = {}
        self.reweighted_MC_counts_columns = {}
        self.reweighted_MC_err_columns = {}    
    
    @staticmethod
    def from_counts(edges_columns,
                    data_counts_columns, data_err_columns,
                    MC_counts_columns, MC_err_columns,
                    **kwargs
                   ):
        """ Load the bin reweighter using 'histogrammed' data.
        
        Parameters
        ----------
        edges_columns: dict
            associate a column with bin edges
        data_counts_columns, data_err_columns: dict
            associate a column with bin counts and errors
            in data
        MC_counts_columns, MC_err_columns: dict
            associate a column with bin counts and errors
             in MC
        kwargs:
            other parameters passed to :py:func:``BinReweighted.__init__``
        """
        
        data = None
        MC = None
        n_bins = None
        
        self = BinReweighter(data, MC, n_bins, **kwargs)
        
        self.edges_columns = edges_columns
        self.data_counts_columns = data_counts_columns
        self.data_err_columns = data_err_columns
        self.MC_counts_columns = MC_counts_columns
        self.MC_err_columns = MC_err_columns
        
        self.reweighted_MC_counts_columns = {}
        self.reweighted_MC_err_columns = {}
        
        return self
        
    @staticmethod
    def from_MC(MC, name,
                n_bins=None,
                MC_weights=None,
                ):
        """ Load the binreweighter with only data
        """
        data = None
        data_weights = None
        data_color = None
        
        return BinReweighter(
            data=data, MC=MC, 
            n_bins=n_bins, name=name,
            MC_weights=MC_weights, 
            data_weights=data_weights,
            **kwargs
        )

    @property
    def trained_columns(self):
        """ Columns which weights have been applied to.
        """
        return self._trained_columns
    
    def load_reweighted_MC_counts(self, 
                                  MC_counts_columns, 
                                  MC_err_columns):
        """ Load the reweighted MC via a histogram
        (since the weights cannot be applied) directly
        in a histogram but rather in data.   
        
        Parameters
        ----------
        MC_counts_columns: dict
            Associates a column with bin counts
        MC_err_columns: dict
            Associates a column with uncertainties 
            in bin counts
        """
        for column in MC_counts_columns.keys():
            counts = MC_counts_columns[column]
            err = MC_err_columns[column]
            
            self.reweighted_MC_counts_columns[column] = counts
            self.reweighted_MC_err_columns[column] = err
    
    ## HISTOGRAMS =============================================================
    
    def get_n_bins(self, column):
        """ Get the number of bins for the histogram of a column
        
        Parameters
        ----------
        column: str
            column (e.g., `B0_M`)
        
        Returns
        -------
            Number of bins required for this histogram according to the attributes
            of the class
        """
        
        if self.n_bins is None or isinstance(self.n_bins, int):
            return self.n_bins
        else:
            assert isinstance(self.n_bins, dict)
            assert column in self.n_bins
            
            return self.n_bins[column]
        
    
    def get_low_high(self, column, low=None, high=None):
        """ get low and high value of a column
        
        Parameters
        ----------
        column: str
            name of the column in the dataframes
            
            
        Returns
        -------
        low: float or None
            ``low`` if it was initially not ``None``,
            or low value written in ``column_ranges``.
            If the data is provided directly through
            a histogram, it is ``edges[0]``.
        high: float or None
            ``high`` if it was initially not ``None``,
            or high value written in ``column_ranges``.
            If the data is provided directly through
            a histogram, it is ``edges[-1]``.
        """
        if self.counts_specified(column):
            edges = self.edges_columns[column]
            if low is None:
                low =  edges[0]
                
            if high is None:
                high =  edges[-1]
        else:    
            if low is None:
                if column in self.column_ranges:
                    low, _ = self.column_ranges[column]
                else:
                    low = None

            if high is None:
                if column in self.column_ranges:
                    _, high = self.column_ranges[column]
                else:
                    high = None
        
        
            datasets = [self.MC[column]]
            if self.data is not None:
                datasets += [self.data[column]]
            low, high = pt._redefine_low_high(low=low, 
                                              high=high, 
                                              data=datasets)
        
        return low, high
    
    def counts_specified(self, column):
        """ Is the column has been provided using a histogram?
        (and not directly the data)
        
        Parameters
        ----------
        column: str
            column
            
        Returns
        -------
        self.counts_specified: bool
            ``True`` if ``column`` is in ``edges_columns``
            and ``data_counts_columns`` and ``MC_counts_columns``            
        """
        
        return column in self.edges_columns \
            and column in self.data_counts_columns \
            and column in self.MC_counts_columns
    
    def get_data_counts(self, column):
        """ Get the data counts and errors of binned data
        
        Parameters
        ----------
        column: str
            column that we want to get the counts of
            
        Returns
        -------
        data_counts : array-like
            counts in data
        data_err: array-like
            errors in counts in data
        """
        
        data_counts = self.data_counts_columns[column]
        data_err = self.data_err_columns[column]
        
        return data_counts, data_err
        
    def get_MC_counts(self, column, with_MC_weights=True):
        """ Get the MC counts and errors of binned data
        
        Parameters
        ----------
        column: str
            column that we want to get the counts of
        with_MC_weights: bool
            do we want the reweighted MC?
            
        Returns
        -------
        MC_counts : array-like
            counts in MC
        MC_err: array-like
            errors in counts in MC
        """
        
        if with_MC_weights \
            and column in self.reweighted_MC_counts_columns\
            and column in self.reweighted_MC_err_columns:
                
            MC_counts = self.reweighted_MC_counts_columns[column]
            MC_err = self.reweighted_MC_err_columns[column]
        else:
            if with_MC_weights:
                print("There is no reweighting weights yet.")
            
            MC_counts = self.MC_counts_columns[column]
            MC_err = self.MC_err_columns[column]
       
        return MC_counts, MC_err
    
    def get_data_MC_ratio(self, column, 
                          low=None, high=None, 
                          with_MC_weights=True):
        """ Get ``MC[column][bin i]/data[column][bin i]``
        
        Parameters
        ----------
        column: str
            name of the column in the dataframes
        
        Returns
        -------
        ratio_counts: array-like
            ``MC[column][bin i]/MC[column][bin i]`` histogram
        edges: array-like
            Edges of the bins
        ratio_err: array-like
            Errors on the ratios
        """
        if self.counts_specified(column):
            
            edges = self.edges_columns[column]
            centres = (edges[1:] + edges[:-1]) /2
            
            counts1, err1 = self.get_data_counts(column)
            counts2, err2 = self.get_MC_counts(column, with_MC_weights)
            
            ratio_counts, ratio_err = hist.divide_counts(
                counts1, counts2,
                err1, err2,
                self.normalise
            )
            
            
            return ratio_counts, centres, ratio_err
        
        else:
            low, high = self.get_low_high(column)

            MC_weights = self.MC_weights if with_MC_weights else None

            return dist.get_count_err_ratio(
                data1=self.data[column], 
                data2=self.MC[column], 
                n_bins=self.get_n_bins(column), 
                low=low, high=high,
                weights=[self.data_weights, 
                         MC_weights],
                normalise=self.normalise
            )
        
    
    ## PLOTTING ===============================================================
    
    def get_chi2_latex(self, column, low=None, high=None, 
                       with_MC_weights=False,
                      **kwargs):
        """ return ``$\\chi^2$ = <chi2 with two significant figures>``
        
        Parameters
        ----------
        column: str
            Name of the column to plot.
        low: float
            low value of the distribution.
            If ``None``, taken from the dict. ``column_ranges``.
        high : float
            high value of the distribution.
            If ``None``, taken from the dict. ``column_ranges``.
        kwargs: 
            passed to :py:func:`HEA.tools.dist.get_chi2_2samp`
            and :py:func:`HEA.tools.dist.get_chi2_2counts`
        
        with_MC_weights: bool
        
        Returns
        -------
        latex_chi2: str
            ``$\\chi^2$ = <chi2 with two significant figures>``
        
            
        """
        
        if self.counts_specified(column):
            counts1, err1 = self.get_data_counts(column)
            counts2, err2 = self.get_MC_counts(column, with_MC_weights)
            
            chi2 = dist.get_chi2_2counts(
                counts1, counts2,
                err1, err2,
                **kwargs
            )
            
        else:
            low, high = self.get_low_high(column, low, high)

            MC_weights = self.MC_weights if with_MC_weights else None 

            chi2 = dist.get_chi2_2samp(data1=self.MC[column], 
                                       data2=self.data[column], 
                                       n_bins=self.get_n_bins(column), 
                                       low=low, high=high, 
                                       weights1=MC_weights, 
                                       weights2=self.data_weights,
                                       **kwargs)
        
        chi2 = str_number_into_latex(f"{chi2:.2g}")
        return '$\\chi^2 = {}$'.format(chi2)
    
    def get_fig_folder_name(self, column, plot_reweighted,
                            mode='vs',
                            inter=None):
        """ Get the figure name and the folder name of 
        the figure that needs to be saved.
        
        Parameters
        ----------
        column: str
            name of the variable which is plotted
        plot_reweighed: bool
            is the reweighted data plotted?
        mode: 'vs' or 'd'
            If 'vs', superimposed plot. If 'd',
            divide plot.
        inter: int or None
            if not None, save the figure in a folder 
            ending with ``'_inter'``,
            and and ``_{inter}`` at the end of the name
            of the figure.
            This allows to save intermediate figures.
        
        Returns
        -------
        fig_name: str
            name of the figure
        second_folder: str
            name of the second folder
            where to save the plot
        """
        if self.data_weights is not None:
            data_name = 'sWeighted_data'
        else:
            data_name = 'data'
        
        if plot_reweighted or inter is not None:
            second_folder = f"bin_reweighted_MC_{mode}_{data_name}" 
        else:
            if self.MC_weights is None:
                second_folder = f"MC_{mode}_{data_name}"
            else:
                second_folder = f"sWeighted_vs_{data_name}"
        
        if inter is not None:
            second_folder = second_folder + '_inter'
            text_inter = f"_{inter}"
        else:
            text_inter = ""
        
        fig_name = column + text_inter
        
        return fig_name, second_folder
        
    def plot_hist(self, column, show_chi2=True,
                  plot_reweighted=None,
                  plot_original=True,
                  low=None, high=None,
                  inter=None, factor_ymax=1.5,
                  with_text_LHC=True, 
                  **kwargs):
        """ Plot the normalised histogram of column 
        for MC and data
        
        Parameters
        ----------
        column: str
            Name of the column to plot.
        show_chi2 : bool
            if True, show the chi2 between MC and LHCb data
        plot_reweighted: bool
            if the reweighted MC is plotted
        plot_not_reweighted: bool
            if the original MC is plotted
        low: float
            low value of the distribution.
            If ``None``, taken from the dict. ``column_ranges``.
        high : float
            high value of the distribution.
            If ``None``, taken from the dict. ``column_ranges``.
        inter: int
            passed to 
            :py:func:`HEA.reweighting.bin_reweighter.BinReweighter.get_fig_folder_name`
        with_text_LHC: bool
            Do we plot the LHC text 
            (e.g., "LHCb simulation", "LHCb preliminary, ...)

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure of the plot 
            (only if ``ax`` is not specified)
        ax : matplotlib.figure.Axes
            Axis of the plot 
            (only if ``ax`` is not specified)
        """
        if not self.counts_specified(column) and self.data is None:
            show_chi2 = False
                
        if self.counts_specified(column):
            kwargs['edges'] = self.edges_columns[column]
            low, high = None, None
        else:
            low, high = self.get_low_high(column, low, high)
        
        # What to plot
        
        if self.counts_specified(column):
            if plot_reweighted \
                and column not in self.reweighted_MC_counts_columns:
                
                print("No reweighted histogram available for MC")
                plot_reweighted = False
                plot_original = True
                
            if plot_reweighted is None and \
                column in self.reweighted_MC_counts_columns:
                
                plot_reweighted = True
                
        else:
            if plot_reweighted and self.MC_weights is None:
                print("No reweighting available for MC")
                plot_reweighted = False
                plot_original = True
            if plot_reweighted is None \
                and self.MC_weights is not None:
                
                plot_reweighted = True           
                
        samples_dict = {}
        
        alpha = []
        colors = []
        bar_modes = []
        labels = []
        
        if self.counts_specified(column):
            weights = None
        else:
            weights = []
        
        # Original MC
        if plot_original:
            if self.counts_specified(column):
                MC_counts, MC_err = self.get_MC_counts(
                    column, 
                    with_MC_weights=False
                )
                
                samples_dict[self.MC_label] = (MC_counts, MC_err)
                
            else:
                samples_dict[self.MC_label] = self.MC
                weights.append(None)
            
            if show_chi2:
                labels.append(
                    ', ' + self.get_chi2_latex(
                        column, 
                        low=low, high=high, 
                        with_MC_weights=False)
                )
            else:
                labels.append(None)
            
            alpha.append(0.7)
            colors.append([None, self.MC_color])
            bar_modes.append(True)
            
        else:
            labels.append(None)
        
        # Reweighted MC
        
        if plot_reweighted:
            if self.counts_specified(column):
                reweighted_MC_counts, reweighted_MC_err = \
                    self.get_MC_counts(column, with_MC_weights=True)
                
                samples_dict[self.reweighted_MC_label] = \
                    (reweighted_MC_counts, reweighted_MC_err)

            else:    
                samples_dict[self.reweighted_MC_label] = self.MC
                weights.append(self.MC_weights)
            
            if show_chi2:
                labels.append( ', ' +
                              self.get_chi2_latex(
                                  column, 
                                  low=low, high=high,
                                  with_MC_weights=True)
                             )
            else:
                labels.append(None)                                      
            alpha.append(0.4)
            colors.append(self.reweighted_MC_color)
            bar_modes.append(True)
        
        else:
            labels.append(None)
        
        # Data
        
        if self.counts_specified(column):
            data_counts, data_err = self.get_data_counts(column)
            samples_dict[self.data_label] = (data_counts, data_err)

        else:
            if self.data is not None:
                samples_dict[self.data_label] = self.data
                weights.append(self.data_weights)
        
        if self.counts_specified(column) or self.data is not None:
            bar_modes.append(False)
            alpha.append(1)
            labels.append(None)
            colors.append(self.data_color)
        
        # Plot       
        fig_name, second_folder = self.get_fig_folder_name(
            column, plot_reweighted,
            mode='vs',
            inter=inter)
        
        if self.data is not None or self.counts_specified(column):
            LHC_text_type = 'data_MC'
        else: 
            LHC_text_type = 'MC'

        if with_text_LHC:
            pos_text_LHC = {'ha': 'left',
                          'type': LHC_text_type,
                          'fontsize':20}
        else:
            pos_text_LHC = None

        return hist.plot_hist_auto(
            samples_dict,
            column, 
            fig_name=fig_name,
            folder_name=f"{self.folder_name}/{second_folder}",
            n_bins=self.get_n_bins(column),
            low=low,high=high,
            bar_mode=bar_modes,
            colors=colors,
            factor_ymax=factor_ymax,
            density=self.normalise,
            pos_text_LHC=pos_text_LHC,
            alpha=alpha,
            weights=weights,
            labels=labels, loc_leg='upper right',
            **kwargs
        )
    
    
    def plot_ratio(self, column, 
                   plot_reweighted=None,
                   plot_original=True,
                   low=None, high=None,
                   plot_spline=None,
                   inter=None,
                   with_text_LHC=False,
                   show_chi2=True,
                   **kwargs):
        """ Plot ``MC[column][bin i]/data[column][bin i]``
        
        Parameters
        ----------
        column: str
            Name of the column to plot.
        plot_reweighted: bool
            if the reweighted MC is plotted
        plot_not_reweighted: bool
            if the original MC is plotted
        low: float
            low value of the distribution.
            If ``None``, taken from the dict. ``column_ranges``.
        high : float
            high value of the distribution.
            If ``None``, taken from the dict. ``column_ranges``.
        inter: int
            if not None, save the figure in a folder 
            ending with ``'_inter'``,
            and and ``_{inter}`` at the end of the name
            of the figure.
            This allows to save intermediate figures.
        with_text_LHC: bool
            Do we plot the LHC text 
            (e.g., "LHCb simulation", "LHCb preliminary, ...)
        **kwargs : dict
            passed to 
            :py:func:`HEA.plot.histogram.plot_divide_alone`
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure of the plot 
            (only if ``ax`` is not specified)
        ax : matplotlib.figure.Axes
            Axis of the plot 
            (only if ``ax`` is not specified)
        
        """
        
        if not self.counts_specified(column) and self.data is None:
            show_chi2 = False
        
        if self.counts_specified(column):
            edges = self.edges_columns[column]
            centres = (edges[1:] + edges[:-1]) / 2
            
            # Get bin width
            bin_widths = edges[1:] - edges[:-1]
            if np.all(bin_widths == bin_widths[0]):
                bin_width = bin_widths[0]
            else:
                # if non-uniform bin width
                # bin width not well defined
                bin_width = None
            
            low = edges[0]
            high = edges[-1]
            
            if plot_reweighted \
                and column not in self.reweighted_MC_counts_columns:
                
                print("No reweighted histogram available for MC")
                plot_reweighted = False
                plot_original = True
                
            if plot_reweighted is None and \
                column in self.reweighted_MC_counts_columns:
                
                plot_reweighted = True
            
            counts_err_dict = {}
            
        else:
#             assert self.data is not None
#             assert (plot_reweighted or plot_original)
            
            low, high = self.get_low_high(column, low, high)
            bin_width = hist.get_bin_width(low, high, self.get_n_bins(column))
        
        
            if plot_reweighted and self.MC_weights is None:
                print("No reweighting available for MC")
                plot_reweighted = False
                plot_original = True

            if plot_reweighted is None and self.MC_weights is not None:
                plot_reweighted = True
        
            weights_dict = {}
        
        labels_dict = {}
        colors_dict = {}
                
        if plot_original:
            colors_dict['original'] = self.MC_color
            labels_dict['original'] = f"{self.MC_label}, "
            if show_chi2:
                labels_dict['original'] += self.get_chi2_latex(
                    column,
                    low=low, high=high,
                    with_MC_weights=False
                )
            
            if self.counts_specified(column):
                counts_err_dict['original'] = self.get_MC_counts(
                    column,
                    with_MC_weights=False
                )
                
            else:
                weights_dict['original'] = [
                    self.data_weights,
                    None
                ]
                
            
        if plot_reweighted:
            colors_dict['reweighted'] = self.reweighted_MC_color
            labels_dict['reweighted'] = f"{self.reweighted_MC_label}, "
            if show_chi2:
                labels_dict['reweighted'] += self.get_chi2_latex(
                    column,
                    low=low, high=high,
                    with_MC_weights=True
                )
            
            if self.counts_specified(column):
                counts_err_dict['reweighted'] = self.get_MC_counts(
                    column,
                    with_MC_weights=True
                )
            else:
                weights_dict['reweighted'] = [self.data_weights,
                                          self.MC_weights]
        
        fig, ax = hist.get_fig_ax()
        
        # Plotting
        
        
        if self.counts_specified(column):
            data_counts, data_err = self.get_data_counts(column)
            
            for type_MC in counts_err_dict.keys():
                hist.plot_divide_alone(
                    ax, 
                    data1=(data_counts, data_err), 
                    data2=counts_err_dict[type_MC],
                    color=colors_dict[type_MC], 
                    label=labels_dict[type_MC],
                    bin_centres=centres,
                    edges=edges,
                    **kwargs
                )
                
        else:
            for type_MC in weights_dict.keys():
                _, bin_centres, _ = hist.plot_divide_alone(
                    ax, 
                    data1=self.data[column], 
                    data2=self.MC[column], 
                    low=low, high=high, n_bins=self.get_n_bins(column), 
                    color=colors_dict[type_MC], 
                    label=labels_dict[type_MC], 
                    weights=weights_dict[type_MC],
                    **kwargs
                )
        
        # Labels
        data_names = ['data', 'MC']
        latex_branch, unit = pt.get_latex_branches_units(column)
        hist.set_label_divided_hist(ax, latex_branch, unit,
                                    bin_width, 
                                    data_names=data_names)
        
        if with_text_LHC:
            pos_text_LHC = {'ha': 'right', 
                            'type': 'data_MC',
                            'fontsize': 20}
        else:
            pos_text_LHC = None

        pt.fix_plot(ax, factor_ymax=1.4, show_leg=True,
                    ymin_to_0=False,
                    pos_text_LHC=None,
                    loc_leg='upper left')
        
        # Spline
        if plot_spline is None:
            plot_spline = column in self.column_tcks
        if plot_spline:
#             x = np.linspace(low, high, self.n_bins*4) 
#             x+= (high - low) / (4 * self.n_bins) / 2
            x = np.linspace(low, high, 1000)
    
    
            spline = self.get_spline(column, x)
            if spline is not None:
                ax.plot(x, spline, color='k')
        
        fig_name, second_folder = self.get_fig_folder_name(
            column, plot_reweighted,
            mode='d',
            inter=inter)
                
        pt.save_fig(fig, fig_name, 
                    folder_name=f"{self.folder_name}/{second_folder}"
                   )
        
        return fig, ax
       
    def plot_MC_weights(self, n_bins=None, inter=None):
        """ Plot the weight distribution, if it exists
        
        """
        
        if n_bins is None:
            n_bins = self.get_n_bins(column)
        
        if self.MC_weights is None:
            print("There are no MC weights!")
            return
        
        if inter is not None:
            text_inter = f'_{inter}'
            folder_text_inter = '_inter'
        text_inter = ""
        folder_text_inter = ""
        
        return hist.plot_hist_var(self.MC_weights,
                                  'Reweighting weights', 
                                  fig_name='reweighting_weight'+text_inter,
                                  folder_name=f"{self.folder_name}/bin_reweight_MC"+folder_text_inter,
                                  n_bins=self.get_n_bins(column),
                                  bar_mode=True,
                                  colors=self.reweighted_MC_color,
                                  factor_ymax=1.2,
                                  pos_text_LHC={'ha': 'right',
                                          'type': 'data_MC',
                                          'fontsize':20},
                                 )
        
        
        
    ## COMPUTE WEIGHTS ======================================================
    def fit_spline(self, column, k=3, 
                    recompute=False):
        """ Return the weights to apply to the weighted MC 
        to align it with weighted data.
        
        Parameters
        ----------
        column: str
            Name of the MC column to align to data column
        k: int
            degree of the spline fit
        recompute: bool
            Even if ``self.splines`` already contains the splin,
            does we need to recompute it anyway?
        return_bin_centres: bool
            Do we return the bin centres?
        Returns
        -------
        tck: tuple
            A tuple (t,c,k) containing the vector of knots,
            the B-spline coefficients, 
            and the degree of the spline.
            To be used with ``splev(x2, tck)``
        
        """
        
        if recompute or (column not in self.column_tcks):
            
            ratio_counts, bin_centres, ratio_err = \
                self.get_data_MC_ratio(column, 
                                       with_MC_weights=True)
            
            tck = splrep(x=bin_centres,
                         y=ratio_counts,
                         w=1/ratio_err, # lower error implies higher weight
                         k=k,
                        )
            self.column_tcks[column] = tck
        
        else:
            tck = self.column_tcks[column]
        
        return tck
        
    
    def get_spline(self, column, x):
        """
        Parameters
        ----------
        column: str
            Name of the MC column to align to data column
        **kwargs: dict
            passed to :py:func:`BinReweighter.get_splines_info`
        
        Returns
        -------
        splines: array_line
            An array of values representing the spline function 
            evaluated at the points given in ``bin_centres``
        bin_centres: array-like, optional
            Bin centres.
        """
        
        if column in self.column_tcks:
            tck = self.column_tcks[column]
            return splev(x, tck) 
        
        else:
            print("To get the spline, please first generate it" \
                "with fit_spline")
            return None
        
    def get_new_MC_weights(self, column, 
                          MC=None, fit_spline=False):
        """ From a spline fit, get the corresponding weights to
        MC data.
        To do so, apply the weight ``spine(value)`` to the event
        whose column is equal to ``value``
        
        Parameters
        ----------
        column: str
            Name of the MC column to align to data column
        MC : pd.DataFrame, optional
            Sample which the weights are computed for.
            if not specified, ``MC`` attribute.
        fit_spline: bool
            If ``True``, perform the fit of the spline to
            compute the MC weights.
        Returns
        -------
        """
        
        if MC is None:
            MC = self.MC
        
        return self.get_spline(column, MC[column])
        
    
    def apply_new_MC_weights(self, column, fit_spline=False):
        """ From a spline fit, apply the corresponding weights to
        MC data.
        To do so, apply the weight ``spine(value)`` to the event
        whose column is equal to ``value``
        
        Parameters
        ----------
        column: str
            Name of the MC column to align to data column
        fit_spline: bool
            If ``True``, perform the fit of the spline to
            compute the MC weights.
        """
        
        
        if fit_spline:
            self.fit_spline(column)
        
        ## Get the new weights
        
        if self.MC is not None: 
            new_MC_weights = self.get_spline(column, self.MC[column])
            if new_MC_weights is not None:
                if self.MC_weights is None:
                    self.MC_weights = new_MC_weights
                else:
                    self.MC_weights = self.MC_weights * new_MC_weights
        
        self._trained_columns.append(column)
    
    def apply_new_MC_weights_from_columns(self, columns):
        """ From a list of columns, fit the corresponding splines 
        and apply the corresponding weights to MC data 
        (through a loop).
                
        Parameters
        ----------
        columns: list(str)
            List of names of MC columns to align to the
            corresponding data columns            
        """
        
        for column in columns:
            print(f"Reweighting of column {column}")
            self.apply_new_MC_weights(column, 
                                      fit_spline=True)
    
    def save_weights(self):
        """ Save the ``tck`` of the splines
        """
        
        for column in self.trained_columns:
            dump_pickle(self.column_tcks[column], 
                        f"{self.name}_{column}_tck", self.folder_name)
            
        info_reweighting = {
            'columns': self.trained_columns,
            'n_bins' : self.get_n_bins(column),
        }

        dump_json(info_reweighting, f"{self.name}_info_reweighting", 
                  self.folder_name)
    
    def load_weights(self, name, 
                     folder_name):
        """ Load the splines (in order to apply the weights in a 
        possibly new dataset).
        
        Parameters
        ----------
        name: str
            name of the the reweighter used to compute the spline
            we are going to retrieve now
        folder_name: str
            folder name where the file is located. 
        
        Returns
        -------
        columns: list(str)
            Ordered list of columns where the wieghts were computed
        """
        
        ## Columns and number of bins
        info_reweighting = retrieve_json(f"{name}_info_reweighting", 
                                     folder_name)
        
        self.n_bins = info_reweighting['n_bins']
        
        for column in info_reweighting['columns']:
            self.column_tcks[column] = \
                retrieve_pickle(f"{name}_{column}_tck", 
                                folder_name)
        
        return info_reweighting['columns']

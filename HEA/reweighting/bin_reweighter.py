""" Reweighting with fitted spline
"""

import numpy as np
from scipy.interpolate import splrep, splev

from HEA.tools import dist
import HEA.plot.tools as pt
import HEA.plot.histogram as hist
from HEA.tools.string import str_number_into_latex
from HEA.tools.serial import dump_pickle

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
    column_tcks : dict
        Dictionnary that associates to a column the tck of the spline, 
        which was used to compute the weights.
    """
    
        
    def __init__(self, data, MC, 
                 n_bins, name,
                 MC_weights=None, data_weights=None,
                 column_ranges={},
                 MC_color=MC_color,
                 reweighted_MC_color=reweighted_MC_color,
                 data_color=data_color,
                 folder_name=None
                 ):
        """ Produce a BinReweighter
        
        Parameters
        ----------
        data : pd.DataFrame
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
        
        self.column_tcks = {}
     
    @property
    def trained_columns(self):
        """ Columns which weights have been applied to.
        """
        return self._trained_columns
    
        
    
    ## HISTOGRAMS =============================================================
    
    def get_low_high(self, column, low=None, high=None):
        """ get low and high value of a branch
        
        Parameters
        ----------
        column: str
            name of the branch in the dataframes
            
            
        Returns
        -------
        low: float or None
            ``low`` if it was initially not ``None``,
            or low value written in ``column_ranges``.
        high: float or None
            ``high`` if it was initially not ``None``,
            or high value written in ``column_ranges``.
        """
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
        
        low, high = pt._redefine_low_high(low=low, 
                                          high=high, 
                                          data=[self.data[column], 
                                                self.MC[column]])
        
        return low, high
    
    def get_data_MC_ratio(self, column, 
                          low=None, high=None, 
                          with_MC_weights=True):
        """ Get ``MC[column][bin i]/data[column][bin i]``
        
        Parameters
        ----------
        column: str
            name of the branch in the dataframes
        
        Returns
        -------
        ratio_counts: array-like
            ``MC[column][bin i]/MC[column][bin i]`` histogram
        centres: array-like
            Centres of the bins
        ratio_err: array-like
            Errors on the ratios
        """
        
        low, high = self.get_low_high(column)
        
        MC_weights = self.MC_weights if with_MC_weights else None
        
        return dist.get_count_err_ratio(
            data1=self.data[column], 
            data2=self.MC[column], 
            n_bins=self.n_bins, 
            low=low, high=high,
            weights=[self.data_weights, 
                     MC_weights], 
        )
        
    
    ## PLOTTING ===============================================================
    
    def get_chi2_latex(self, column, low, high, with_MC_weights=False):
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
            
        with_MC_weights: bool
        
        Returns
        -------
        latex_chi2: str
            ``$\\chi^2$ = <chi2 with two significant figures>``
        
            
        """
        
        low, high = self.get_low_high(column, low, high)
        
        MC_weights = self.MC_weights if with_MC_weights else None 
        
        chi2 = dist.get_chi2_2samp(data1=self.MC[column], 
                              data2=self.data[column], 
                              n_bins=self.n_bins, 
                              low=low, high=high, 
                              weights1=MC_weights, 
                              weights2=self.data_weights)
        
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
        
    def plot_hist(self, column, 
                  plot_reweighted=None,
                  plot_original=True,
                  low=None, high=None,
                  inter=None):
        """ Plot the normalised histogram of column 
        for MC and data
        
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
            passed to 
            :py:func:`HEA.reweighting.bin_reweighter.BinReweighter.get_fig_folder_name`
        
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure of the plot 
            (only if ``ax`` is not specified)
        ax : matplotlib.figure.Axes
            Axis of the plot 
            (only if ``ax`` is not specified)
        """
        
        # plot the reweighted MC data
        
        low, high = self.get_low_high(column, low, high)
        
        if plot_reweighted and self.MC_weights is None:
            print("No reweighting available for MC")
            plot_reweighted = False
            plot_original = True
        if plot_reweighted is None and self.MC_weights is not None:
            plot_reweighted = True           
                
        samples_dict = {}
        
        alpha = []
        colors = []
        bar_modes = []
        weights = []
        labels = []
        
        
        if plot_original:
            samples_dict['Original MC'] = self.MC
            alpha.append(0.7)
            colors.append([None, self.MC_color])
            bar_modes.append(True)
            weights.append(None)
            
            labels.append(', ' +
                self.get_chi2_latex(column, 
                                    low=low, high=high, 
                                    with_MC_weights=False)
            )
            
        if plot_reweighted:
            samples_dict['Reweighted MC'] = self.MC
            alpha.append(0.4)
            colors.append(self.reweighted_MC_color)
            bar_modes.append(True)
            weights.append(self.MC_weights)
            
            labels.append( ', ' +
                self.get_chi2_latex(column, 
                                    low=low, high=high, 
                                    with_MC_weights=True)
            )
                
        
        samples_dict['data'] = self.data
        colors.append(self.data_color)
        bar_modes.append(False)
        weights.append(self.data_weights)
        alpha.append(1)
        labels.append(None)
        
               
        fig_name, second_folder = self.get_fig_folder_name(
            column, plot_reweighted,
            mode='vs',
            inter=inter)
            
        return hist.plot_hist_auto(samples_dict,
                                   column, 
                                   fig_name=fig_name,
                                   folder_name=f"{self.folder_name}/{second_folder}",
                                   n_bins=self.n_bins,
                                   low=low,
                                   high=high,
                                   bar_mode=bar_modes,
                                   colors=colors,
                                   factor_ymax=1.5,
                                   pos_text_LHC={'ha': 'left',
                                           'type': 'data_MC',
                                           'fontsize':20},
                                   alpha=alpha,
                                   weights=weights,
                                   labels=labels
                                  )
    
    
    def plot_ratio(self, column, 
                   plot_reweighted=None,
                   plot_original=True,
                   low=None, high=None,
                   plot_spline=None,
                   inter=None,
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
        assert (plot_reweighted or plot_original)
        
        low, high = self.get_low_high(column, low, high)
        bin_width = hist.get_bin_width(low, high, self.n_bins)
        
        
        if plot_reweighted and self.MC_weights is None:
            print("No reweighting available for MC")
            plot_reweighted = False
            plot_original = True
        
        if plot_reweighted is None and self.MC_weights is not None:
            plot_reweighted = True
        
        dfs = {
            'data': self.data,
            'MC': self.MC
        }
        
        labels_dict = {}
        weights_dict = {}
        colors_dict = {}
                
        if plot_original:
            colors_dict['original'] = self.MC_color
            weights_dict['original'] = [self.data_weights,
                                        None
                                       ]
            labels_dict['original'] = "Original MC, "
            labels_dict['original'] += self.get_chi2_latex(
                column,
                low=low, high=high,
                with_MC_weights=False
            )
            
        if plot_reweighted:
            colors_dict['reweighted'] = self.reweighted_MC_color
            weights_dict['reweighted'] = [self.data_weights,
                                          self.MC_weights]
            labels_dict['reweighted'] = "Reweighted MC, "
            labels_dict['reweighted'] += self.get_chi2_latex(
                column,
                low=low, high=high,
                with_MC_weights=True
            )
        
        fig, ax = hist.get_fig_ax()
        
        # Plotting
        for type_MC in weights_dict.keys():        
            _, bin_centres, _ = hist.plot_divide_alone(
                ax, 
                data1=self.data[column], 
                data2=self.MC[column], 
                low=low, high=high, n_bins=self.n_bins, 
                color=colors_dict[type_MC], 
                label=labels_dict[type_MC], 
                weights=weights_dict[type_MC]
            )
        
        # Labels
        latex_branch, unit = pt.get_latex_branches_units(column)
        hist.set_label_divided_hist(ax, latex_branch, unit,
                                    bin_width, 
                                    data_names=list(dfs.keys())
                                   )
        
        pt.fix_plot(ax, factor_ymax=1.4, show_leg=True,
                    ymin_to_0=False,
                    pos_text_LHC={'ha': 'right', 
                                  'type': 'data_MC',
                                 'fontsize': 20}
                    , loc_leg='upper left')
        
        # Spline
        if plot_spline is None:
            plot_spline = column in self.column_tcks
        if plot_spline:
            x = np.linspace(low, high, self.n_bins*4) + (high - low) / (4 * self.n_bins) / 2
            spline = self.get_spline(column, x)
            if spline is not None:
                ax.plot(x, spline, color='k')
        
        fig_name, second_folder = self.get_fig_folder_name(
            column, plot_reweighted,
            mode='d',
            inter=inter)
                
        pt.save_fig(fig, fig_name, folder_name=f"{self.folder_name}/{second_folder}")
        
        return fig, ax
       
    def plot_MC_weights(self, n_bins=None, inter=None):
        """ Plot the weight distribution, if it exists
        
        """
        
        if n_bins is None:
            n_bins = self.n_bins
        
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
                                  n_bins=n_bins,
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
    
    def save_tcks(self):
        """ Save the ``tck`` of the splines for the applied sWeights.
        """
        
        for column in self.trained_columns:
            dump_pickle(self.column_tcks[column], f"{self.name}_{column}_tck", self.folder_name)
            dump_pickle(self.trained_columns, f"{self.name}_tck_columns", self.folder_name)
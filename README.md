The full documentation of the HEA package is available [here](https://hea.readthedocs.io/en/bd2dsttaunu/index.html).

The goal of the HEA library is to store a bunch of functions I often need to use. Some are very simple, and just allow to save time. For instance, `HEA.fit.fit.perform_fit` perform a fit using the `Minuit` minimizer, compute the errors with `Hesse`, print the result, and get the nominal values of and errors on the fitted parameters, putting them in a dictionnary. Some others are a bit more complex, for instance plotting a histogram with the right label, enabling all sort of inputs (ROOT, dataframe, counts & edges), ...

Moreover, the ouputs are saved in the right folders. The folders are specified in `HEA/config/config.ini`. 
`folder_name` specified in the functions that save some outputs, is the path following the path specified in the `config.ini` file.

An  advantage of using such a library is, if a bug is found in one of the functions, the issue can be solved globally, in the code in `HEA`. It also allows to save a lot of time.
A downside is that my analysis codes might be less transparent for a reader.


<!-- - `HEA.definition`: 
    - contains global variables
    - function to define the names of the branches, retrieve the latex names and units, etc.
- `HEA.create_config`: create the configuration file
- `HEA.config`: access the configuration file (contains variables such as default fontsize, and location where to save the plots, json, pickle and root files)
- `HEA.pandas_root`:
    - Load and save root files
- `HEA.fit.fit`:
    - Define some PDFs
    - Define the parameters from a dictionnary of training variables
    - Perform a fit (using zfit)
    - Check if the fit has an error (if it did not converge, has not a positive-definite hesse, ...)
    - Save the result of the fit in a JSON file
- `HEA.fit.params`:
    - Retrieve params
    - Print the result of a fit as a latex table
    - Some formatting function of the dictionnary containing the fitted parameters
- `HEA.fit.PDF`:
    - Get the number of d.o.f. of a model
    - get the chi2 of a model and the mean/std of the pull histogram
- `HEA.BDT`
    - Plot the super-imposed signal and background distributions of the training variables
    - Plot the correlation matrix of the training variables
    - Prepare the signal and background sample in order to be used for the training
    - Train a gradient or adaboost classifier
    - Test the BDT: ROC curve, over-traning plots, Kolmogorov-Smirnov test
    - Apply the BDT to LHCb data and save the ouput
- `HEA.tools.da`: function to manipulate lists
- `HEA.tools.serial`: dump and retrieve json and pickle files
- `HEA.tools.dir`: create directories
- `HEA.tools.string`: manipulate strings
- `HEA.tools.assertion`:
    - assert that a python object is a list or tuple
    - some assertion function
- `HEA.plot.tools`
    - Functions to change what the plot looks like (grid, logscale, 'LHCb preliminary' test, ...)
    - Save a plot
    - Functions to retrieve the name and unit of the variable (for the plot), from `variables.py`. This is used in the legend of the plots.
    - Some other tool functions useful to plot (for instance, remove the latex syntax of a string in order to use it in the name of the saved file, etc.)
- `HEA.plot.histogram`:
    - Plot 1D and 2D histograms (with the correct label for the axes)
    - Plot scatter plots
    - Plot histograms of the quotient of 2 variables (with the correct label for the axes)
- `HEA.plot.fit`:
    - Plot the distribution of a variable together with his fitted PDF and the pull histogram
    - Compute the number of degrees of freedom of a model only from the zfit model
    - Compute the reduced chi2 of a model
- `HEA.plot.line`: plot y vs x, or y1, y2, ... vs x (i.e., several curves as a function of x) -->
"""
HEA (High Energy Analysis) is a personal package to ease analyses of LHCb data:

The goal of the HEA library is to store a bunch of functions I often need to use. Some are very simple, and just allow to save time. For instance, ``HEA.fit.fit.perform_fit`` perform a fit using the ``Minuit`` minimizer, compute the errors with ``Hesse``, print the result, and get the nominal values of and errors on the fitted parameters, putting them in a dictionnary. Some others are a bit more complex, for instance plotting a histogram with the right label, enabling all sort of inputs (ROOT, dataframe, counts & edges), ...

Moreover, the ouputs are saved in the right folders. The folders are specified in ``HEA/config/config.ini``. 
``folder_name`` specified in the functions that save some outputs, is the path following the path specified in the ``config.ini`` file.

An  advantage of using such a library is, if a bug is found in one of the functions, the issue can be solved globally, in the code in ``HEA``. It also allows to save a lot of time.
A downside is that my analysis code might be less transparent for a reader.

* Load root files with :py:mod:`HEA.pandas_root`
* Produce plots with : :py:mod:`HEA.plot`
* Perform fits with :py:mod:`HEA.fit`
* Train a Boosted Decision Tree and test it with :py:mod:`HEA.ML`
* Reweight with :py:mod:`HEA.reweighting`
* Some tools are also available in :py:mod:`HEA.tools`. 
* Global variables and a class ``RVariable`` to concoct the label of the variables, are defined in ``HEA.definition``
"""

from HEA.definition import (
    RVariable,
    get_branches_from_raw_branches_functions,
    get_raw_branches_from_raw_branches_functions,
    print_used_branches_per_particles,
)

from HEA.pandas_root import load_dataframe, load_dataframe_YY_MM, perform_cut
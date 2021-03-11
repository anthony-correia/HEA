"""
HEA (High Energy Analysis) is a personal package to ease analyses of LHCb data:

* Load root files with :py:mod:`HEA.pandas_root`
* Produce plots with : :py:mod:`HEA.plot`
* Perform fits with :py:mod:`HEA.fit`
* Train a Boosted Decision Tree and test it with :py:mod:`HEA.ML`
* Reweight with :py:mod:`HEA.reweighting`
* Some tools are also available in :py:mod:`HEA.tools`. They are rather used within this package.
"""

from HEA.definition import (
    RVariable,
    get_branches_from_raw_branches_functions,
    get_raw_branches_from_raw_branches_functions,
    print_used_branches_per_particles
)

from HEA.pandas_root import load_dataframe, load_dataframe_YY_MM
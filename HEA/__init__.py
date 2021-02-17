"""
HEA (High Energy Analysis) is a personal package to ease analyses of LHCb data:

* Load root files with :py:mod:`HEA.pandas_root`
* Produce plots with : :py:mod:`HEA.plot`
* Perform fits with :py:mod:`HEA.fit`
* Train a Boosted Decision Tree with :py:mod:`HEA.fit`
* Some tools are also available in :py:mod:`HEA.tools`. They are rather used within this package.
"""

import os
import sys
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, basedir)

from HEA.definition import (
    RVariable, 
    get_branches_from_raw_branches_functions, 
    get_raw_branches_from_raw_branches_functions,
    print_used_branches_per_particles
)
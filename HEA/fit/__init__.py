"""
Package in order to:

* Perform fits and save the result: :py:mod:`HEA.fit.fit`
* Manage the parameters of the fit: :py:mod:`HEA.fit.params`
* Perform fits: :py:mod:`HEA.fit.fit`
* Get characteristics of zFit PDFs: :py:mod:`HEA.fit.PDF`
"""

import os
import sys
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, basedir)

from HEA.fit.params import (
    retrieve_params,
    get_params_without_BDT, get_params_without_err,
    json_to_latex_table, show_latex_table
)
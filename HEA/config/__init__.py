"""Module to handle the configuration file ``config.ini``"""

import os
import sys
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, basedir)

from HEA.plot.config import loc, default_fontsize, default_project
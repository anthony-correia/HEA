"""
Package with tool functions used within the library HEA.
"""

import os
import sys
basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, basedir)

from HEA.tools.assertion import is_list_tuple

from HEA.tools.da import el_to_list, list_included, show_dictionnary

from HEA.tools.serial import dump_pickle, dump_json, retrieve_pickle, retrieve_json

from HEA.tools.string import list_into_string, add_text

"""
Anthony Correia
08/02/21
Load the configuration file
"""

import os
import configparser

def _load_loc_file():
    """ Load the configuration file
    Returns
    -------
    configparser.Configparser
        `config.ini` file, which contains 
        - the location where to save output files
        - the default fontsizes
    """
    
    config = configparser.ConfigParser()
    
    path = os.path.join(os.path.dirname(__file__), 'config.ini')
    
    config.read(path)
    return config



config = _load_loc_file()

loc = config['location']
default_fontsize = dict(config['fontsize'])
default_project = config['project']

for key in default_fontsize.keys():
    default_fontsize[key] = float(default_fontsize[key])
    
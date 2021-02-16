"""
Load the configuration file ``config.ini``.

* The dictionnary `loc` contains the keys:
    * ``"root"`` : root directory of the project
    * ``"out"`` : output of analysis software
    * ``"plots"`` : plot folder
    * ``"tables"`` : latex table folder
    * ``"json"`` : json folder
    * ``"pickle"`` : pickle folder
    * ``"definition"`` : python file with the definitions of:
        * ``latex_particles``
        * ``definition_quantities``
* The dictionnary `default_fontsize` contains the default fontsizee of:
    * ``"ticks"``
    * ``"legend"``
    * ``"label"``
    * ``"text"``
    * ``"annotation"``
* The dictionnary `default_project` contains:
    * ``'name'``: the name of the project
    * ``'text_plot'``: what is written in the plot for the report (e.g., ``LHCb preliminary \\n 2fb$^-1$``)
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
    
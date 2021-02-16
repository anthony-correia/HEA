"""
Module to create directories
"""

from os import makedirs
import os.path as op 


def try_makedirs(path):
    """Create all the necessary directories of a path
    
    Parameters
    ----------
    path : str
         path
    """
    
    try:
        makedirs(path)
    except OSError:
        pass

def create_directory(directory, name_folder=None):
    """ Creates a folder in ``{directory}/{name_folder}``
    
    Parameters
    ----------
    directory : str
        name of the directory where to create the folder
    name_folder : str
        name of the folder to create
        
    Returns
    -------
    path: str
        ``{directory}/{name_folder}``
    
    """
    if name_folder is not None:
        path = op.join(directory, name_folder)
        try_makedirs(path)
    else:
        path = directory
    return path
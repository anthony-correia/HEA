"""
Anthony Correia
07/02/21
Serialisation into JSON and pickle files
"""

import pickle
import json
from functools import partial

from HEA.config import loc
from .dir import create_directory

library_from_str = {
    'pickle': pickle,
    'json' : json
}

def _dump(data, file_name=None, folder_name=None, library='json', byte_write=False, **params):
    """ Save the data in a file in {loc['out']}/{type_data}/
    Parameters
    ----------
    data      : python object
        element to be saved (can be a list)
        if type_data is 'json', must be a dictionnary
    file_name : str
        name of the pickle file
    folder_name : str
        name of folder where the json file is
        (if None, there is no folder)
    library : str
        'json' or 'pickle'
    byte_write: bool
        Write in byte mode
    **params: dict
        parameters passed to the dump function of the corresponding serial library.
    """
    directory = create_directory(loc[library], folder_name)
    path = f"{directory}/{file_name}.{library}"
    
    
    with open(path, 'w' + byte_write*'b') as f:
        library_from_str[library].dump(data, f, **params)
    
    print(f"{library.title()} file saved in {path}")

dump_pickle = partial(_dump, library='pickle', byte_write=True)    
dump_json   = partial(_dump, library='json', sort_keys=True, indent=4)

def _retrieve(file_name, folder_name=None, library='json', byte_read=False):
    """ Retrieve the content of a json file
    
    Parameters
    ----------
    file_name   : str
        name of the file
    folder_name : str
        name of folder where the json file is
        (if None, there is no folder)
    library : str
        'json' or 'pickle'
    byte_read: bool
        Read in byte mode    
    
    Returns
    -------
    Python object or dic
        dictionnary that contains the variables stored in the json file
        in {loc['json']}/{folder_name}/{name_data}.json
        or python object tored in the pickle file
        in {loc['pickle']}/{folder_name}/{name_data}.json
    """
    
    directory = create_directory(loc[library], folder_name)
    path = f"{directory}/{file_name}.{library}"
    
    with open(path, 'r' + byte_read*'b') as f:
        params = library_from_str[library].load(f)
    
    return params
    
    
retrieve_pickle = partial(_retrieve, library='pickle', byte_read=True)
retrieve_json   = partial(_retrieve, library='json', byte_read=False)


def get_latex_column_table(L):
    """ Return a sub latex column table from a list
    
    Parameters
    ----------
    L : list
    
    Returns
    -------
    latex_table : str
         latex column table
    """
    if isinstance(L, tuple) or isinstance(L, set):
        latex_table = '\\begin{tabular}[c]{@{}l@{}} '
        for i, l in enumerate(L):
            latex_table += l
            if i != len(L) - 1:
                latex_table += ' \\\\ '
        latex_table+= '\\end{tabular}'
    else:
        assert isinstance(L, str), print(f'\n \n {L}')
        latex_table = L
    return latex_table
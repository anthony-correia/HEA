"""
**Serialisation** into JSON and pickle files
"""

import pickle
import json
import joblib
from functools import partial

from HEA.config import loc
from .dir import create_directory

import uncertainties
ufloat_type = uncertainties.core.AffineScalarFunc



library_from_str = {
    'pickle': pickle,
    'json': json,
    'joblib': joblib
}


def _dump(data, file_name=None, folder_name=None,
          library='json', byte_write=False, **params):
    """ Save the data in a file in ``{loc['out']}/{type_data}/``

    Parameters
    ----------
    data      : python object
        element to be saved (can be a list)
        if ``type_data`` is ``'json'``, must be a dictionnary
    file_name : str
        name of the pickle file
    folder_name : str
        name of folder where the json file is
        (if None, there is no folder)
    library   : 'json', 'pickle' or joblib
        library used to save the file
    byte_write: bool
        Write in byte mode
    params  : dict
        parameters passed to the dump function of the corresponding serial library.
    """
    directory = create_directory(loc[library], folder_name)
    path = f"{directory}/{file_name}.{library}"

    with open(path, 'w' + byte_write * 'b') as f:
        if library=='joblib':
            f = f.name
        library_from_str[library].dump(data, f, **params)

    print(f"{library.title()} file saved in {path}")


dump_pickle = partial(_dump, library='pickle', byte_write=True)
dump_json = partial(_dump, library='json', sort_keys=True, indent=4)
dump_joblib = partial(_dump, library='joblib', byte_write=False)


dump_pickle.__doc__ = """Dump a pickle file in ``{loc['pickle']}/`` (in byte mode)

    Parameters
    ----------
    data      : python object
        element to be saved (can be a list)
    file_name : str
        name of the pickle file
    folder_name : str
        name of folder where the pickle file is saved
        (if None, there is no folder)
"""

dump_json__doc__ = """Dump a json file in ``{loc['json']}/``

    Parameters
    ----------
    data      : dict
        element to be saved in a json file
    file_name : str
        name of the pickle file
    folder_name : str
        name of folder where the json file is saved
        (if ``None``, there is no folder)
"""


def _retrieve(file_name, folder_name=None, library='json', byte_read=False):
    """ Retrieve the content of a file

    Parameters
    ----------
    file_name   : str
        name of the file
    folder_name : str
        name of folder where the file is (if ``None``, there is no folder)
    library : str
        ``'json'`` or ``'pickle'`` or ``'joblib``
    byte_read: bool
        Read in byte mode

    Returns
    -------
    Python object or dic
        dictionnary that contains the variables stored in a json file\
        in ``{loc['json']}/{folder_name}/{name_data}.json``\
        or python object stored in a pickle file\
        in ``{loc['pickle']}/{folder_name}/{name_data}.pickle``
    """

    directory = create_directory(loc[library], folder_name)
    path = f"{directory}/{file_name}.{library}"

    with open(path, 'r' + byte_read * 'b') as f:
        if library=='joblib':
            f = f.name
            
        params = library_from_str[library].load(f)

    return params


retrieve_pickle = partial(_retrieve, library='pickle', byte_read=True)
retrieve_json = partial(_retrieve, library='json', byte_read=False)
retrieve_joblib = partial(_retrieve, library='joblib', byte_read=False)

retrieve_pickle.__doc__ = """ Retrieve the content of a pickle file

    Parameters
    ----------
    file_name   : str
        name of the file
    folder_name : str
        name of folder where the pickle file is (if ``None``, there is no folder)


    Returns
    -------
    Python object or dic
        python object stored in a pickle file
        in ``{loc['pickle']}/{folder_name}/{name_data}.pickle``
"""

retrieve_json.__doc__ = """ Retrieve the content of a json file

    Parameters
    ----------
    file_name   : str
        name of the file
    folder_name : str
        name of folder where the json file is (if ``None``, there is no folder)


    Returns
    -------
    dict
        dict stored in the json file
        in ``{loc['json']}/{folder_name}/{name_data}.json``
"""


def get_latex_column_table(L):
    """ Return a sub latex column table from a list

    Parameters
    ----------
    L : list
        List whose each element is a cellule of the sub latex column table

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
        latex_table += '\\end{tabular}'
    else:
        assert isinstance(L, str), print(f'\n \n {L}')
        latex_table = L
    return latex_table


def write_table(table, name, folder_name, show=True, title='line'):
    """ Write a latex table from a table.
    The first line is the title line, separated by
    a double line from the other lines.
    The ufloat numbers are automatically formatted.
    The floated numbers are shown with 3 decimals.
    
    Parameters
    ----------
    table: list of list (2D list)
        table to convert into latex
    name: str
        name of the .tex file to save
    folder_name: str
        folder name where to save the .tex file
    show: bool
        Do we print the .tex content afterwards?
    title: 'l' or 'c'
        Title column or title line is separated from
        the rest of the table by a double line
    """
    
    ## IMPORT  =========================
    from HEA.fit.params import (
        show_latex_table,
        get_str_from_ufloat_mode,
        ufloat_string_into_latex_format
    )
    
    ## PATH ============================
    
    ## LATEX TABLE =====================
    directory = create_directory(loc['tables'], folder_name)
    file_path = f'{directory}/{name}.tex'
    
    with open(file_path, 'w') as f:
        n_column = len(table[0])
        if title=='line':
            f.write('\\begin{tabular}[t]{'+ 'l'*(n_column) +'}')
        else:
            f.write('\\begin{tabular}[t]{l||'+ 'c'*(n_column-1) +'}')
        f.write('\n')
        for i, line in enumerate(table):
            formatted_line = []
            for e in line:
                if isinstance(e, ufloat_type):
                    
                    str_ufloat = get_str_from_ufloat_mode(e, cat='other')
                    latex_ufloat = ufloat_string_into_latex_format(str_ufloat)
                    
                    formatted_line.append(f"${latex_ufloat}$")
                elif isinstance(e, float):
                    formatted_line.append(f"{e:.3f}")
                else:
                    formatted_line.append(str(e))

            f.write("&".join(formatted_line) + "\\\\")
            f.write('\n')
            f.write('\\hline')
            if i==0 and title=='line':
                f.write('\\hline')
            f.write('\n')
                    
        f.write("\\end{tabular}")
    
    if show:
        show_latex_table(name, folder_name, add_name="")
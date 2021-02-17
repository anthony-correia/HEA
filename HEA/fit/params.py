"""
* Retrieve params
* Print the result of a fit as a latex table
* Some formatting function of the dictionnary containing the fitted parameters
"""

from HEA.tools.serial import dump_json, retrieve_json
from HEA.tools.dir import create_directory
from HEA.config import loc
from uncertainties import ufloat


def retrieve_params(file_name, folder_name=None):
    """ Retrieve the parameters saved in a json file

    Parameters
    ----------
    file_name   : str
        name of the file
    folder_name : str
        name of folder where the json file is
        (if None, there is no folder)
    library : str
        `'json'` or `'pickle'`
    byte_read: bool
        Read in byte mode

    Returns
    -------
    Python object or dic
        dictionnary that contains the parameters stored in the json file
        in ``{loc['json']}/{folder_name}/{name_data}_params.json``
    """
    return retrieve_json(file_name=file_name + '_params',
                         folder_name=folder_name)


def get_params_without_BDT(df_params, retrieve_err=False):
    """
    In the keys of the dictionnary, remove the text from ``|`` onwards, except `'_err'`

    Parameters
    ----------
    df_params       : dict
        It is supposed to contain the result of the fit, saved in the json files.
        For instance:

        * ``{'alphaL': value, 'alphaL_err': value ...}``
        * ``{'alphaL|BDT-0.5': value, 'alphaL|BDT-0.5'_err': value ...}``
    retrieve_err    : bool
        if ``True``, include the error of the variables in the file,
        i.e., the variables whose name contains ``'_err'``

    Returns
    -------
    dict
        Formatted dictionnary

    Notes
    -----
    With my notations, for a BDT cut ``BDT > 0.2``, the variables' names ends with ``'|BDT0.2'``.
    Then, this function will the a variable named ``'variable|BDT0.2'`` into a variable named ``'variable'``.
    """

    df_params_formatted = {}

    for key, value in df_params.items():

        is_variable_err = '_err' in key

        if not is_variable_err or retrieve_err:
            if is_variable_err:
                # remove '_err'
                # which will be added back after removing what is after '|'
                new_key = key.replace('_err', '')
            else:
                new_key = key

            index = new_key.find('|')

            if index == -1:
                index = None

            new_key = new_key[:index]

            if is_variable_err:
                new_key += '_err'

            df_params_formatted[new_key] = value

    return df_params_formatted


def get_params_without_err(params):
    """ get the list of variables from the dictionnary of fitted parameters (including errors)

    Parameters
    ----------
    params   : dict
        fitted parameters ``{'alphaL': value, 'alphaL_err': value ...}``

    Returns
    -------
    list
        list of variables (excluding the error variables, whose name contain the ``'_err'`` string.
    """

    keys = list(params.keys())
    variables = []

    for key in keys:
        if not key.endswith('_err'):
            variables.append(key)

    return variables


def json_to_latex_table(name_json, path, latex_params, show=True):
    """ transform a json file that contains the fitted parameters and uncertainties into a latex table
    The latex table is stored into a .tex file in ``{loc['tables']}/{name_json.tex}``

    Parameters
    ----------
    name_json     : str
        name of the json file to load
        also the name of the future .tex file that will be saved
    path          : str
        path of the json file compared to ``loc['json']``, the default folder for json files.
    latex_params   : dict
        alternative name of the parameters (in latex)
    show          : bool
        if True, print the content of the created latex table code
    """

    # Open the json file
    directory = create_directory(loc['json'], path)
    params = retrieve_params(name_json, folder_name=path)
    params = get_params_without_BDT(params, True)

    # Load the variables into ufloats
    ufloats = {}
    for param in params:
        if f'{param}_err' in params:
            ufloats[param] = ufloat(params[param], params[f"{param}_err"])
    # Write the .tex file
    directory = create_directory(loc['tables'], path)
    file_path = f'{directory}/{name_json}_params.tex'
    with open(file_path, 'w') as f:
        f.write('\\begin{tabular}[t]{lc}')
        f.write('\n')
        f.write('\\hline')
        f.write('\n')
        f.write('Variable &Fitted Value\\\\')
        f.write('\n')
        f.write('\\hline\\hline')
        f.write('\n')

        for param, value in ufloats.items():
            formatted_value = f'{value:.2u}'.replace(
                '+/-', '\\pm').replace('e+0', '\\times 10^')
            latex_param = latex_params[param]
            f.write(f"{latex_param}&${formatted_value}$\\\\")
            f.write('\n')
            f.write('\\hline')
            f.write('\n')
        f.write("\\end{tabular}")

    if show:
        show_latex_table(name_json, path)


def show_latex_table(name, path=None):
    """ Print the latex table that contains the result of a fit.
    It prints the content of the tex file in ``{loc['table']}/{path}/{name}_params.tex``

    Parameters
    ----------
    name  : str
        name of the fit which we want to get the latex table with its result
    path  : str
        path of the .tex file from ``loc['tables']``, where the .tex file is

    Notes
    -----
    the latex table must have been already generated, for instance with :py:func:`json_to_latex_table`.
    """
    directory = create_directory(loc['tables'], path)
    file_path = f'{directory}/{name}_params.tex'
    print("Latex table in " + file_path)

    with open(file_path, 'r') as f:
        print(f.read())

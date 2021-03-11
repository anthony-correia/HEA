"""
* Retrieve params
* Print the result of a fit as a latex table
* Some formatting function of the dictionnary containing the fitted parameters
"""

from HEA.tools.serial import dump_json, retrieve_json
from HEA.tools.dir import create_directory
from HEA.config import loc
from uncertainties import ufloat
from HEA.tools.string import _latex_format
from HEA.tools.da import el_to_list


def params_into_dict(result_params, uncertainty=True, remove=None,
                    method='zfit', remove_semicolon=True):
    """
    Parameters
    ----------
    result_params : dict[zfit.zfitParameter, float] or RooFitResult
        Result ``'result.params'`` of the minimisation of the loss function (given by :py:func:`launch_fit`)
    uncertainty   : bool
        do we retrieve the uncertainty (error) on the variable?
    remove        : list[str] or str or None
        if not ``None``, string to be removed from the names of the parameters
    method        : 'zfit' or 'root'
    
    Returns
    -------
    params_dict : dict
        dictionnary of the result of the fit.
        Associates to a fitted parameter (key)
        a dictionnary that contains its value (key: 'v')
        and its error(key: 'e')   
    """
    
    if remove is not None:
        remove = el_to_list(remove)
    
    params_dict = {}
    if method=='zfit':
        for p in list(result_params.keys()):  # loop through the parameters
            # Retrieve name, value and error
            name_param = p.name
            
            if remove_semicolon:
                pos_semicolon = name_param.find(';')
                if pos_semicolon!=-1:
                    new_name_param = name_param[:pos_semicolon]
            else:
                new_name_param = name_param
            if remove is not None:
                for t in remove:
                    new_name_param = new_name_param.replace(t, '')
                    
            params_dict[new_name_param] = {}
            value_param = result_params[p]['value']
            params_dict[new_name_param]['v'] = value_param
            
            if uncertainty:
                error_param = result_params[p]['minuit_hesse']['error']
                params_dict[new_name_param]['e'] = error_param
    
    elif method=='root':
        params_dict = {}
        for p in result_params.floatParsFinal():
            name = p.GetTitle()
    
            if remove is not None:
                for t in remove:
                    name_param = name_param.replace(t, '')

            value = p.getVal()
            err = p.getError()
            params_dict[name] = {
                'v': value,
                'e': err
                }
    return params_dict
            
def save_params(param_results, name_file,
                dic_add=None, folder_name=None, 
                method='zfit', remove=None):
    """ Save the parameters of the fit in ``{loc['json']}/{name_file}_params.json``

    Parameters
    ----------
    param_results : 
        dictionnary of the result of the fit.
        Associates to a fitted parameter (key)
        a dictionnary that contains its value (key: 'v')
        and its error(key: 'e')
    name_file     : str
        name of the file that will be saved
    uncertainty   : bool
        if True, save also the uncertainties (variables that contain '_err' in their name)
    dic_add       : dict or None
        other parameters to be saved in the json file
    folder_name   : str
        name of the folder
    remove        : list[str] or str or None
        if not ``None``, string to be removed from the names of the parameters
    """
    

    if remove is not None:
        remove = el_to_list(remove)
        params_result_r = {}
        for key, value in param_results.items():
            for t in remove:
                name_param = key.replace(t, '')
            
            params_result_r[name_param] = value
    
    if dic_add is not None:
        for key, value in dic_add.items():
            param_results[key] = value
            
    dump_json(param_results, name_file + '_params', folder_name=folder_name)


def retrieve_params(file_name, folder_name=None, only_val=True):
    """ Retrieve the parameters saved in a json file

    Parameters
    ----------
    file_name   : str
        name of the file
    folder_name : str
        name of folder where the json file is
        (if None, there is no folder)
    only_val: bool
        if True, import only the variables with their values,
        and not their errors

    Returns
    -------
    Python object or dic
        dictionnary that contains the parameters stored in the json file
        in ``{loc['json']}/{folder_name}/{name_data}_params.json``
    """
    params =  retrieve_json(file_name=file_name + '_params',
                            folder_name=folder_name)
    
    if only_val:
        params_vals = {}
        for name_param in params.keys():
            if 'v' in  params[name_param] and 'e' in  params[name_param]:
                params_vals[name_param] = params[name_param]['v']
        
        return params_vals
    else:
        return params
    
    


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
    """ get the list of variables from the dictionnary of fitted parameters

    Parameters
    ----------
    params   : dict
        fitted parameters ``{'alphaL': value, 'alphaL_err': value ...}``

    Returns
    -------
    list
        list of variables (excluding the error variables, whose name contain the ``'_err'`` string.
    """

    return params.keys()

def get_str_from_ufloat_mode(ufloat_number, cat='other'):
    """ Get the string format of the ufloat, 
    where the number of significative figures depends on the specified``cat``(egory).
    
    Parameters
    ----------
    ufloat_number : uncertainties.ufloat
        Parameter value
    cat : 'main', 'other' or 'yield'
        Category that specifies the significant figures of the nominal and error values
        
        * ``'main'`` (for :math:`\\mu` and :math:`\\sigma`): the error has 2 significant figures
        * ``'yield'``: the nominal value is an integer
        * ``'other'`` (default): the error has 2 significant figures
    
    Returns
    -------
    ufloat_string: str
        ufloat number with the good number of significant figures, with:
        
        * ``'+/-'`` for :math:`\\pm`
        * ``'e{n}'``  for :math:`\\times 10^{n}`
    """
    
    possible_cats = ['main', 'yield', 'other']
    
    if cat=='other':
        ufloat_string = f"{ufloat_number:.2ue}"
    elif cat=='main':
        ufloat_string = f"{ufloat_number:.1u}"
    elif cat=='yield':
        ufloat_string = f"{ufloat_number:.0f}"
    else:
        raise AssertionError(f'cat = {cat} should belong to {possible_cats}')
    return ufloat_string.replace('e+0', 'e').replace('e-0', 'e-')
    
def ufloat_string_into_latex_format(ufloat_string):
    """ Turn a raw ufloat string into an ufloat string in latex format:
    
    * ``'+/-'`` turned into ``'\\pm'``
    * ``'e...'`` turned into ``'\times 10^{...}'``
    
    Parameters
    ----------
    ufloat_string: str
        ufloat written in string, output of :py:func:`get_str_from_ufloat_mode`
    
    Returns
    -------
    ufloat_latex: str
        ufloat in latex format, without the ``$``.
            
    """
    ufloat_latex = ufloat_string.replace('+/-', '\\pm')
    
    if 'e' in ufloat_latex:
        if ufloat_latex.endswith(')e0'):
            ufloat_latex = ufloat_latex.replace(')e0', '').replace('(', '')
        else:
            ufloat_latex = ufloat_latex.replace('e', '\\times 10^{') + '}'
    
    return ufloat_latex


def get_latex_cat_from_param_latex_params(param, latex_params=None):
    """ Extract the latex name and category of ``param`` in ``latex_params``
    
    Parameters
    ----------
    param: str
        name of the zfit parameter
    latex_params   : dict[str:str] or dict[str:dict]
        alternative name of the parameters (in latex). 2 possible forms
        
        * key = name of the variable and value = latex name of the variable
        * key = name of the variable and value = dictionnary with 2 keys
            
            * ``'latex'``: latex name of the variable
            * ``'cat'``: passed to :py:func:`get_str_from_ufloat_mode`
        
        In the first case, ``'cat'`` is not specified and is set at ``'other'`` by default.
        
    Returns
    -------
    latex: str
        latex name of ``param``
    cat: str
        category of the parameter (as specified in :py:func:`get_str_from_ufloat_mode`
    
    Returns
    -------
    ufloat_latex: str
        ufloat in latex format, without the ``$``.
    """
    if latex_params is None:
        return _latex_format(param), 'other'
    
    assert param in latex_params, f"{param} is not in the dictionnary latex_params={latex_params}"
    
    if isinstance(latex_params[param], dict):
        latex = latex_params[param]['latex']
        cat = latex_params[param]['cat']
    else:
        latex = latex_params[param]['latex']
        cat = 'other'
    
    return latex, cat

def get_ufloat_latex_from_param_latex_params(param, ufloat_value, latex_params=None):
    """ Extract the latex name and category of ``param`` in ``latex_params``,
    turn it into the latex format of the value of the param.
    
    Parameters
    ----------
    param: str
        name of the zfit parameter
    ufloat_value: uncertainties.ufloat
        value of the parameter
    latex_params   : dict[str:str] or dict[str:dict]
        alternative name of the parameters (in latex). 2 possible forms
        
        * key = name of the variable and value = latex name of the variable
        * key = name of the variable and value = dictionnary with 2 keys
            
            * ``'latex'``: latex name of the variable
            * ``'cat'``: passed to :py:func:`get_str_from_ufloat_mode`
        
        In the first case, ``'cat'`` is not specified and is set at ``'other'`` by default.
    
    Returns
    -------
    latex_param: str
        latex name of ``param``
    ufloat_latex: str
        ufloat in latex format, without the ``$``.
    """
    
    latex_param, cat = get_latex_cat_from_param_latex_params(param, latex_params)
    ufloat_str = get_str_from_ufloat_mode(ufloat_value, cat=cat)

    ufloat_latex = ufloat_string_into_latex_format(ufloat_str)
    
    return latex_param, ufloat_latex

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
    latex_params   : dict[str:str] or dict[str:dict]
        passed to :py:func:`get_latex_cat_from_param_latex_params`
        
    show          : bool
        if True, print the content of the created latex table code
    """

    # Open the json file
    directory = create_directory(loc['json'], path)
    params = retrieve_params(name_json, folder_name=path, only_val=False)
    params = get_params_without_BDT(params, True)

    # Load the variables into ufloats
    ufloats = {}
    for param in params:
        if 'v' in params[param] and 'e' in params[param]:
            ufloats[param] = ufloat(params[param]['v'], params[param]['e'])
    
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
        
        for param in latex_params.keys():
            if param in ufloats.keys():
                value = ufloats[param]
                latex_param, ufloat_latex = get_ufloat_latex_from_param_latex_params(
                    param, 
                    value, 
                    latex_params
                )

                f.write(f"{latex_param}&${ufloat_latex}$\\\\")
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

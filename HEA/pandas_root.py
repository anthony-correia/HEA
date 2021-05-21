"""
Handle root files with Pandas

* Load a root file into a Pandas dataframe
* Apply cuts the a Pandas dataframe
* Save a Pandas dataframe into a root file
* Get the needed branches to load from a root file
* Act on the branches of a dataframe with functions
"""

import pandas as pd
from root_pandas import read_root

from HEA.tools.da import el_to_list
from HEA.tools.dir import create_directory, try_makedirs
from HEA.config import loc
from HEA.definition import definition_functions

from HEA import RVariable
from HEA.tools.serial import dump_json
from HEA.tools.string import add_text

import uproot

import os.path as op

def assert_needed_variables(needed_variables, df):
    """ assert that all needed variables are in a dataframe

    Parameters
    ----------
    needed_variables: str or list of str
        list of the variables that will be checked to be in a pandas dataframe
    df: pandas.Dataframe
     Dataframe that should contain all the needed variables

    Returns
    -------
    bool
        True if all the variables in ``needed_variables`` are in the dataframe ``df``

    """
    needed_variables = el_to_list(needed_variables, 1)

    for needed_variable in needed_variables:
        assert needed_variable in df, f"{needed_variable} is not in the pandas dataframe"

def perform_cut(df, cut, *args, verbose=True, **kwargs):
    """ Perform a cut on a Pandas dataframe and print the number of cut-out events.
    Save the cut result in a json file.

    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe that will be cut
    cut : str
        Performed cut
    verbose: Bool
        if True, print the number of cut-out events
    """

    n_events = len(df)
    df = df.query(cut)
    n_cut_events = len(df)

    if verbose:
        print(f"{cut} cut has removed {n_events - n_cut_events} out of {n_events} events")
        
    save_result_cut(cut, n_events, n_cut_events, 
                    *args, **kwargs)
    
    return df

def save_result_cut(cut, before, after, num_cut=None,
                    data_name=None, folder_name=None):
    """ Save the result of the cut in a json file in 
    ``{folder_name}/{data_name}/{num_cut}.json``
    
    Parameters
    ----------
    cut: str
        performed cut
    before: int
        number of events before the cut
    after : int
        number of events after the cut
    num_cut : int
        Reference of the cut (just a number
        to tell the difference between the other cuts
        and for the name of the saved json file)
    data_name : str
        name of the data, last folder where the json file is saved
    folder_name : str
        folder name. The json file will be saved in the 
        ``folder_name/data_name/`` directory
    """
    
    
    if data_name is not None and num_cut is not None:
        result_cut = {}
        result_cut['cut'] = cut
        result_cut['before'] = before
        result_cut['after'] = after
        path = add_text('cuts', folder_name, '/')
        path = add_text(path, data_name, '/')

        dump_json(result_cut, file_name=str(num_cut), 
                  folder_name=path)

    
    
    
    
def drop_na(df, subset, *args, **kwargs):
    """ Drop NaN values of a dataframe and save the result
     in a json file.
     
     Parameters
     ----------
     df : pandas.DataFrame
        Dataframe that will be cut
     subset: array-like
         columns which NaN values are dropped from
     num_cut : int
        Reference of the cut (just a number
        to tell the difference between the other cuts
        and for the name of the saved json file)
     *args, **kwargs:
         passed to :py:func:`save_result_cut`
     """
    
    n_events=len(df)
    df = df.dropna(subset=subset)
    n_events_dropna = len(df)
    print(f"Dropping NaN has removed {n_events - n_events_dropna} out of {n_events} events")
    save_result_cut("drop nan", n_events, n_events_dropna, 
                    *args, **kwargs)   


def load_dataframe(
    paths, columns=None, tree_name='DecayTree', method='read_root', 
    verbose=True,
    **kwds):
    """ load dataframe from a root file (also print the path of the root file)

    Parameters
    ----------
    paths     : str or list(str)
        location of the root file
    tree_name : str
        name of the tree where the data to be loaded is
    columns   : list or None
        columns of the root files that are loaded.
        If ``None``, load everything
    method    : str
        method used to load the data: ``'read_root'`` or ``'uproot'``
    **kwds    : dict
        parameters passed to the function to read the root file 
        (e.g., ``read_root``)
    
    Returns
    -------
    pandas.DataFrame
        loaded pandas dataframe
    """
    paths = el_to_list(paths)
    df = pd.DataFrame() 
    for path in paths:
        if verbose:
            print("Loading " + path)

        if method == 'read_root':
            df = df.append(read_root(path, tree_name, columns=columns, **kwds))

        elif method == 'uproot':
            
            file = uproot.open(path)[tree_name]
            df = df.append(file.arrays(library="pd", how="zip", filter_name=columns))
            # df = df.append(file.arrays(vars, library="pd"))
            del file

    return df

def load_dataframe_YY_MM(path, YY=None, MM=None, **kwds):
    """ Load the dataframe given by path 
    for the years and polarisation given by ``YY`` and ``M``
    
    Parameters
    ----------
    
    path   : str
        path with ``'{MM}'``, ``'{YY}'`` 
        to represent where to fill in the year (``YY``) and polarisation (``MM``)
    YY     : int or list(int)
        list of two last figures of the year to load
    MM     : "up" or "down" or list("up" and "down")
        list of polarisation to load
    **kwds : dict
        other parameters to pass to :py:func:`load_dataframe`
    """
    YY = el_to_list(YY)
    MM = el_to_list(MM)
    
    paths = list({path.format(YY=yy, MM=mm) for yy in YY for mm in MM})
    
    return load_dataframe(paths, **kwds)
            
                
def load_saved_root(name_data, columns=None, folder_name="",
                    tree_name='DecayTree', cut_BDT=None, method='read_root'):
    """

    Parameters
    ----------
    name_data  : str, name of the root file
    vars       : list of str,
        list of the desired variables
    method     : str,
        method to retrieve the data (``'read_root'`` or ``'uproot'``) (read_root is faster)
    cut_BDT    : str or float or None
        if not ``None``, the root file with the BDT cut ``BDT > cut_BDT`` should have a name finishing by ``"_BDT{cutBDT}"``.

    Returns
    -------
    pandas.Dataframe
        loaded, with the desired variables
    """

    text_cut_BDT = "" if cut_BDT is None else f'_BDT{cut_BDT}'

    complete_path = f"{loc['out']}/root/{folder_name}/{name_data}{text_cut_BDT}.root"

    return load_dataframe(complete_path, tree_name=tree_name, columns=columns, method=method)


def save_root(df, file_name, name_key, folder_name=None, path=None):
    """ save the dataframe in a .root file

    Parameters
    ----------
    df        : pandas.DataFrame
        dataframe to save
    file_name : str
        name of the root file that will be saved
    name_key  : str
        name of the tree where the file will be saved

    """
    path = loc['out'] + 'root/'
    path = create_directory(path, folder_name)
    path += f"/{file_name}.root"
    
    print(f"Root file saved in {path}")
    df.to_root(path, key=name_key)


def get_dataframe_from_raw_branches_functions(
        df, raw_branches_functions, mode='new', functions=definition_functions):
    """ From a list of variables with possibly the functions that will be applied to the variable afterwards, add the variables which a function is applied to, to a pandas dataframe.

    Parameters
    ----------
    raw_branches_functions :  list
        list of

            - variable
            - tuple ``(variable, function)``, where ``function`` is the name of the function applied to the variable
            - tuple ``(variables, function)``, where ``variables`` is a tuple of variables, inputs of the function

    df                  : Pandas dataframe
        original dataframe
    mode                : str
        3 modes:

                - 'add': add the variables to the dataframe (in place)
                - 'new': create a new dataframe with the new variables only
                - 'both' : do both

    Returns
    -------
    pandas.Dataframe
        Dataframe with the variables specified in ``raw_branches_functions``
    """

    new_df_required = (mode == 'new' or mode == 'both')

    if new_df_required:
        new_df = pd.DataFrame()

    for raw_branch_function in raw_branches_functions:
        # Retrieve the name of the variable and the function applied to it
        if isinstance(raw_branch_function, tuple):
            raw_branch = raw_branch_function[0]
            name_function = raw_branch_function[1]
        else:
            raw_branch = raw_branch_function
            name_function = None

            assert raw_branch in df, f"The branch {raw_branch} is not in the Pandas dataframe"

        if name_function is None and new_df_required:
            new_df[raw_branch] = df[raw_branch].values

        if name_function is not None:
            new_branch = RVariable.get_branch_from_raw_branch_name_function(
                raw_branch, name_function)

            if isinstance(raw_branch, tuple) or isinstance(raw_branch, list):
                data = tuple([df[var] for var in raw_branch])
            else:
                data = df[raw_branch]

            new_data = functions[name_function](data).values

            if new_df_required:
                new_df[new_branch] = new_data

            if mode == 'add' or mode == 'both':
                df[new_branch] = new_data

    if new_df_required:
        return new_df

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
from HEA.tools.dir import create_directory
from HEA.config import loc
from HEA.definition import definition_functions

from HEA import RVariable


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


def show_cut(df, cut, verbose=True):
    """ Perform a cut on a Pandas dataframe and print the number of cut-out events

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
        print(f"{cut} cut has removed {n_events - n_cut_events} over {n_events} events")

    return df


def load_dataframe(path, tree_name, columns, method='read_root'):
    """ load dataframe from a root file (also print the path of the root file)

    Parameters
    ----------
    path      : str
        location of the root file
    tree_name : str
        name of the tree where the data to be loaded is
    columns   : list
        columns of the root files that are loaded
    method    : str
        method used to load the data: ``'read_root'`` or ``'uproot'``

    Returns
    -------
    pandas.DataFrame
        loaded pandas dataframe
    """

    print("Loading " + path)

    if method == 'read_root':
        return read_root(path, tree_name, columns=columns)

    elif method == 'uproot':
        import uproot4  # not imported by default...
        file = uproot4.open(path)[tree_name]
        df = file.arrays(vars, library="pd")
        del file

        return df


def load_saved_root(name_data, vars=None, folder_name="",
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

    return load_dataframe(complete_path, tree_name, vars, method=method)


def save_root(df, file_name, name_key, folder_name=None):
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

"""
General tool functions used with ROOT
"""

import pandas as pd
from ROOT import TChain

from HEA.tools.da import el_to_list

def load_tree(paths, branches, cuts=None,
              verbose=True):
    """ Load and cut on a tree of data
    
    Parameters
    ----------
    paths: str or list(str)
        path(s) of the root files
    branches: str or list(str)
        branch(es) to load
    verbose: bool
        if ``True``, plot the number of events
        in the tree
    cut : str
        cut to apply to the varaibles
    
    Returns
    -------
    tree : TChain
        Data
    """
    
    paths = el_to_list(paths)
    branches = el_to_list(branches)
    
    # Create the tree
    tree = TChain("DecayTree")
    for path in paths:
        tree.Add(path)
    
    # Only activate the interesting variables
    tree.SetBranchStatus("*", 0)
    
    for branch in branches:
        print(f"Activate the branch {branch}")
        tree.SetBranchStatus(branch, 1)
    
    # Cuts
    if cuts is not None:
        if verbose:
            print("Performed cuts:")
            print(cuts)
        before = tree.GetEntries()
        tree = tree.CopyTree(cuts)
        after = tree.GetEntries()
        print("Efficiency of the cuts:", (before - after)/before)
    
    if verbose:
        print("Events in tree: %s" % tree.GetEntries())
    
    return tree
    
def tree_to_df(tree, branch):
    """ Converts a branch of a tree into an
    dataframe. Useful for plotting.
    
    Parameters
    ----------
    tree: TTree
        tree were the data is
    branch: str
        branch to plot
        
    Returns
    -------
    df : pd.DataFrame
        dataframe containing the ``branch``
    """

    array = tree.AsMatrix(columns=[branch])
    df = pd.DataFrame()
    df[branch] = array.ravel()
    
    return df
"""
General tool functions used with ROOT
"""

import pandas as pd
import os.path as op
from ROOT import TChain, TFile, RooWorkspace

from HEA.tools.da import el_to_list
from HEA.tools.dir import try_makedirs, create_directory
from HEA.config import loc

def load_tree(paths, branches, tree_name='DecayTree',
              cuts=None,
              verbose=True):
    """ Load and cut on a tree of data
    
    Parameters
    ----------
    paths: str or list(str)
        path(s) of the root files
    branches: str or list(str)
        branch(es) to load
    tree_name: str
        name of the tree to load
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
    tree = TChain(tree_name)
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

def save_tree_root(data, file_name, path=None):
    """ Save a rooDataset
    
    Parameters
    ----------
    data: RooDataSet
        dataset to save
    file_name: str
        name of the root file
    path : str
        directory where to save the root file
    """
    try_makedirs(path) 
    out_tree = data.GetClonedTree()
    full_path = op.join(path, file_name)
    out_file = TFile(full_path, "RECREATE")
    out_tree.Write()
    out_file.Write()
    out_file.Close()
    
    print(f"Root file saved in {full_path}")
    
def save_workspace(w, file_name, folder_name=None):
    """ Save a workspace in a root file.
    
    Parameters
    ----------
    w: RooWorkspace
        workspace to save
    file_name: str
        name of the root file to save
    folder_name: str
        name of the folderwhere to save the workspace
    """
    
    path = op.join(loc['out'], 'root/')
    path = create_directory(path, folder_name) 
    full_path = op.join(path, f"{file_name}_ws.root")
    w.writeToFile(full_path)
    
    print(f"Worspace save in {full_path}")
    
def load_workspace(file_name, folder_name=None):
    """ Load a workspace from a root file.
    
    Parameters
    ----------
    file_name: str
        name of the root file to save
    folder_name: str
        name of the folderwhere to save the workspace
    
    Returns
    -------
    w: RooWorkspace
        workspace that is saved in the root file
    """
    path = op.join(loc['out'], 'root/')
    path = create_directory(path, folder_name) 
    full_path = op.join(path, f"{file_name}_ws.root")
    
    file = TFile(full_path)
        
    w = file.Get('w')
    file.Close()
    return w
    
    
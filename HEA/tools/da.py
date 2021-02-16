"""
**Data handling**: function to manipulate lists and dictionnary

* Transform an element into a list
* Check if a list is included in another
* Add a key-value couple to a dictionnary if the key is not in the dictionnary
- ...
"""

from HEA.tools import assertion

def el_to_list(el, len_list):
    """ Turn a single non-list element ``el`` into a list of size ``len_list``
    
    Parameters
    ----------
    el : python object
        element
    len_list : int
        length of the list to create
    
    Returns
    -------
    transformed_el : python object
        If ``el`` is a ``list``, ``el``. Else, list of size ``len_list``: ``[el, el, ...]``
    
    Notes
    -----
    A function taking two variables might have arguments that need to be a 2-list, whose elements correspond to the two variables. This function allows for the user to specify only a common element that will be converted into a 2-list.    
    """
    if not assertion.is_list_tuple(el):
        return [el for i in range(len_list)]
    else:
        return el

def list_included(L1, L2):
    """ Check whether the list ``L1`` included in the list ``L2``
    
    Parameters
    ----------
    L1: list
        potential sub-list
    L2: list
        potential larger list
    
    Return
    ------
    L1_included_L2: bool
        ``True`` if ``L1`` included in ``L2``, else ``False``
    """
        
    for l in L1:
        if l not in L2:
            return False
    return True


def add_in_dic(key, dic, default=None):
    """ if a ``key`` is not in the dictionnary ``dic``, add it with value specified by ``default``. In place.
       
    Parameters
    ----------
    key : immutable python object
        new key of the dictionnary
    dic : dict
        dictionary that might contain the ``key``
    default : python object
        default value of the ``key`` is not in the ``dic``
    """
    if key not in dic:
        dic[key] = default

def show_dictionnary(dic, name_dic=None):
    """ Print the dictionnary elements with line break
    
    Parameters
    ----------
    dic: dict
        dictionnary to show
    name_dic: str or None
        name of the dictionnary    
    """
    
    if name_dic is not None:
        print(f"{name_dic}:")
        
    for key, value in dic.items():
        print(f"{key}: {value}")

def flatten_2Dlist(L2):
    """ Flatten a n-dimensional list into a (n-1)-dimensional list (if it is indeed a list of dimension greater than or equal to 2)
    
    Parameters
    ----------
    L2 : python object
    
    Returns
    -------
    python object or 2D list
        Flatten (n-1)-dimensional list
    """
    if not isinstance(L2,list):
        return L2
    elif not isinstance(L2[0],list):
        return L2
    else:
        return [el for L in L2 for el in L]

def get_element_list(L, index):
    """ Return an element of a list/tuple.
    If ``L`` is not a list, returns it directly.
    
    Parameters
    ----------
    L      : python object
        potential list
    index  : int
        index of the list to get
    
    Returns
    -------
    python object
        If ``list_candidate`` is a list its element if index ``index``.
        Else, returns ``list_candidate``
    """
    if assertion.is_list_tuple(L):
        return L[index]
    else:
        return L
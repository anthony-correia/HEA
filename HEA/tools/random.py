"""
Functions used to generate random stuff.
"""

import numpy as np

def systematics_variation_params(
    params, 
    list_params=None, require_same_sign=False,
    seed=15):
    """ Get the parameters under systematics variation.

    Parameters
    ----------
    params: dict
        associate a param name with another dict,
        that indicates its value (``'v'``) and error (``'e'``).
    list_params: dict
        Parameters to vary under systematics variation.
        If not given, all the variables in
        ``params`` are varied.
    require_same_sign: bool
        if ``True``, requires that the varied value as the same
        sign as the nominal value.
    seed: int
        Seed to generate randomly the updated number
        (for reproducible randomness)

    returns
    -------
    systematics_params: dict
        Associates the name of a parameter to vary
        with its new value, found under systematics variation.
    """

    if list_params is None:
        list_params = list(params.keys())
    
    np.random.seed(seed)
    systematics_params = {}

    for param_name in list_params:
        
        nom_value = params[param_name]['v']
        width     = params[param_name]['e']

        new_value = np.random.normal(
            loc=nom_value,
            scale=width
        )

        if require_same_sign:
            while np.sign(new_value)!=np.sign(nom_value):
                new_value = np.random.normal(
                    loc=nom_value,
                    scale=width
                )

        systematics_params[param_name] = new_value
        print(
            f"Systematic variation of {param_name}:",
            f"{nom_value} +/- {width} -> {new_value}"
        ) 
    return systematics_params


def blind_params(
    params, list_blinded_params, seed=30,
    width=0.1, unblind=False):
    """ Blind specified parameters by
    multiplying each of them by a 
    random number generated using a gaussian
    of width ``width`` and centre 1. Each
    parameter get its own random number.

    Parameters
    ----------
    params: dict[str:float] or dict[str:dict]
        Associates a parameter name with
        its value 
        OR
        a dictionnary, that gives its
        value (key ``'v'``) and
        uncertainty (key ``'e'``)
    list_blinded_params: list(str)
        list of the parameters to blind
    seed: int
        Used for reproducible randomness
    width: float
        width of the gaussian to generate the
        random numbers.
    unblind: bool
        Do we unblind? default: we blind!
    
    Returns
    -------
    blinded_paramers: dict[str:dict]
        Same as ``params``, except that
        some the parameters listed in 
        ``list_blinded_params``
        have their values blinded
    """

    if seed is not None:
        np.random.seed(seed)
    
    for blinded_param in list_blinded_params:
        if blinded_param in params:
            print(f"Blinding of {blinded_param}")
            rd_nb = np.random.normal(loc=1., scale=width)
            while rd_nb < 0:
                rd_nb = np.random.normal(loc=1., scale=width)
            if 'v' in params[blinded_param] and 'e' in params[blinded_param]:
                if not unblind:
                    params[blinded_param]['v'] *= rd_nb
                    params[blinded_param]['e'] *= rd_nb
                else:
                    params[blinded_param]['v'] /= rd_nb
                    params[blinded_param]['e'] /= rd_nb
            else:
                if not unblind:
                    params[blinded_param] *= rd_nb
                else: 
                    params[blinded_param] /= rd_nb

    return params


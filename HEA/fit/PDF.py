"""
* Get the number of d.o.f. of a model
* Get the :math:`\\chi^2` of a model
* Get the mean/std of an array
"""

import numpy as np



##########################################################################
################################# number of d.o.f. of a model ############
##########################################################################


def get_n_dof_params_recurs_old(params, params_seen=[], n_dof_previous=0):
    """ Recursive function to get the number of d.o.f. of a parameter

    Parameters
    ----------
    params          : zfit.pdf.Parameter or zfit.pdf.ComposedParameter
        Parameter
    params_seen    : list(str)
        list of the names of the parameters that have been seen so far
    n_dof_previous : int
        number of d.o.f. counted so far

    Returns
    -------
    n_dof: int
        number of d.o.f. contained in the list of parameters that aren't in the list of
        parameters already seen, to which is added the number of d.o.f. already seen previously.
    params_seen: list(str)
        list of the names of the parameters that have been seen so far

    """
    n_dof = 0

    for param in params.values():

        name_param = param.name
        if name_param not in params_seen:
            params_seen.append(name_param)
            # if it is a composed parameter, check to composite parameters
            if isinstance(param, ComposedParameter):
                n_dof, params_seen = get_n_dof_params_recurs_old(
                    param.params, params_seen, n_dof)
            else:
                if param.floating:
                    n_dof += 1
    return n_dof + n_dof_previous, params_seen


def __get_n_dof_model_recurs_old(model, params_seen=[], n_dof_previous=0):
    """ Recurvive function to get the number of d.o.f. of a model.
    A d.o.f. corresponds to a floated parameter in the model.

    Parameters
    ----------
    model: zfit.pdf.BasePDF
        Model (PDF)
    params_seen    : list(str)
        list of the names of the parameters that have been seen so far
    n_dof_previous : int
        number of d.o.f. counted so far
    Returns
    -------
    n_dof: int
        number of d.o.f. contained in the model, corresponding to parameters that aren't in the list of
        parameters already seen, to which is added the number of d.o.f. already seen previously.
    params_seen: list(str)
        list of the names of the parameters that have been seen so far
    """
    n_dof = 0

    # Explore the parameters of model
    n_dof, params_seen = get_n_dof_params_recurs_old(
        model.params, params_seen, n_dof)

    # Explore the parameters of the submodels of model
    if isinstance(model, SumPDF):
        for submodel in model.models:
            n_dof, params_seen = __get_n_dof_model_recurs_old(
                submodel, params_seen, n_dof)

    return n_dof + n_dof_previous, params_seen


def get_n_dof_model_old(model):
    """ Get the number of d.o.f. of a zfit model.
    A d.o.f. corresponds to a floated parameter in the model.

    Parameters
    ----------
    model: zfit.pdf.BasePDF
        Model (PDF)

    Returns
    -------
    n_dof: int
        number of d.o.f. in the model
    """
    from zfit.core.parameter import ComposedParameter
    from zfit.models.functor import SumPDF
    
    n_dof, _ = __get_n_dof_model_recurs_old(model, params_seen=[])
    return n_dof

def get_n_dof_model(model):
    """ Get the number of d.o.f. of a zfit model. (new version!)
    A d.o.f. corresponds to a floated parameter in the model.

    Parameters
    ----------
    model: zfit.pdf.BasePDF
        Model (PDF)

    Returns
    -------
    n_dof: int
        number of d.o.f. in the model
    """
    n_dof = 0
    for param in model.get_params(): # loop over all the parameters
        n_dof += param.floating
    
    return n_dof
        
def get_n_dof_model(model):
    """ Get the number of d.o.f. of a zfit model. (new version!)
    A d.o.f. corresponds to a floated parameter in the model.

    Parameters
    ----------
    model: zfit.pdf.BasePDF
        Model (PDF)

    Returns
    -------
    n_dof: int
        number of d.o.f. in the model
    """
    n_dof = 0
    for param in model.get_params(): # loop over all the parameters
        n_dof += param.floating
    
    return n_dof    


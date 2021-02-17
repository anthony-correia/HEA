"""
* Define some PDFs
* Define the parameters from a dictionnary of training variables
* Perform a fit (using zfit)
* Check if the fit has an error (if it did not converge, has not a positive-definite hesse, ...)
"""

from HEA.tools.dir import create_directory
from HEA.tools.da import el_to_list, add_in_dic
from HEA.tools.serial import dump_json

import zfit

##########################################################################
############################################# PDFs #######################
##########################################################################


def crystall_ball_or_gaussian(mu, sigma, obs, alpha=None, n=None):
    """ return a Crystall Ball (CB) or gaussian PDF

    * Gaussian if ``alpha`` or ``n`` is ``None``
    * CB if none of them are ``None``

    Parameters
    ----------
    mu : zfit.Parameter
        mean
    sigma: zfit.Parameter
        standard deviation
    obs: zfit.Space
        Space
    alpha: zfit.Parameter
        :math:`\\alpha` parameter of the tail
    n: zfit.Parameter
        :math:`n` parameter of the tail

    Returns
    -------
    zfit.PDF.Gauss or zfit.PDF.CrystallBall
        PDF
    """
    if alpha is None or n is None:
        pdf = zfit.pdf.Gauss(mu, sigma, obs=obs)
    else:
        pdf = zfit.pdf.CrystalBall(mu, sigma, alpha, n, obs=obs)
    return pdf


def sum_crystalball_or_gaussian(
        muL, muR, sigmaL, sigmaR, frac, obs, alphaL=None, alphaR=None, nL=None, nR=None):
    """ Return the sum of 2 crystall ball PDFs with tails in opposite side (left or right) or Gaussian PDFs.
    If the ``alpha`` or ``n`` is ``None``, the corresponding distribution (left or right) is a gaussian.

    Parameters
    ----------
    muL : zfit.Parameter
        mean of the left distribution
    muR : zfit.Parameter
        mean of the right distribution
    sigmaL: zfit.Parameter
        standard deviation of the left distribution
    sigmaR: zfit.Parameter
        standard deviation of the right distribution
    obs: zfit.Space
        Space
    alphaL: zfit.Parameter
        :math:`\\alpha_L` parameter of the left tail of the left distribution
    alphaR: zfit.Parameter
        :math:`\\alpha_R` parameter of the left tail of the right distribution
    nL: zfit.Parameter
        :math:`n_L` parameter of the left tail of the left distribution
    nL: zfit.Parameter
        :math:`n_R` parameter of the right tail of the right distribution

    Returns
    -------
    zfit.PDF.SumPDF
        PDF sum of 2 crystall ball or Gaussian PDFs.

    """
    pdfL = crystall_ball_or_gaussian(muL, sigmaL, obs,
                                     alphaL, nL)
    pdfR = crystall_ball_or_gaussian(muR, sigmaR, obs,
                                     alphaR, nR)

    model = zfit.pdf.SumPDF([pdfL, pdfR], fracs=frac)

    return model, pdfL, pdfR

##########################################################################
########################################## Parameters ####################
##########################################################################


def define_zparams(initial_values, cut_BDT=None, num=None):
    """Define zparams from the dictionnary initial_values

    Parameters
    ----------

    initial_values : dict
        {"name_variable": {"value":, "low":, "high":, "floating":}}
    cut_BDT        : float
        performed cut on the BDT (BDT > cutBDT)
    num            : integer
        Index of the fit. add ``";{num}"`` at the end of the variable/
        the other functions I wrote allow to ignore the ``";{num}"`` in the name of the variable. This is used manely in order to define a parameter several times (when tuning their values to make the fit convergent)

    Returns
    -------
    zparams        : dict[str, zfit.Parameter]
        Dictionnary of zfit Parameters whose keys are the name of the variables and:

        * if cut_BDT is None, the key is just the name of the variable
        * else, the key is ``"{name_variable}|BDT{cut_BDT}"``
    """
    zparams = {}
    for var in initial_values.keys():
        if cut_BDT is not None:
            name_var = f"{var}|BDT{cut_BDT}"
        else:
            name_var = var
        if num is not None:
            name_var += f';{num}'

        init = initial_values[var]
        add_in_dic('value', init, default=None)
        add_in_dic('low', init, default=None)
        add_in_dic('high', init, default=None)
        add_in_dic('floating', init, default=True)

        zparams[var] = zfit.Parameter(name_var, init['value'], init['low'], init['high'],
                                      floating=init['floating'])

    return zparams


def check_fit(result):
    """ Check if the fit has gone well. If a check has gone wrong, point it out.

    Parameters
    ----------
    result  : zfit.minimize.FitResult
        result of the minimisation of the likelihood

    Returns
    -------
    Bool
        ``False`` if something has gone wrong in the fit.
        ``True`` if the fit has gone smoothly.

    """
    fit_ok = True
    info_fit = result.info['original']

    checks = {}
    checks['is_valid'] = True
    checks['has_valid_parameters'] = True
    checks['has_accurate_covar'] = True
    checks['has_posdef_covar'] = True
    checks['has_made_posdef_covar'] = False
    checks['hesse_failed'] = False
    checks['has_covariance'] = True
    checks['is_above_max_edm'] = False
    checks['has_reached_call_limit'] = False

    for check, desired in checks.items():
        if info_fit[check] != desired:
            print(f'Problem: {check} is ' + str(info_fit[check]))
            fit_ok = False

    edm = info_fit['edm']
    if edm > 0.001:
        print(f'edm = {edm} > 0.001')
        fit_ok = False

    if result.params_at_limit:
        print(f'param at limit')
        fit_ok = False

    return fit_ok


def save_params(params, name_file, uncertainty=True,
                dic_add=None, folder_name=None, remove=None):
    """ Save the parameters of the fit in ``{loc['json']}/{name_file}_params.json``

    Parameters
    ----------
    params        : dict[zfit.zfitParameter, float]
        Result ``'result.params'`` of the minimisation of the loss function (given by :py:func:`launch_fit`)
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
        remove = el_to_list(remove, 1)

    param_results = {}
    for p in list(params.keys()):  # loop through the parameters
        # Retrieve name, value and error
        name_param = p.name
        if remove is not None:
            for t in remove:
                name_param = name_param.replace(t, '')

        value_param = params[p]['value']
        param_results[name_param] = value_param
        if uncertainty:
            error_param = params[p]['minuit_hesse']['error']
            param_results[name_param + '_err'] = error_param

    if dic_add is not None:
        for key, value in dic_add.items():
            param_results[key] = value

    dump_json(param_results, name_file + '_params', folder_name=folder_name)


def launch_fit(model, data, extended=False, verbose=True):
    """Fit the data with the model

    Parameters
    ----------
    model    : zfit.pdf.BasePDF
        Overall PDF for the fit
    data     : zfit.data.Data
        Data that will be fitted to
    extended : if True
        Define or not an extended loss function
    verbose  : Bool
        print or not the result of the fit

    Returns
    --------
    result: zfit.minimize.FitResult
        result of the minimisation of the likelihood
    param : Dict[ZfitParameter, float]
        Result ``result.params`` of the minimisation of the loss function (given by :py:func:`launch_fit`)
    """

    # Minimisation of the loss function
    if extended:
        nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=data)
    else:
        nll = zfit.loss.UnbinnedNLL(model=model, data=data)

    # create a minimizer
    minimizer = zfit.minimize.Minuit(verbosity=verbose * 5)
    # minimise with nll the model with the data
    result = minimizer.minimize(nll)

    # do the error calculations, here with Hesse
    param_hesse = result.hesse()  # get he hessien
    # param_errors, _ = result.errors(method='minuit_minos') # get the errors
    # (gaussian)

    param = result.params
    if verbose:
        print(param)
    return result, param

"""
Fitting tools working with roofit
"""

import numpy as np
import os.path as op



from ROOT import TFile, RooFit, RooRealVar, RooArgList
from HEA.config import loc
import ROOT

from HEA.fit.params import params_into_dict
from HEA.tools.dir import create_directory
from HEA.tools.da import el_to_list, add_in_dic
from HEA.tools.string import add_text
from HEA.tools.assertion import is_list_tuple

# ROOT.gROOT.LoadMacro(op.join(loc['libraries'], 
#                              "B2DHPartRecoPDFs/src/RooHILLdini.cpp"))
# ROOT.gROOT.LoadMacro(op.join(loc['libraries'], 
#                              "B2DHPartRecoPDFs/src/RooHORNSdini.cpp"))

# ROOT.gROOT.LoadMacro(op.join(loc['libraries'],
#                              "B2DHPartRecoPDFs/src/RooHILLdini_misID.cpp"))                    
# ROOT.gROOT.LoadMacro(op.join(loc['libraries'],
#                              "B2DHPartRecoPDFs/src/RooHORNSdini_misID.cpp"))

from ROOT import RooCBShape, RooAddPdf #, RooCrystalBall 
    
def evaluate_pdf_root(x, model_hist):
    """ Use interpolation to compute the value of the pdf
    (given by ``model_hist``) at x.
    
    Parameters
    ----------
    x : array-like
        where to evaluate the pdf
    model_hist: RooCurve
        Histogram of the pdf
    
    Returns
    -------
    y: np.array
        Model evaluated at x
    """
    return np.array([model_hist.interpolate(v) for v in x])


def define_params_root(initial_values, cut_BDT=None, num=None, other_params=None):
    """Define zparams from the dictionnary initial_values

    Parameters
    ----------

    initial_values : dict
        {"name_variable": {"value":, "low":, "high":}}
        or {"name variable": [value, low, high]}
    cut_BDT        : float
        performed cut on the BDT (BDT > cutBDT)
    num            : integer
        Index of the fit. add ``";{num}"`` at the end of the variable/
        the other functions I wrote allow to ignore the ``";{num}"`` in the name of the variable. This is used manely in order to define a parameter several times (when tuning their values to make the fit convergent)
    other_params : list of dict
        list of parameters already defined
    
    Returns
    -------
    rooParams     : dict[str, RooRealVar]
        Dictionnary of root parameters whose keys are the name of the variables and:

        * if cut_BDT is None, the key is just the name of the variable
        * else, the key is ``"{name_variable}|BDT{cut_BDT}"``
    """
    
    rooParams = {}
    for var in initial_values.keys():
        init = initial_values[var]
        
        if isinstance(init, int):
            assert other_params is not None
            rooParams[var] = other_params[init][var] 
            # retrieve the parameter already defined in another pdf
            
        else:
            if cut_BDT is not None:
                name_var = f"{var}|BDT{cut_BDT}"
            else:
                name_var = var
            if num is not None:
                name_var_num = add_text(name_var, str(num), ';')
            else:
                name_var_num = name_var
            
            if isinstance(init, dict):
                if 'low' not in init or 'high' not in init:
                    rooParams[var] = RooRealVar(name_var_num, name_var, 
                                                init['value'])
                elif 'value' not in init:
                    rooParams[var] = RooRealVar(name_var_num, name_var, 
                                                init['low'], init['high'])
                else:
                    rooParams[var] = RooRealVar(name_var_num, name_var, 
                                                init['value'], init['low'],
                                                init['high'])
            else:
                rooParams[var] = RooRealVar(name_var_num, name_var,
                                            *init)
            

    return rooParams


def define_all_params(all_initial_values, **kwargs):
    """ Define all the parameters from their 
    initial values.
    
    Parameters
    ----------
    
    all_initial_values: dict or list(dict)
        Initial values or list of initial values
        of the parameters (if there are more than
        1 pdf).
        Each dictionnary specifies (or not) the 'value',
        and the 'low' and 'high' bounds.
        It can also be a tuple instead of a dictionnary,
        wehre the three latter quantities must be defined
        in this order.
    **kwargs:
        :py:func: passed to :py:func:`define_params_root`
    
    Returns
    -------
    rooParams : dict
        Associates to a name a parameter
    
    """

    if is_list_tuple(all_initial_values):
        rooParams = []
        for one_pdf_initial_values in all_initial_values:
            rooParams.append(
                define_params_root(
                    one_pdf_initial_values,
                    other_params=rooParams,
                    **kwargs)
            )

    else:
        rooParams = define_params_root(all_initial_values, **kwargs)
    
    return rooParams

def find_key_startswith(dic, start_key):
    """ Find the first key in ```dic``
    that starts with ``start_key``
    
    Parameters
    ----------
    dic: dict
        python dictionnary
    start_key : str
        Pattern that we a key of ``dict`` to start with
    
    Returns
    -------
    matched_key: str
        first key that starts with ``start_key``
    """
    for key in dic.keys():
        if key.startswith(start_key):
            return key

def unpack_rooParams(rooParams, PDF_name):
    """
    
    Parameters
    ----------
    rooParams : dict
        Associates to a name a parameter
    PDF_name: str
        name of the model (in order to define the correct
        order of the parameters)
        If there is a ``';'`` in the PDF name, don't take
        into account what is after.
    
    Returns
    -------
    rooParams_list: tuple
        list of root parameters to be passed to the pdf,
        in the correct order!
    """
    
    index = PDF_name.find(';')
    if index!=-1:
        PDF_name = PDF_name[:index]
    
    if PDF_name=='HILL' or PDF_name=='HORNS':
        ordered_keys = ['a', 'b', 'csi', 'shift', 'sigma', 'r', 'f']

    elif PDF_name=='CB':
        ordered_keys = ['mu', 'sigma', 'alpha', 'n']
    
    elif PDF_name=='doubleCB':
        ordered_keys = ['mu', 'sigma', 'alphaL', 'nL', 'alphaR', 'nR']
    ordered_rooParams = []
    for key in ordered_keys:
        key_dict = find_key_startswith(rooParams, key)
        ordered_rooParams.append(rooParams[key_dict])
    
    return tuple(ordered_rooParams)

def get_PDF(PDF_name):
    """ Return a PDF given its name
    
    Parameters
    ----------
    PDF_name: str
        name of the model (in order to define the correct
        order of the parameters)
        If there is a ``';'`` in the PDF name, don't take
        into account what is after.
        
    Returns
    -------
    pdf : RooArgPdf
        Root PDF
    
    """
    index = PDF_name.find(';')
    if index!=-1:
        PDF_name = PDF_name[:index]
    
    from ROOT import RooHILLdini, RooHORNSdini
    
    name_to_PDF = {
        'HORNS': RooHORNSdini,
        'HILL' : RooHILLdini,
        'CB'   : RooCBShape,
#         'doubleCB': RooCrystalBall
    }
    
    return name_to_PDF[PDF_name]

def get_models(rooVariable, rooParams,
               PDF_names, models_types='n'):
    """
    
    Parameters
    ----------
    rooVariable: rooRealVar
           Fitted variable
    PDF_names: str or list(str)
        list of model names, defined in the :py:func:`get_PDF`
        dictionnary. No more than 2 PDFs.
        The first corresponds to the total model.
    models_types  : str
        type of each mode 
        (one character for each model or for a list of models):

        * ``'m'`` : model (sum) ; should always be the FIRST ONE !!
        * ``'s'`` : signal
        * ``'b'`` : background
    *rooParams:
        lists of RooParameters
        * 0th is the frac between the two PDFs (if two PDFs)
        * First for the first PDF
        * Second for the second PDF
    
    Returns
    -------
    models: RooAbsPdf or list(RooAbsPdf)
        corresponding model or ``[sum model, *submodels]``
    """
    
        
    # If only one model
    if is_list_tuple(PDF_names) and len(PDF_names)==1:
        PDF_names = PDF_names[0]
    
    if not is_list_tuple(PDF_names):
        return get_PDF(PDF_names)(
            PDF_names, models_types, 
            rooVariable,
            *unpack_rooParams(rooParams, PDF_names))
    
    else:
        models_types = el_to_list(models_types, len(PDF_names) + 1)
        assert len(PDF_names)<=2
        models = [None]
        for i in range(len(PDF_names)):
            models.append(
                get_PDF(PDF_names[i])(
                    PDF_names[i], models_types[i+1],
                    rooVariable,
                    *unpack_rooParams(rooParams[i+1], PDF_names[i]),
                )
            )
        
        models[0] = RooAddPdf(
            f'Sum of {PDF_names[0]} and {PDF_names[1]}', models_types[0],
            RooArgList(models[1], models[2]),
            RooArgList(tuple(rooParams[0].values()))
        )
   
        return models
    
    
    

def get_n_dof_model_root(model, data):
    """ Get the number of d.o.f. of a zfit model. (new version!)
    A d.o.f. corresponds to a floated parameter in the model.

    Parameters
    ----------
    model: RooArgPDF
        Model (PDF)
    data: RooArgSet or RooRealVar concerned by the model
        Data
        
    Returns
    -------
    n_dof: int
        number of d.o.f. in the model
    """
    n_dof = 0
    for param in model.getParameters(data): # loop over all the parameters
        n_dof += (not param.isConstant())
    
    return n_dof



def launch_fit(pdf, data, with_weights=False, extended=False):
    """ Fit a pdf to data
    
    Parameters
    ----------
    pdf: RooRealPdf
        pdf that needs to be fitted to data
    data : RooDataSet or RooDataHist
        Data to be fitted to
    
    
    Returns
    -------
    result: RooFitResult
        result of the fit
    params: dict
        dictionnary of the result of the fit.
        Associates to a fitted parameter (key)
        a dictionnary that contains its value (key: 'v')
        and its error(key: 'e')
    
    """
    
    if with_weights:
        result = pdf.fitTo(
            data,
            RooFit.Save(), RooFit.NumCPU(8,0), RooFit.Optimize(False), 
            RooFit.Offset(True), RooFit.Minimizer("Minuit2", "migrad"),
            RooFit.Strategy(2), RooFit.SumW2Error(True),
            RooFit.Extended(extended)
        )
    else:
        result = pdf.fitTo(
            data,
            RooFit.Save(), RooFit.NumCPU(8,0), RooFit.Optimize(False), 
            RooFit.Offset(True), RooFit.Minimizer("Minuit2", "migrad"),
            RooFit.Strategy(2), RooFit.Extended(extended)
        )
    params = params_into_dict(result, method='root')
    
    return result, params
    
def get_reduced_chi2_root(rooVariable, model, data_h):
    """ Get the chi2 of a roofit fit
    
    Parameters
    ----------
    rooVariable: rooRealVar
        Fitted variable of the dataset
    model: RooRealPdf
        Fitted pdf
    data_h :  RooDataHist
        Binned data
    
    
    Returns
    -------
    reduced_chi2: float
        Reduced :math:`\\chi^2` of the fit
    """
    # Get chi2 ----------------------------
    frame = rooVariable.frame()
    data_h.plotOn(frame)
    model.plotOn(frame)
    reduced_chi2 = frame.chiSquare()
    print('******************************************')
    print("chi2/ndof = %.2f" % reduced_chi2)
    print('******************************************')
    return reduced_chi2
    
def save_fit_result_root(result, file_name, folder_name=None):
    """
    Parameters
    ----------
    result: RooFitResult
        resut of a fit
    file_name: str
        name of the root file to save
    folder_name: str
        name of the folder where to save the root file
    """
    
    path = loc['out'] + 'root/'
    path = create_directory(path, folder_name)
    path += f"/{file_name}.root"
    print(f"Root file saved in {path}")
    
    file = TFile(path, 'recreate')
    file.cd() 
    result.Write()
    file.Write()
    file.Close()
    
def retrieve_fit_result_root(file_name, folder_name):
    """ Retrieve a fit result file localted 
    in ``loc['out']/root/``
    
    Parameters
    ----------
    file_name: str
        name of the file to retrieve
    folder_name: str
        folder where the file is located
    
    Returns
    -------
    fit_results: rooFitResult
        result of a fit saved in the root file
    """
    path = loc['out'] + 'root/'
    path = create_directory(path, folder_name)
    path += f"/{file_name}.root"
    
    fit_results = TFile.Open(path, folder_name)
    
    return fit_results
    
"""
Fit tools working with roofit
"""

import numpy as np

from ROOT import TFile, RooFit
from HEA.config import loc

from HEA.fit.params import params_into_dict
from HEA.tools.dir import create_directory

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



def launch_fit(pdf, data):
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
    
    result = pdf.fitTo(
        data,
        RooFit.Save(), RooFit.NumCPU(8,0), RooFit.Optimize(False), 
        RooFit.Offset(True), RooFit.Minimizer("Minuit2", "migrad"),
        RooFit.Strategy(2),
    )
    params = params_into_dict(result, method='root')

    return result, params
    



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
    

    
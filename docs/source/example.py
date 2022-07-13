"""Sample code to illustrate fitterpp for a hyperbola."""

import fitterpp as fpp
import lmfit
import matplotlib.pyplot as plt
import numpy as np

CENTER = 10
MULT = 2.0

XVALUES = range(20)
DATA = np.array([MULT*(n - CENTER)**2 + 0*np.random.rand() for n in XVALUES])
if False:
    plt.scatter(XVALUES, DATA)
    plt.show()

def myFunc(params):
    """
    This is an example of an residuals calculation function.
    The function calculates the residuals between DATA and the 
    estimated values using the parameters provided.

    Parameters
    ----------
    params: list-float
        params[0]: parameter provided for center
        params[1]: parameter provided for multiplier
    
    Returns
    -------
    list-float (residuals)
    """
    center = params["center"].value
    mult = params["mult"].value
    estimates = np.array([mult*(n - center)**2 for n in XVALUES])
    print((center, mult))
    print(np.log10(sum(estimates**2)))
    import pdb; pdb.set_trace()
    return DATA - estimates


parameters = lmfit.Parameters()
parameters.add("center", value=1, min=1, max=15)
parameters.add("mult", value=5, min=5, max=15)
methods = fpp.Fitterpp.mkFitterMethod(methodNames=fpp.METHOD_DIFFERENTIAL_EVOLUTION)
methods = fpp.Fitterpp.mkFitterMethod(methodNames=fpp.METHOD_LEASTSQ)
fitter = fpp.Fitterpp(myFunc, parameters, methods=methods)
fitter.execute()
import pdb; pdb.set_trace()
   

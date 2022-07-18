import fitterpp as fpp
import lmfit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

CENTER = 10
MULT = 2.0
XVALUES = range(20)

def calcParabola(center=0, mult=1, xvalues=XVALUES, is_noise=False):
    """
    Calculates values for a parabola, optionally with noise.

    Parameters
    ----------
    center: float (parameter to fit)
    mult: float (parameter to fit)
    xvalues: int (not a fitted parameter)
    is_noise: bool (include noise)
    
    Returns
    -------
    DataFrame
        Columns: x, v
    """
    estimates = np.array([mult*(n - center)**2 
          + 10*is_noise*np.random.rand() for n in xvalues])
    return pd.DataFrame({"x": xvalues, "y": estimates})


DATA_DF = calcParabola(center=CENTER, mult=MULT, is_noise=True)


# Construct the inputs to the fitter
parameters = lmfit.Parameters()
parameters.add("center", value=0, min=0, max=100)
parameters.add("mult", value=0, min=0, max=100)
fitter = fpp.Fitterpp(calcParabola, parameters, DATA_DF, is_collect=True)
fitter.execute()
# Statistics
print(fitter.report())
fitter.plotPerformance()
fitter.plotQuality()
# Plot the result
center = fitter.final_params["center"].value
mult = fitter.final_params["mult"].value
fitted_df = calcParabola(center=center, mult=mult)
plt.scatter(DATA_DF["x"], DATA_DF["y"])
plt.plot(fitted_df["x"], fitted_df["y"], color="red")
plt.show()

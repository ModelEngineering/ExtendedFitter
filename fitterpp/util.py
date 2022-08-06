
from fitterpp import constants as cn

import copy
import inspect
import lmfit
import numpy as np
import scipy.stats as stats

MIN_FRAC = 0.5
MAX_FRAC = 2.0
VALUE_FRAC = 1.0


def calcRelError(actual:float, estimated:float, isAbsolute:bool=True):
    """
    Calculates the relative error of the estimate.

    Parameters
    ----------
    actual: actual value
    estimated: estimated values
    isAbsolute: return absolute value

    Returns
    -------
    float
    """
    if np.isclose(actual, 0):
        return np.nan
    relError = (estimated - actual) / actual
    if isAbsolute:
        relError = np.abs(relError)
    return relError

def filterOutliersFromZero(data, maxSL):
    """
    Removes values that are distant from 0 using a F-statistic criteria.
    Extreme values are iteratively removed until the F-statistic exceeds
    the significance level.

    Parameters
    ----------
    data: iterable-float
    maxSL: float
        Maximum significance level to accept a difference in variance
        A larger maxSL means more filtering since it's more likely that an
        extreme value will be filtered.

    Returns
    -------
    np.array
    """
    def calcSL(arr1, arr2):
        """
        Calculates the significance level that the variance of the first array
        is larger than the variance of the second array.

        Returns
        -------
        float
        """
        def calc(arr):
            return np.var(arr), len(arr) - 1
        #
        var1, df1 = calc(arr1)
        var2, df2 = calc(arr2)
        if var2 > 0:
            fstat = var1/var2
        else:
            fstat = 1000*var1
        sl = 1 - stats.f.cdf(fstat, df1, df2)
        return sl
    sortedData = sorted(data, key=lambda v: np.abs(v), reverse=True)
    sortedData = np.array(sortedData)
    # Construct the array without outliers
    for _ in range(len(data)):
        shortSortedData = sortedData[1:]
        sl = calcSL(sortedData, shortSortedData)
        if sl < maxSL:
            sortedData = shortSortedData
        else:
            break
    #
    return sortedData

def copyObject(oldObject, newInstance=None):
    """
    Copies the non "__" instance variables of the old object into the new instance.

    Parameters
    ----------
    oldObject: an existing object
    newInstance: updated
    """
    if newInstance is None:
        newInstance = oldObject.__class__()
    for attr in oldObject.__dict__:
        if len(attr) >= 2:
            if attr[0:2] == "__":
                continue
        value = oldObject.__getattribute__(attr)
        if "copy" in dir(value):
            newValue = value.copy()
        else:
            newValue = copy.deepcopy(value)
        try:
            newInstance.__setattr__(attr, newValue)
        except:
            continue
    return newInstance

def getKwargNames(func):
    """
    Obtains the keyword arguments in the function definition.

    Parameters
    ----------
    func: Function

    Returns
    -------
    list-str
    """
    spec = inspect.getfullargspec(func)
    argList = spec.args
    numKwarg = len(spec.defaults)
    kwargNames = argList[-numKwarg:]
    return kwargNames

def kwargs():
    """
    Decorator that provides adds properties to a function:
        defined: keyword arguments in function definition
        passed: keyword arguments/values passed
    """
    def decorator(function):
        def inner(*args, **kwwargs):
            inner.defined = getKwargNames(function)
            inner.passed = kwwargs
            inner.name = function.__qualname__
            return function(*args, **kwwargs)
        return inner
    return decorator

def validateKwargs(function):
    """
    Validates that the keywords passed to the function are a subset of the
    parameters defined.
    The function must use the @kwargs decorator.

    Parameters
    ----------
    function: function

    Raises: ValueError
    """
    missing = [p for p in function.passed.keys() for p in function.defined]
    if len(missing) > 0:
        raise ValueError(
              "The following keyword parameters do not match: %s" % str(missing))

def ppDict(dct, indent=0):
    """
    Does a pretty print of a dictionary.values.

    Parameters
    ----------
    dct: dict
    indent: int
        spaes indented

    Returns
    -------
    str
    """
    spaces = "".join([" " for _ in range(indent)])
    lines = []
    for key, value in dct.items():
        line = "%s%s:  %s" % (spaces, key, value)
        lines.append(line)
    return "\n".join(lines)

def updateParameterValues(parameters, newValuesDct):
    """
    Updates the values of lmfit.Parameters.

    Parameters
    ----------
    parameters: lmfit.Parameters
    newVvaluesDct: dict
        key: parameter name
        value: value of parameter
    """
    currentValuesDct = parameters.valuesdict()
    for parameterName in currentValuesDct.keys():
        parameters[parameterName].set(
              value=newValuesDct[parameterName])

def dictToParameters(dct, min_frac=MIN_FRAC, max_frac=MAX_FRAC,
      value_frac=VALUE_FRAC, is_random_initial=False):
    """
    Converts a dictionary to parameters according to the argument
    specifications. Parameters near 0 are ignored.

    Parameters
    ----------
    dct: dict
        key: name of parameter
        value: initial value
    min_frac: float (fraction of value to set as parameter min)
    max_frac: float (fraction of value to set as parameter min)
    value_frac: float (set initial to a fraction of the value)
    is_random_initial: bool (initial value is random between min & max)
    
    Returns
    -------
    lmfit.Parameters
    """
    parameters = lmfit.Parameters()
    for name, value in dct.items():
        val = float(value)
        if np.isclose(val, 0.0):
            continue
        min_val = val*min_frac
        max_val = val*max_frac
        if is_random_initial:
            rand = np.random.rand()
            value_val = min_val + rand*(max_val - min_val)
        else:
            value_val = val*value_frac
        try:
            parameters.add(name=name, value=value_val, min=min_val, max=max_val)
        except KeyError:
            continue
    return parameters

######################### CLASSES ########
class FitterppMethod():

    """Container for optimization information"""

    def __init__(self, method, kwargs):
        self.method = method
        self.kwargs = dict(kwargs)

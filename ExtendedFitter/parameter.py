"""
Manages parameters for evaluation functions.

A Parameter is an abstraction for a variable that is constant during
model evaluation has a lower bound, upper bound, and value.

A ParameterCollection is a collection of Parameters.

A ParameterManager manages parameters from multiple models.

"""

from ExtendedFitter import constants as cn

import matplotlib.pyplot as plt
import numpy as np
import lmfit

LOWER_PARAMETER_MULT = 0.95
UPPER_PARAMETER_MULT = 1.05


class Parameter():

    def __init__(self, name, lower=cn.PARAMETER_LOWER_BOUND,
              value=None, upper=cn.PARAMETER_UPPER_BOUND,
              possible_values=None):
        """
        Parameters
        ----------
        name: str (Name of the parameter)
        lower: float (minimum value)
        upper: float (maxium value)
        possible_values: object (enum type of parameters)
        
        Returns
        -------
        """
        self.name = name
        self.lower = lower
        self.upper = upper
        self.value = value
        self.possible_values = possible_values
        if value is None:
            self.value = (lower + upper)/2.0
        if self.value <= self.lower:
            self.lower = LOWER_PARAMETER_MULT*self.value
        if self.value >= self.upper:
            self.upper = UPPER_PARAMETER_MULT*self.value
        if np.isclose(self.lower, 0.0):
            self.lower = -0.001
        if np.isclose(self.upper, 0.0):
            self.upper = 0.001

    def __str__(self):
        return self.name

    def copy(self, name=None):
        if name is None:
            name = self.name
        return Parameter(name, lower=self.lower, upper=self.upper,
              value=self.value, possible_values=self.possible_values)

    def updateLower(self, value):
        self.lower = min(self.lower, value)

    def updateUpper(self, value):
        self.upper = max(self.upper, value)

    @staticmethod
    def mkParameter(parameter):
        """
        Creates an lmfit parameter

        Parameters
        ----------
        parameter: Parameter or lmfit.Parameter
        
        Returns
        -------
        lmfit.Parameter
        """
        if isinstance(parameter, lmfit.Parameter):
            return parameter
        lmfitParameter = lmfit.Parameter(
              name=parameter.name,
              value=parameter.value,
              min=parameter.lower,
              max=parameter.upper)
        return lmfitParameter

    @staticmethod
    def toLMfit(parameters):
        """
        Creates lmfit.Parameters

        Parameters
        ----------
        parameters: list-Parameter
        
        Returns
        -------
        lmfit.Parameters
        """
        lmfitParameters = lmfit.Parameters()
        _ = [lmfitParameters.add(Parameter.mkParameter(p)) for p in parameters]
        return lmfitParameters
        

class ParameterManager():
    """
    Manages overlapping parameters for models.
    - modelDct: key is model name; value is lmfit.Parameters
    - parameterDct: key is parameter name: value lmfit.Parameter
    """

    def __init__(self, model_names, parameter_collection):
        """
        Parameters
        ----------
        parameter_collection: list-lmfit.Parameters
        model_names: list-str
            name of models corresponding to parameters
        """
        self.parameterDct = self._mkParameterDct(parameter_collection)
        self.modelDct = self._mkModelDct(model_names, parameter_collection)

    def _mkParameterDct(self, parameter_collection):
        """
        The dictionary that relates parameter names to models.
        Where there are multiple occurrences of the same parameter,
        the min, max, and value of the parameter are adjusted.

        Returns
        -------
        dict
            key: str (parameter name)
            value: list-model_name
        """
        parameterDct = {}
        countDct = {}
        for parameters in parameter_collection:
            for parameterName, parameter in parameters.items():
                if not parameterName in parameterDct.keys():
                    newParameter = lmfit.Parameter(
                          name=parameter.name,
                          min=parameter.min,
                          max=parameter.max,
                          value=parameter.value)
                    parameterDct[parameterName] = newParameter
                    countDct[parameterName] = 1
                else:
                    # Adjust parameter values
                    curParameter = parameterDct[parameterName]
                    curParameter.set(min=min(curParameter.min, parameter.min))
                    curParameter.set(max=max(curParameter.max, parameter.max))
                    curParameter.set(value=curParameter.value + parameter.value)
                    countDct[parameterName] += 1
        for parameterName, parameter in parameterDct.items():
            parameter.set(value=parameter.value/countDct[parameterName])
        return parameterDct

    def _mkModelDct(self, model_names, parameter_collection):
        """
        Ensures that use the same lmfit.Parameter object for shared
        parameters.

        Parameters
        ----------
        model_names: list-str
        parameter_collection: list-lmfit.Parameters
        
        Returns
        -------
        dict
            key: model_name
            value: lmfit.Parameters
        """
        modelDct = {}
        for model_name, parameters in zip(model_names, parameter_collection):
            modelParameters = lmfit.Parameters()
            for parameterName, _ in parameters.items():
                modelParameters.add(self.parameterDct[parameterName])
            modelDct[model_name] = modelParameters
        # Consruct parameters for ALL model
        parameters = lmfit.Parameters()
        [parameters.add(p) for p in self.parameterDct.values()]
        modelDct[cn.ALL] = parameters
        return modelDct

    def updateValues(self, parameters):
        """
        Updates parameter values.

        Parameters
        ----------
        parameters: lmfit.Parameters
        """
        for parameterName, parameter in parameters.items():
            self.parameterDct[parameterName].set(value=parameter.value)

    def getParameters(self, model_name=cn.ALL):
        """
        Makes lmfit.Parameters for the model. If none, then constructs one
        for all parameters.

        Parameters
        ----------
        model: str

        Returns
        -------
        lmfit.Parameters
        """
        return self.modelDct[model_name]

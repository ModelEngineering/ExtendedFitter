# -*- coding: utf-8 -*-
"""Extended Parameter Fitting

Created on July 4, 2022

fitterpp extends lmfit optimizations by:
1. Ensuring that the parameters chosen have the lowest residuals sum of squares
2. Providing for a sequence of optimization methods
3. Providing an option to repeat a method sequence with different randomly
   chosen initial parameter values (numRandomRestart).

TODO

1. fitterpp runs tests

"""

from fitterpp.logs import Logger
from fitterpp import util
from fitterpp import constants as cn
from fitterpp.function_wrapper import FunctionWrapper

import copy
import lmfit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


class Fitterpp():
    """
    Implements an interface to parameter fitting methods that provides
    additional capabilities and bug fixes.
    The class also handles an oddity with lmfit that the final parameters
    returned may not be the best.

    Usage
    -----
    fitter = fitterpp(calcResiduals, params, [cn.METHOD_LEASTSQ])
    fitter.execute()
    """

    def __init__(self, user_function, initial_params, data_df, methods=None, logger=None,
          is_collect=False):
        """
        Parameters
        ----------
        user_function: Funtion
           Parameters
               keyword parameters that correspond to the names in initial_params
           Returns
               pd.DataFrame
                   columns: variable returned
                   row: instance of variable values
           Arguments
            lmfit.parameters
        initial_params: lmfit.Parameters (initial values of parameters)
        data_df: pd.DataFrame
            Has same structure as the output of the function
        methods: list-str (e.g., "leastsq", "differential_evolution")
        """
        self.initial_params = initial_params
        self.user_function = user_function
        self.data_df = data_df
        self.fitting_columns = list(data_df.columns)
        self.function = self._mkFitterFunction()
        self.methods = methods
        if self.methods is None:
            self.methods = self.mkFitterMethod()
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        self.is_collect = is_collect
        # Statistics
        self.performance_stats = []  # durations of function executions
        self.quality_stats = []  # residual sum of squares, a quality measure
        # Outputs
        self.duration = None  # Duration of parameter search
        self.final_params = None
        self.minimizer_result = None
        self.rssq = None
       
    def execute(self):
        """
        Performs parameter fitting function.
        Result is self.final_params
        """
        start_time = time.time()
        last_excp = None
        self.final_params = self.initial_params.copy()
        minimizer = None
        for fitter_method in self.methods:
            method = fitter_method.method
            kwargs = fitter_method.kwargs
            wrapper_function = FunctionWrapper(self.function,
                  is_collect=self.is_collect)
            minimizer = lmfit.Minimizer(wrapper_function.execute, self.final_params)
            self.minimizer_result = minimizer.minimize(method=method, **kwargs)
            # Update the parameters
            if wrapper_function.bestParamDct is not None:
                util.updateParameterValues(self.final_params,
                      wrapper_function.bestParamDct)
            # Update other statistics
            self.rssq = wrapper_function.rssq
            self.performance_stats.append(list(wrapper_function.perfStatistics))
            self.quality_stats.append(list(wrapper_function.rssqStatistics))
        if minimizer is None:
            msg = "*** Optimization failed."
            self.logger.error(msg, last_excp)
        else:
            self.duration = time.time() - start_time

    def report(self):
        """
        Reports the result of an optimization.

        Returns
        -------
        str
        """
        VARIABLE_STG = "[[Variables]]"
        CORRELATION_STG = "[[Correlations]]"
        if self.minimizer_result is None:
            raise ValueError("Must do execute before doing report.")
        value_dct = self.final_params.valuesdict()
        values_stg = util.ppDict(dict(value_dct), indent=4)
        reportSplit = str(lmfit.fit_report(self.minimizer_result)).split("\n")
        # Eliminate Variables section
        inVariableSection = False
        trimmedReportSplit = []
        for line in reportSplit:
            if VARIABLE_STG in line:
                inVariableSection = True
            if CORRELATION_STG in line:
                inVariableSection = False
            if inVariableSection:
                continue
            trimmedReportSplit.append(line)
        # Construct the report
        newReportSplit = [VARIABLE_STG]
        newReportSplit.extend(values_stg.split("\n"))
        newReportSplit.extend(trimmedReportSplit)
        return "\n".join(newReportSplit)

    @staticmethod
    def mkFitterMethod(method_names=None, method_kwargs=None,
          max_fev=cn.MAX_NFEV_DFT):
        """
        Constructs an FitterMethod
        Parameters
        ----------
        method_names: list-str/str
        method_kwargs: list-dict/dict

        Returns
        -------
        list-FitterMethod
        """
        if method_names is None:
            method_names = [cn.METHOD_LEASTSQ]
        if isinstance(method_names, str):
            method_names = [method_names]
        if method_kwargs is None:
            method_kwargs = {}
        # Ensure that there is a limit of function evaluations
        new_method_kwargs = dict(method_kwargs)
        if cn.MAX_NFEV not in new_method_kwargs.keys():
            new_method_kwargs[cn.MAX_NFEV] = max_fev
        elif max_fev is None:
            del new_method_kwargs[cn.MAX_NFEV]
        method_kwargs = np.repeat(new_method_kwargs, len(method_names))
        #
        results = [util.FitterMethod(n, k) for n, k  \
              in zip(method_names, method_kwargs)]
        return results

    def plotPerformance(self, is_plot=True):
        """
        Plots the statistics for running the objective function.
        """
        if not self.is_collect:
            msg = "Must construct with isCollect = True "
            msg += "to get performance plot."
            raise ValueError(msg)
        # Compute statistics
        TOT = "Tot"
        CNT = "Cnt"
        AVG = "Avg"
        total_times = [sum(v) for v in self.performance_stats]
        counts = [len(v) for v in self.performance_stats]
        averages = [np.mean(v) for v in self.performance_stats]
        df = pd.DataFrame({
            TOT: total_times,
            CNT: counts,
            AVG: averages,
            })
        #
        df.index = [m.method for m in self.methods]
        _, axes = plt.subplots(1, 3)
        df.plot.bar(y=TOT, ax=axes[0], title="Total time",
              xlabel="method")
        df.plot.bar(y=AVG, ax=axes[1], title="Average time",
              xlabel="method")
        df.plot.bar(y=CNT, ax=axes[2], title="Number calls",
              xlabel="method")
        if is_plot:
            plt.show()

    def plotQuality(self, is_plot=True):
        """
        Plots the quality results
        """
        if not self.is_collect:
            msg = "Must construct with isCollect = True "
            msg += "to get quality plots."
            raise ValueError(msg)
        ITERATION = "iteration"
        _, axes = plt.subplots(len(self.methods))
        minLength = min([len(v) for v in self.quality_stats])
        # Compute statistics
        dct = {self.methods[i].method: self.quality_stats[i][:minLength]
            for i in range(len(self.methods))}
        df = pd.DataFrame(dct)
        df[ITERATION] = range(minLength)
        #
        for idx, method in enumerate(self.methods):
            if "AxesSubplot" in str(type(axes)):
                ax = axes
            else:
                ax = axes[idx]
            df.plot.line(x=ITERATION, y=method.method, ax=ax, xlabel="")
            ax.set_ylabel("SSQ")
            if idx == len(self.methods) - 1:
                ax.set_xlabel(ITERATION)
        if is_plot:
            plt.show()

    def _mkFitterFunction(self):
        """
        Creates the function used for doing fits.

        Parameters
        ----------
        function:
            Parameters
                only has keyword parameters, which are the same names
                as the parameters in lmfit.Parmeters 
            Returns
                DataFrame
        df: DataFrame
           Columns: variables
           Index: aligned with function  output
        
        Returns
        -------
        Function
            Parameters: lmfit.Parameters
            Returns: array(float)
        """
        kw_names = set(self.initial_params.valuesdict().keys())
        def fitter_func(parameters):
            dct  = parameters.valuesdict()
            parameter_names = dct.keys()
            diff = kw_names.symmetric_difference(parameter_names)
            if len(diff) > 0:
                msg = "Missing or extra keywards on call to fitter function: %s" % diff
                raise ValueError(msg)
            function_df = self.user_function(**dct)
            function_df = function_df[self.fitting_columns]
            return (self.data_df - function_df).values.flatten()
        #
        return fitter_func

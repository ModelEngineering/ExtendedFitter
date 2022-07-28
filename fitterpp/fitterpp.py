# -*- coding: utf-8 -*-
"""Extended Parameter Fitting

Created on July 4, 2022

fitterpp fits parameters of a function. Key features are;
1. Simplify fitting parameters to data;
2. Ensuring that the parameters chosen have the lowest residuals sum of squares
3. More sophisticated choice of optimization algorithms, such as using
   a sequence of algorithms and repeat a method sequence with different randomly
   chosen initial parameter values (numRandomRestart).

The fitting function should operate as follows:
    Inputs:
        keyword argument for each parameter name
        is_dataframe kewyword argument: returns DataFrame if True
    Returns:
        DataFrame for numpy.array. Index is the row key.
        Arr: 2d array (even if only 1 column)

TODO:
1. calc function returns np.array
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


ITERATION = "iteration"


class DFIntersectionFinder:

    # Finds the rows and columns that are common between two dataframes.
    # Provides the array index for the common rows and columns.
    #    self.row_idxs
    #    self.column_idxs

    def __init__(self, df, other_df):
        """
        Parameters
        ----------
        df: DataFrame
        other_df: DataFram
        """
        self.df = df
        self.other_df = other_df
        # Common row indices
        indices = list(df.index)
        other_indices = list(other_df.index)
        self.row_idxs = np.array([i for i in range(len(df))
              if indices[i] in other_indices])
        # Common column indices
        columns = list(df.columns)
        other_columns = list(other_df.columns)
        self.column_idxs = np.array([i for i in range(len(df.columns))
              if columns[i] in other_columns])

    def isCorrectShape(self, arr):
        """
        Validates that an array has the correct shape.

        Parameters
        ----------
        arr: np.array
        
        Returns
        -------
        bool
        """
        expected_shape = [len(self.row_idxs), len(self.column_idxs)]
        try:
            np.testing.assert_array_equal(np.shape(arr), expected_shape)
            return True
        except:
            return False


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
               is_dataframe (boolean)
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
        # The values array does not include the key
        self.is_collect = is_collect
        self.fitting_columns = list(data_df.columns)
        self.function = self._mkFitterFunction()
        self.methods = methods
        if self.methods is None:
            self.methods = self.mkFitterMethod()
        self.logger = logger
        if self.logger is None:
            self.logger = Logger()
        # Common indexes 
        kwargs = self.makeKwargs(self.initial_params)
        function_df = self.user_function(is_dataframe=True, **kwargs)
        self.function_common = DFIntersectionFinder(function_df,
              self.data_df)
        self.data_common = DFIntersectionFinder(self.data_df, function_df)
        self.data_arr = self.data_df.values[:, self.data_common.column_idxs]
        self.data_arr = self.data_arr[self.data_common.row_idxs, :]
        self.data_arr = self.data_arr.flatten()
        # Validate the output
        function_arr = self.user_function(is_dataframe=False, **kwargs)
        if not self.function_common.isCorrectShape(function_arr):
            msg = "The user function does not create an array "
            msg += "shape consistent with its DataFrame."
            raise ValueError(msg)
        # Statistics
        self.performance_stats = []  # durations of function executions
        self.quality_stats = []  # residual sum of squares, a quality measure
 
        # Outputs
        self.duration = None  # Duration of parameter search
        self.final_params = None
        self.minimizer_result = None
        self.rssq = None

    @staticmethod
    def makeKwargs(parameters):
        """
        Creates a key word dictionary from the parameters.

        Parameters
        ----------
        parameters: lmfit.Parameters
        
        Returns
        -------
        dict
        """
        kwargs = {}
        for key, value in parameters.valuesdict().items():
            kwargs[key] = value
        return kwargs

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
            method_names = cn.METHOD_FITTER_DEFAULTS
        if isinstance(method_names, str):
            method_names = [method_names]
        if method_kwargs is None:
            method_kwargs = {cn.MAX_NFEV: cn.MAX_NFEV_DFT}
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
        tick_names = [m.method for m in self.methods]
        tick_vals = list(range(len(tick_names)))
        df.index = tick_names
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
        df.plot.bar(y=TOT, ax=axes[0], title="Total time",
              xlabel="method", fontsize=18)
        df.plot.bar(y=AVG, ax=axes[1], title="Average time",
              xlabel="method", fontsize=18)
        df.plot.bar(y=CNT, ax=axes[2], title="Number calls",
              xlabel="method", fontsize=18)
        for idx in range(3):
            axes[idx].set_xticks(tick_vals, labels=tick_names,
                  rotation=25, fontsize=18)
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
        _, axes = plt.subplots(1, len(self.methods))
        # Compute statistics
        dct = {self.methods[i].method: self.quality_stats[i]
            for i in range(len(self.methods))}
        #
        for idx, method_name in enumerate(dct.keys()):
            if "AxesSubplot" in str(type(axes)):
                ax = axes
            else:
                ax = axes[idx]
            stats = dct[method_name]
            xvals = range(1, len(stats)+1)
            ax.plot(xvals, stats)
            if idx == 0:
                ax.set_ylabel("SSQ")
            ax.set_xlabel(ITERATION)
            ymax = 10*max(0.1, np.min(stats))
            ax.set_ylim([0, ymax])
            ax.set_title(method_name)
        if is_plot:
            plt.show()

    @staticmethod
    def _make2dArray(arr):
        """
        Makes the array two dimensional if it isn't already.

        Parameters
        ----------
        arr: np.array-float

        Returns
        -------
        arr: np.array-float - 2d
        """
        if arr.ndim == 2:
            return arr
        if arr.ndim == 1:
            nrow = len(arr)
            return np.reshape(arr, (nrow, 1))
        raise RuntimeError("Unexpected array shape.")

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
                msg = "Missing or extra keywards on call to fitter "
                msg += "function: %s" % diff
                raise ValueError(msg)
            function_arr = self.user_function(is_dataframe=False, **dct)
            function_arr = function_arr[:, self.function_common.column_idxs]
            function_arr = function_arr[self.function_common.row_idxs, :]
            function_arr = function_arr.flatten()
            residuals = self.data_arr - function_arr
            trues = [isinstance(v, float) for v in residuals.flatten()]
            return residuals
        #
        return fitter_func

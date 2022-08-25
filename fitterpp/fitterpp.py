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

import collections
import copy
import lmfit
import lhsmdu
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
    fitter.fit()
    """

    def __init__(self, user_function, initial_params, data_df,
          method_names=None, max_fev=cn.MAX_NFEV_DFT, num_latincube=0,
          logger=None, is_collect=False):
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
            Observational data (same structure as the output of the function)
        method_names: list-str/FitterppMethod
            Examples of names: "leastsq", "differential_evolution"
        max_fev: int (Maximum number of function evaluations)
        num_latincube: int (Num samples for latin cube of parameter initial values)
            A value of 0 means that "value" in each parameter will be used
        """
        self.initial_params = initial_params.copy()
        self.user_function = user_function
        self.data_df = data_df
        self.num_latincube = num_latincube
        # The values array does not include the key
        self.is_collect = is_collect
        self.fitting_columns = list(data_df.columns)
        self.function = self._mkFitterFunction()
        if method_names is None:
            self.methods = self.mkFitterppMethod(max_fev=max_fev)
        elif isinstance(method_names[0], util.FitterppMethod):
            self.methods = method_names
        elif isinstance(method_names[0], str):
            self.methods = self.mkFitterppMethod(method_names=method_names,
                max_fev=max_fev)
        else:
            raise ValueError("Invalid specification of method_names")
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

    def fit(self):
        """
        Performs parameter fitting function.
        Result is self.final_params
        """
        FitterResult = collections.namedtuple("FitterResult",
              ["mzr", "rssq", "prm"])
        #
        start_time = time.clock()
        last_excp = None
        minimizer = None
        if self.num_latincube == 0:
            parameters_lst = [self.initial_params]
        else:
            parameters_lst = self.makeParameterCube(self.initial_params,
                  self.num_latincube)
        best_result = FitterResult(mzr=None, rssq=1e10, prm=None)
        for parameters in parameters_lst:
            result_params = parameters.copy()
            for fitter_method in self.methods:
                method = fitter_method.method
                kwargs = fitter_method.kwargs
                wrapper_function = FunctionWrapper(self.function,
                      is_collect=self.is_collect)
                minimizer = lmfit.Minimizer(wrapper_function.execute, result_params)
                minimizer_result = minimizer.minimize(method=method, **kwargs)
                self.performance_stats.append(list(wrapper_function.perfStatistics))
                self.quality_stats.append(list(wrapper_function.rssqStatistics))
                # Update the parameters
                rssq = wrapper_function.rssq
                if wrapper_function.bestParamDct is not None:
                    util.updateParameterValues(result_params,
                          wrapper_function.bestParamDct)
            if rssq < best_result.rssq:
                best_result = FitterResult(mzr=minimizer_result, rssq=rssq,
                      prm=result_params)
        # Check if successful
        if best_result.mzr is None:
            msg = "*** Optimization failed."
            self.logger.error(msg, last_excp)
        else:
            self.duration = time.clock() - start_time
        # Seve the best result
        self.final_params = best_result.prm
        self.minimizer_result = best_result.mzr
        self.rssq = best_result.rssq

    @staticmethod
    def makeParameterCube(parameters, num_sample):
        """
        Creates an lmfit.Parameters based on the number of Latin Cube samples desired.

        Returns
        -------
        list-lmfit.Parameters
        """
        parameters_lst = []
        num_parameter = len(parameters.valuesdict())
        samples = lhsmdu.sample(num_sample, num_parameter).tolist()
        for sample in samples:
            new_parameters = lmfit.Parameters()
            for idx, name in enumerate(parameters.valuesdict().keys()):
                parameter = parameters.get(name)
                initial_value = parameter.min  \
                      + sample[idx]*(parameter.max - parameter.min)
                new_parameters.add(name=name, min=parameter.min, max=parameter.max,
                      value=initial_value)
            parameters_lst.append(new_parameters)
        return parameters_lst


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
    def mkFitterppMethod(method_names=None, method_kwargs=None,
          max_fev=cn.MAX_NFEV_DFT):
        """
        Constructs an FitterppMethod
        Parameters
        ----------
        method_names: list-str/str
        method_kwargs: list-dict/dict

        Returns
        -------
        list-FitterppMethod
        """
        if method_names is None:
            method_names = cn.METHOD_FITTER_DEFAULTS
        if isinstance(method_names, str):
            method_names = [method_names]
        if method_kwargs is None:
            method_kwargs = {cn.MAX_NFEV: max_fev}
        # Ensure that there is a limit of function evaluations
        new_method_kwargs = dict(method_kwargs)
        if cn.MAX_NFEV not in new_method_kwargs.keys():
            new_method_kwargs[cn.MAX_NFEV] = max_fev
        elif max_fev is None:
            del new_method_kwargs[cn.MAX_NFEV]
        method_kwargs = np.repeat(new_method_kwargs, len(method_names))
        #
        results = [util.FitterppMethod(n, k) for n, k  \
              in zip(method_names, method_kwargs)]
        return results

    def plotPerformance(self, is_plot=True):
        """
        Plots the statistics for running the objective function.

        Parameters
        ----------
        is_plot: bool (plot the output)

        Returns
        -------
        pd.DataFrame
            Columns	
                tot: total_times
                cnt: counts
                avg: averages
            index: method

        """
        if not self.is_collect:
            msg = "Must construct with isCollect = True "
            msg += "to get performance plot."
            raise ValueError(msg)
        # Compute statistics
        TOT = "tot"
        CNT = "cnt"
        AVG = "avg"
        total_times = [sum(v) for v in self.performance_stats]
        counts = [len(v) for v in self.performance_stats]
        averages = [np.mean(v) for v in self.performance_stats]
        df = pd.DataFrame({
            TOT: total_times,
            CNT: counts,
            AVG: averages,
            })
        # Construct the index
        tick_names = [m.method for m in self.methods]
        tick_vals = list(range(len(tick_names)))
        num_entry = len(df)
        if np.mod(num_entry, len(tick_names)) != 0:
            raise RuntimeError("Entries should be len(methods)*num_latincube")
        index_names = []
        for idx in range(num_entry//len(tick_names)):
            index_names.extend(["%s--%d" % (n, idx+1) for n in tick_names])
        df.index = index_names
        _, axes = plt.subplots(1, 3, figsize=(15, 5))
        df.plot.bar(y=TOT, ax=axes[0], title="Total time",
              xlabel="method", fontsize=18)
        df.plot.bar(y=AVG, ax=axes[1], title="Average time",
              xlabel="method", fontsize=18)
        df.plot.bar(y=CNT, ax=axes[2], title="Number calls",
              xlabel="method", fontsize=18)
        for idx in range(3):
            axes[idx].set_xticks(tick_vals)
            axes[idx].set_xticklabels(tick_names,
                  rotation=25, fontsize=18)
        if is_plot:
            plt.show()
        return df

    def plotQuality(self, is_plot=True):
        """
        Plots the quality results

        Parameters
        ----------
        is_plot: bool (plot the output)

        Returns
        -------
        dict
            key: method name
            value: list-float (residual sum of squares)
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
        return dct

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

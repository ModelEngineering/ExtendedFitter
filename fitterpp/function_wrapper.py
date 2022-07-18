"""Abstraction for a function that has parameters to fit."""


import time


class FunctionWrapper:
    # Wraps a function used for fitting.

    def __init__(self, function, is_collect=False):
        """
        Parameters
        ----------
        function: function
            function callable by lmfit.Minimizer
               argument
                   lmfit.Parameter
                   isRawData - boolean to indicate return total SSQ
               returns: np.array (residuals)
        is_collect: bool
            collect performance statistics on function execution
        """
        self._function = function
        self.is_collect = is_collect
        # Results
        self.perfStatistics = []  # durations of function executions
        self.rssqStatistics = []  # residual sum of squares, a quality measure
        self.rssq = 10e10
        self.bestParamDct = None

    @staticmethod
    def calcSSQ(arr):
        return sum(arr**2)

    def execute(self, params, **kwargs):
        """
        Runs the function using its keyword arguments.
        Accumulates statistics.

        Parameters
        ----------
        params: lmfit.Parameters
        kwargs: dict

        Returns
        -------
        array-float
        """
        if self.is_collect:
            startTime = time.time()
        result = self._function(params, **kwargs)
        if self.is_collect:
            duration = time.time() - startTime
        rssq = FunctionWrapper.calcSSQ(result)
        if rssq < self.rssq:
            self.rssq = rssq
            self.bestParamDct = dict(params.valuesdict())
        if self.is_collect:
            self.perfStatistics.append(duration )
            self.rssqStatistics.append(rssq)
        return result

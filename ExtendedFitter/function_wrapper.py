"""Abstraction for a function that has parameters to fit."""


from ExtendedFitter.logs import Logger
from ExtendedFitter import util
from ExtendedFitter import constants as cn

import copy
import lmfit
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time


class _FunctionWrapper():
    """Wraps a function used for optimization."""

    def __init__(self, function, isCollect=False):
        """
        Parameters
        ----------
        function: function
            function callable by lmfit.Minimizer
               argument
                   lmfit.Parameter
                   isRawData - boolean to indicate return total SSQ
               returns: np.array
        isCollect: bool
            collect performance statistics on function execution
        """
        self._function = function
        self._isCollect = isCollect
        # Results
        self.perfStatistics = []  # durations of function executions
        self.rssqStatistics = []  # residual sum of squares, a quality measure
        self.rssq = 10e10
        self.bestParamDct = None

    @staticmethod
    def _calcSSQ(arr):
        return sum(arr**2)

    def execute(self, params, **kwargs):
        if self._isCollect:
            startTime = time.time()
        result = self._function(params, **kwargs)
        if self._isCollect:
            duration = time.time() - startTime
        rssq = _FunctionWrapper._calcSSQ(result)
        if rssq < self.rssq:
            self.rssq = rssq
            self.bestParamDct = dict(params.valuesdict())
        if self._isCollect:
            self.perfStatistics.append(duration - startTime)
            self.rssqStatistics.append(rssq)
        return result

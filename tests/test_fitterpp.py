# -*- coding: utf-8 -*-
"""
Created on July 4, 2022

@author: joseph-hellerstein
"""

import fitterpp.constants as cn
from fitterpp.fitterpp import Fitterpp
from fitterpp import util
from fitterpp.logs import Logger

import collections
import matplotlib
import numpy as np
import lmfit
import unittest

try:
    matplotlib.use('TkAgg')
except ImportError:
    pass


IGNORE_TEST = False
IS_PLOT = False
XKEY = "x"
YKEY = "y"
INITIAL_VALUE = 1
MIN_VALUE = -4
MAX_VALUE = 10
POINT_DCT = {XKEY: 4, YKEY:8}
POINT_VALUES = list(POINT_DCT.values())

########## FUNCTIONS #################
def calcPointResiduals(params:lmfit.Parameters, minArgs:float=POINT_VALUES):
    """
    Implements a function used for optimization with fitterpp.

    Parameters
    ----------
    params: lmfit.Parameters
    minArgs: tupe-float
    
    Returns
    -------
    np.array
        residuals

    Usage
    -----
    residuals = calcPointResiduals(params)
    """
    xValue = params.valuesdict()[XKEY]
    yValue = params.valuesdict()[YKEY]
    residuals = np.array([(xValue-POINT_VALUES[0]), (yValue-POINT_VALUES[1])])
    return np.array(residuals)


def calcPointResidualsWithoutRaw(params:lmfit.Parameters, minArgs:float=POINT_VALUES):
    return calcPointResiduals(params, minArgs=minArgs)
        

################ TEST CLASSES #############
class TestFitterpp(unittest.TestCase):

    def setUp(self):
        self.function = calcPointResiduals
        self.params = lmfit.Parameters()
        self.params.add(XKEY, value=INITIAL_VALUE,
              min=MIN_VALUE, max=MAX_VALUE)
        self.params.add(YKEY, value=INITIAL_VALUE,
              min=MIN_VALUE, max=MAX_VALUE)
        self.methods = Fitterpp.mkFitterMethod()
        self.fitter = Fitterpp(self.function, self.params,
              methods=self.methods)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsNone(self.fitter.duration)

    def testExecute(self):
        if IGNORE_TEST:
            return
        self.checkResult(fitter=self.fitter)

    def testMkFitterMethod(self):
        if IGNORE_TEST:
            return
        def test(results):
            for result in results:
                self.assertTrue(isinstance(result.method, str))
                self.assertTrue(isinstance(result.kwargs, dict))
                self.assertTrue(cn.MAX_NFEV in result.kwargs.keys())
        #
        test(Fitterpp.mkFitterMethod())
        test(Fitterpp.mkFitterMethod(methodNames="aa"))
        test(Fitterpp.mkFitterMethod(methodNames=["aa", "bb"]))
        test(Fitterpp.mkFitterMethod(methodNames=["aa", "bb"],
              methodKwargs={cn.MAX_NFEV: 10}))

    def checkResult(self, fitter=None):
        if fitter is None:
            fitter = self.fitter
        fitter.execute()
        values = fitter.final_params.valuesdict().values()
        for expected, actual in zip(POINT_VALUES, values):
            self.assertLess(np.abs(expected-actual), 0.01)

    def testPlotPerformance(self):
        if IGNORE_TEST:
            return
        methods = Fitterpp.mkFitterMethod(
              methodNames=[cn.METHOD_LEASTSQ, cn.METHOD_DIFFERENTIAL_EVOLUTION])
        fitter = Fitterpp(self.function, self.params, methods,
              is_collect=True)
        fitter.execute()
        fitter.plotPerformance(isPlot=IS_PLOT)

    def testPlotQuality(self):
        if IGNORE_TEST:
            return
        methods = Fitterpp.mkFitterMethod(
              methodNames=[cn.METHOD_DIFFERENTIAL_EVOLUTION, cn.METHOD_LEASTSQ])
              #methodNames=[cn.METHOD_LEASTSQ, cn.METHOD_DIFFERENTIAL_EVOLUTION])
        fitter = Fitterpp(self.function, self.params, methods,
              is_collect=True)
        fitter.execute()
        fitter.plotQuality(isPlot=IS_PLOT)

    def testReport(self):
        if IGNORE_TEST:
            return
        self.checkResult(fitter=self.fitter)
        report = self.fitter.report()
        self.assertTrue(cn.METHOD_LEASTSQ in report)
        if IS_PLOT:
            print(report)
        

if __name__ == '__main__':
    unittest.main()

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
import pandas as pd
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
PARABOLA_PRMS = {"mult": 2, "center": 10}
SIZE = 20
XVALUES = range(SIZE)
DATA_DF = pd.DataFrame({
      YKEY: np.array([PARABOLA_PRMS["mult"]*(n - PARABOLA_PRMS["center"])**2
           + 1*np.random.rand() for n in XVALUES])
      })

########## FUNCTIONS #################
def calcParabola(center=0, mult=1, xvalues=XVALUES):
    """
    Calculates a parabola of a specified size.

    Parameters
    ----------
    center: float (parameter to fit)
    mult: float (parameter to fit)
    xvalues: int (not a fitted parameter)
    
    Returns
    -------
    DataFrame
        Columns: x, v
    """
    estimates = np.array([mult*(n - center)**2 for n in xvalues])
    return pd.DataFrame({XKEY: XVALUES, YKEY: estimates})


################ TEST CLASSES #############
class TestFitterpp(unittest.TestCase):

    def setUp(self):
        self.function = calcParabola
        self.params = lmfit.Parameters()
        for key, value in PARABOLA_PRMS.items():
            self.params.add(key, value=0, min=0, max=10*value)
        self.methods = Fitterpp.mkFitterMethod(
              method_names=["differential_evolution"], max_fev=1000)
        self.fitter = Fitterpp(self.function, self.params, DATA_DF,
              methods=self.methods)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsNone(self.fitter.duration)

    def testMkFitterFunction(self):
        if IGNORE_TEST:
            return
        func = self.fitter._mkFitterFunction()
        result = func(self.params)
        self.assertEqual(len(result), len(DATA_DF))

    def testExecute(self):
        if IGNORE_TEST:
            return
        def test(fitter):
            for key, value in PARABOLA_PRMS.items():
                true_value = value
                fitted_value = fitter.final_params.valuesdict()[key]
                self.assertLess(np.abs(true_value - fitted_value), 0.1)
        params = lmfit.Parameters()
        for key, value in PARABOLA_PRMS.items():
            params.add(key, value=0, min=0, max=10*value)
        methods = Fitterpp.mkFitterMethod(
              method_names=["differential_evolution"],
              max_fev=1000)
        fitter = Fitterpp(calcParabola, params, DATA_DF, methods=methods)
        fitter.execute()
        test(fitter)

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
        test(Fitterpp.mkFitterMethod(method_names="aa"))
        test(Fitterpp.mkFitterMethod(method_names=["aa", "bb"]))
        test(Fitterpp.mkFitterMethod(method_names=["aa", "bb"],
              method_kwargs={cn.MAX_NFEV: 10}))

    def testPlotPerformance(self):
        if IGNORE_TEST:
            return
        methods = Fitterpp.mkFitterMethod(
              method_names=[cn.METHOD_LEASTSQ,
              cn.METHOD_DIFFERENTIAL_EVOLUTION])
        fitter = Fitterpp(self.function, self.params, DATA_DF,
              methods=methods, is_collect=True)
        fitter.execute()
        fitter.plotPerformance(is_plot=IS_PLOT)

    def testPlotQuality(self):
        if IGNORE_TEST:
            return
        methods = Fitterpp.mkFitterMethod(
              method_names=[cn.METHOD_DIFFERENTIAL_EVOLUTION,
                    cn.METHOD_LEASTSQ])
        fitter = Fitterpp(self.function, self.params, DATA_DF,
              methods=methods,
              is_collect=True)
        fitter.execute()
        fitter.plotQuality(is_plot=IS_PLOT)

    def testReport(self):
        if IGNORE_TEST:
            return
        self.fitter.execute()
        report = self.fitter.report()
        self.assertTrue(cn.METHOD_DIFFERENTIAL_EVOLUTION in report)
        if IS_PLOT:
            print(report)
        

if __name__ == '__main__':
    unittest.main()

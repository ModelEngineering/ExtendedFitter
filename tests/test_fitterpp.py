# -*- coding: utf-8 -*-
"""
Created on July 4, 2022

@author: joseph-hellerstein
"""

import fitterpp.constants as cn
from fitterpp.fitterpp import Fitterpp, DFIntersectionFinder
from fitterpp import util
from fitterpp.logs import Logger
from tests import helpers

import collections
import copy
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
YKEY = "y"
ROW_KEY = "row_key"
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
DATA_DF.index = XVALUES
PARAMS = lmfit.Parameters()
for key, value in PARABOLA_PRMS.items():
    PARAMS.add(key, value=0, min=0, max=10*value)

########## FUNCTIONS #################
def calcParabola(center=0, mult=1, xvalues=XVALUES, is_dataframe=True):
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
        Index is row key
    """
    estimates = np.array([mult*(n - center)**2 for n in xvalues])
    if is_dataframe:
        result = pd.DataFrame({ROW_KEY: XVALUES, YKEY: estimates})
        result = result.set_index(ROW_KEY)
    else:
        result = np.array([estimates])
        result = np.reshape(result, (len(estimates), 1))
    return result


################ TEST CLASSES #############
class TestDataframeCommon(unittest.TestCase):

    def setUp(self):
        data = list(range(6))
        data1 = data[2:]
        data2 = data[:-2]
        self.df1 = pd.DataFrame({"a": data1, 
              "c": range(len(data1)),
              "b": range(len(data1))})
        self.df1 = self.df1.set_index("a")
        self.df2 = pd.DataFrame({"a": data2,
              "b": range(len(data1)),
              "d": range(len(data2))})
        self.df2 = self.df2.set_index("a")
        self.common1 = DFIntersectionFinder(self.df1, self.df2)
        self.common2 = DFIntersectionFinder(self.df2, self.df1)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertTrue(helpers.isArrayEqual(self.common1.row_idxs, [0, 1]))
        self.assertTrue(helpers.isArrayEqual(self.common1.column_idxs, [1]))
        self.assertTrue(helpers.isArrayEqual(self.common2.row_idxs, [2, 3]))
        self.assertTrue(helpers.isArrayEqual(self.common2.column_idxs, [0]))

    def testIsCorrectShape(self):
        if IGNORE_TEST:
            return
        df = calcParabola(center=2, mult=2, is_dataframe=True)
        methods = Fitterpp.mkFitterMethod(
              method_names=["differential_evolution"], max_fev=1000)
        fitter = Fitterpp(calcParabola, PARAMS, DATA_DF, methods=methods)
        arr = calcParabola(center=2, mult=2, is_dataframe=False)
        self.assertTrue(fitter.function_common.isCorrectShape(arr))
        #
        shape = np.shape(arr)
        new_arr = np.reshape(arr, (shape[1], shape[0]))
        self.assertFalse(fitter.function_common.isCorrectShape(new_arr))


class TestFitterpp(unittest.TestCase):

    def setUp(self):
        self.function = calcParabola
        self.params = copy.deepcopy(PARAMS)
        self.methods = Fitterpp.mkFitterMethod(
              method_names=["differential_evolution"], max_fev=1000)
        self.fitter = Fitterpp(self.function, self.params, DATA_DF,
              methods=self.methods)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertIsNone(self.fitter.duration)
        result = self.fitter.user_function(center=1, mult=2, is_dataframe=False)
        self.assertTrue([x == y for x, y in zip(result[:, 0], XVALUES)])

    def testMkFitterFunction(self):
        if IGNORE_TEST:
            return
        func = self.fitter._mkFitterFunction()
        result1 = func(self.params)
        self.assertEqual(len(result1), len(DATA_DF))
        #
        kwargs = self.fitter.makeKwargs(self.params)
        arr = self.fitter.user_function(is_dataframe=False, **kwargs)
        self.fitter.function_common.isCorrectShape(arr)

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

# -*- coding: utf-8 -*-
"""
Created on Tue Feb 9, 2021

@author: joseph-hellerstein
"""

import fitterpp.constants as cn
from fitterpp.function_wrapper import FunctionWrapper

import collections
import numpy as np
import lmfit
import unittest


IGNORE_TEST = False
IS_PLOT = False
XKEY = "x"
YKEY = "y"
INITIAL_VALUE = 1
MIN_VALUE = -4
MAX_VALUE = 10
POINT_DCT = {XKEY: 4, YKEY:8}
POINT = list(POINT_DCT.values())

########## FUNCTIONS #################
def calcResiduals(params:lmfit.Parameters, minArgs:float=POINT):
    """
    Calculates the difference beween POINT and a point provided as parameters.

    Parameters
    ----------
    params: lmfit.Parameters (x and y values)
    minArgs: tupe-float
    
    Returns
    -------
    np.array
        residuals

    Usage
    -----
    residuals = calcResiduals(params)
    """
    xValue = params.valuesdict()[XKEY]
    yValue = params.valuesdict()[YKEY]
    residuals = np.array([xValue-POINT[0], yValue-POINT[1]])
    return residuals


def calcResidualsWithoutRaw(params:lmfit.Parameters, minArgs:float=POINT):
    return calcResiduals(params, minArgs=minArgs)
        

################ TEST CLASSES #############
class TestFunctionWrapper(unittest.TestCase):

    def setUp(self):
        self.function = calcResiduals
        self.params = lmfit.Parameters()
        self.params.add(XKEY, value=INITIAL_VALUE, min=MIN_VALUE, max=MAX_VALUE)
        self.params.add(YKEY, value=INITIAL_VALUE, min=MIN_VALUE, max=MAX_VALUE)
        self.wrapper =  FunctionWrapper(self.function)

    def testConstructor(self):
        if IGNORE_TEST:
            return
        self.assertGreater(self.wrapper.rssq, 0)

    def testCalcSSQ(self):
        if IGNORE_TEST:
            return
        arr = np.array(range(3))
        self.assertEqual(self.wrapper.calcSSQ(arr), 5)

    def testExecute(self):
        if IGNORE_TEST:
            return
        result = self.wrapper.execute(self.params)
        self.assertEqual(result[0], INITIAL_VALUE - POINT[0])
        
        

if __name__ == '__main__':
    unittest.main()

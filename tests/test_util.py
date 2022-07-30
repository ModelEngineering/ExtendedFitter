"""
Created on Nov 11, 2020

@author: joseph-hellerstein
"""

from fitterpp import util

import copy
import lmfit
import numpy as np
import os
import unittest


IGNORE_TEST = False
IS_PLOT = False


class TestFunctions(unittest.TestCase):

    def testCalcRelError(self):
        if IGNORE_TEST:
            return
        result = util.calcRelError(1.5, 1)
        self.assertEqual(result, 1.0/3)
        result = util.calcRelError(1.5, 1, isAbsolute=False)
        self.assertEqual(result, -1.0/3)
        #
        result = util.calcRelError(0, 1)
        self.assertTrue(np.isnan(result))

    def testFilterOutliersFromZero(self):
        if IGNORE_TEST:
            return
        MAX_SL = 0.1
        SIZE = 1000
        def test(data, baseData):
            newData = util.filterOutliersFromZero(data, MAX_SL)
            diff = set(baseData).symmetric_difference(newData)
            self.assertEqual(len(diff), 0)
        #
        baseData = [1, 1.1, 1.2, 1.3, 1.4]
        data = list(baseData)
        data.insert(1, 10)
        data.insert(3, -2)
        test(data, baseData)
        #
        data = np.random.random(SIZE)
        test(data, data)
        #
        baseData = list(data)
        data = list(baseData)
        data.insert(SIZE-10, 10)
        data.insert(SIZE-10, 20)
        test(data, baseData)

    def testCopyObject(self):
        if IGNORE_TEST:
            return
        class Sample(object):
            def __init__(self, a=0):
                self.a = a
                self.b = self.a + 1

        class SampleCopy(Sample):
            def copy(self):
                new = SampleCopy(a=self.a)
                new.b = self.a + 1
                return new
        #
        def test(obj):
            newObj = util.copyObject(obj)
            for item in ["a", "b"]:
                true = obj.__getattribute__(item) ==newObj.__getattribute__(item)
                if isinstance(true, np.ndarray):
                    true = all(true)
                self.assertTrue(true)
        #
        for cls in [Sample, SampleCopy]:
            test(cls(np.array(range(4))))
            test(cls(1))

    def testPpDct(self):
        if IGNORE_TEST:
            return
        SIZE = 10
        dct = {n: str(n) for n in range(SIZE)}
        result = util.ppDict(dct)
        self.assertEqual(result.count("\n"), SIZE-1)

    def testDictToParameters(self):
        if IGNORE_TEST:
            return
        dct = {"a": 3, "b": 4}
        parameters = util.dictToParameters(dct)
        self.assertTrue(isinstance(parameters, lmfit.Parameters))
        parameters = util.dictToParameters(dct, value_frac=0.5)
        self.assertTrue(np.isclose(parameters.valuesdict()["a"], dct["a"]*0.5))
         


if __name__ == '__main__':
    unittest.main()

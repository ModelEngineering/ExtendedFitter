# -*- coding: utf-8 -*-
"""
Created on July 21, 2022

@author: joseph-hellerstein
"""

from numpy.testing import assert_array_equal
import numpy as np

def isArrayEqual(arr1, arr2):
    """
    Parameters
    ----------
    arr1: np.array (or array-like)
    arr2: np.array (or array-like)
    
    Returns
    -------
    bool
    """
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    try:
        assert_array_equal(arr1, arr2)
        return True
    except:
        return False

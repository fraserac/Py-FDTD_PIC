# -*- coding: utf-8 -*-
"""
Created on Thu May 13 15:15:41 2021

@author: Fraser
"""
import numpy as np
import numba
from numba import njit as nj
from numba import int32, float32, int64, float64, boolean
from numba.typed import Dict
from numba import jit
import time


forLocs = {"a" : float64, "A" : float64, "be" : float64[:], "b":float64[:], "x" : float64, "yArr" : float64[:]}
#@numba.vectorize(nopython =True)
#@nj(locals=forLocs)
def testFuncNonCubic(yArr, be): 
   for j in range(len(be)):
       b = np.roots(yArr[:4])
       for i in range(len(b)):
           if b[i] >=0:
               if np.imag(b[i])==0:  # if there's more than one positive?
                   be[j] = b[i]
                   break
               else:
                   be[j] = 0.0
           else:
               be[j] =0.0
            
   return be
    
@numba.guvectorize([(float64[:], float64[:], float64[:])], '(n),(n)->(n)', nopython=True, fastmath=True)
def testVectorize(yArr, be, res):
    for j in range(len(be)):
        b = np.roots(yArr[:4])
        for i in range(len(b)):
            if b[i] >=0:
                if np.imag(b[i])==0: 
                    res[j] = b[i]
                    break
                else:
                    res[j] = 0.0
            
            else:
                res[j] =0.0

be = np.zeros(10000)
yArr = np.array([-81, 4, 2e-5, 2e-16]) 
yArr = np.append(yArr,np.zeros(9996))   
a = time.perf_counter()    
#A= testFuncNonCubic(yArr, be)

A = testVectorize(yArr, be)
b = time.perf_counter()
print(b-a, "\n", A)
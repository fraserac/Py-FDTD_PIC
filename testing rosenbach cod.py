# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:23:22 2020

@author: Owner
"""

#from fluidpythran import boost



from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plot
import numpy as np
import time as tim
import numba as num
 



maxim= 10000

amount=100
clock =  np.zeros(amount)
xAvg = 0.0
x = 0.0

ticc = 0.0
tocc =0.0




#@num.jit
def runner(m, a, x):
   
    avg = 0.0
    maxime =  1
    amounte =1
    x= myfunc(1,1.0, amounte, maxime, x)
    xAvg = 0.0
    print(x, " x in myfunc skeleton run")
    for k in range(0, a):    
        xl = 0
        tic = 0.0
        toc = 0.0
        tic =tim.perf_counter()   
        xAvg=myfunc(5.5, 9, a, m, xl)
        toc = tim.perf_counter()
        clock[k]= toc-tic
    avg = 0.0
    avg = np.average(clock)
    
    
    print(avg, xAvg, "avg time, value from big run")
    return clock
    
@num.njit    
def myfunc(a, b, amountee, maxee, xl):
    for j in range(0,amountee):
        for i in range(0, maxee):
            if j == 20:
                xl  = xl + a*b
    return xl
            
ticc = tim.perf_counter()           
clock = runner(maxim, amount, x)
tocc = tim.perf_counter()
print(tocc-ticc)

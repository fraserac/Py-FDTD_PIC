# -*- coding: utf-8 -*-
"""
@author: Fraser
"""
import numpy as np
import scipy.constants as sci
#from MasterController import *


UpHyMat =[]
UpExMat =[]
Ex =[]#,dtype=complex)
Hy=[]#,dtype=complex)
Ex_History= [[]]
Hy_History=[[]]

permit_0 = sci.epsilon_0
permea_0 = sci.mu_0
epsRe =9
epsIm = 0
muRe = 1
muIm = 0

c0 = sci.speed_of_light
freq_in = 4e9

### WILL NEED MAX FREQ WHEN HIGHER HARMONICS ARE PRESENT
lamMin = c0/(np.sqrt(abs(epsRe)*abs(muRe))*freq_in)
Nlam = 20
dz =lamMin/Nlam  
delT = (dz)/(c0)   # LOOK INTO HOW DZ AND DELT ARE USED, COURANT NO?
courantNo = c0*delT/dz
CharImp =np.sqrt(permea_0)/np.sqrt(permit_0)
MaterialFrontEdge = 100  # Discrete tile where material begins (array index)
MaterialRearEdge = 130


Nz = 200   #Grid size

timeSteps =2**8
t=np.arange(0, timeSteps, 1)*(delT)  # FOR VERIFICATION PLOTTING, EVALUATE IN CLASS

nzsrc = 50 # Position of source 



x1ColBe=[[]]*timeSteps 
x1ColAf=[[]]*timeSteps

class Params(object):
    def __init__(self, epsRe, muRe, f_in, lMin, nlm, dz, dt, crntNo, matRear, matFront, gridNo, timeSteps):
        self.permit_0 = 8.854187817620389e-12#
        self.permea_0 =1.2566370614359173e-06
        self.c0 = 299792458.0
        self.epsRe = epsRe
        self.muRe = muRe
        self.freq_in = f_in
        self.lamMin = lMin
        self.Nlam = Nlm
        self.dz = dz
        self.delT = dt
        self.courantNo= crntNo
        self.MaterialFrontEdge = matFront
        self.MaterialRearEdge = matRear
        self.Nz = gridNo
        self.timeSteps = timeSteps
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(', "Real Epsilon:",f'{self.epsRe!r}, "Real Mu", 'f'{self.muRe!r})
    
    def __str__(self):
        return 'Class containing all values that remain constant throughout a sim'
    
    #set up class for FDTD parameters, so mastercontroller is unit testable
    
    
    
    

# -*- coding: utf-8 -*-
"""
@author: Fraser
"""
import numpy as np
import scipy.constants as sci
#from MasterController import *


permit_0 = sci.epsilon_0
permea_0 = sci.mu_0
epsRe =9
epsIm = 0
muRe = 1
muIm = 0

c0 = sci.speed_of_light
freq_in = 1e9

### WILL NEED MAX FREQ WHEN HIGHER HARMONICS ARE PRESENT
lamMin = c0/freq_in
Nlam = 20*np.sqrt(epsRe*muRe)
dz =lamMin/Nlam  
courantNo = 1
delT = (courantNo*dz)/(c0)   # LOOK INTO HOW DZ AND DELT ARE USED, COURANT NO?

CharImp =np.sqrt(permea_0)/np.sqrt(permit_0)
MaterialFrontEdge = 100  # Discrete tile where material begins (array index)
MaterialRearEdge = 170
period = 1/freq_in

Nz = 200   #Grid size

timeSteps =2**8
t=np.arange(0, timeSteps, 1)*(delT)  # FOR VERIFICATION PLOTTING, EVALUATE IN CLASS

nzsrc = 10 # Position of source 
x1Loc = 80

eLoss =0.02
mLoss = 0
eSelfCo = (1-eLoss)/(1+eLoss)#
eHcompsCo = 1/(1+eLoss)

x1ColBe=[[]]*timeSteps 
x1ColAf=[[]]*timeSteps
UpHySelf= np.zeros(Nz)
UpExSelf = np.zeros(Nz)
UpExHcompsCo =np.zeros(Nz)
UpExMat =np.zeros(Nz)
Ex =np.zeros(Nz)#,dtype=complex)
Hy=np.zeros(Nz)#,dtype=complex)
Ex_History= [[]]*timeSteps
Hy_History= [[]]*timeSteps
Hys = []
Exs = []

epsilon = np.ones(Nz)
mu = np.ones(Nz)

    
    #set up class for FDTD parameters, so mastercontroller is unit testable
    
    
    
    

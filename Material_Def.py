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
courantNo = 1   # LOOK INTO 2D VERSION
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

eLoss =0   # sigma e* delT/2*epsilon
mLoss = 0.02
eSelfCo = (1-eLoss)/(1+eLoss)#
eHcompsCo = 1/(1+eLoss)
hSelfCo = (1-mLoss)/(1+mLoss)
hEcompsCo = 1/(1+mLoss)


x1ColBe=[[]]*timeSteps 
x1ColAf=[[]]*timeSteps
UpHySelf= np.zeros(Nz)
UpHyEcompsCo = np.zeros(Nz)
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
pmlWidth = 10






###PML STUFF PARAMS

kappaMax = 1 # 'Stretching co-ordinate of pml, to minimise numerical dispersion set it as 1' : DOI: 10.22190/FUACR1703229G see conclusion
r_scale = 3 # Within ideal bounds see Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17 (scaling power is called 'm' )
r_a_scale=1.05
sigmaOpt  =(0.8*(r_scale+1))/(CharImp*dz)#Optimal value of pml conductivity at far end of pml: DOI: 10.22190/FUACR1703229G see equation 13
sigmaEMax = sigmaOpt # Within ideal bounds for value, : Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
sigmaHMax = sigmaOpt # See International Journal of Computer Science and Network Security, VOL.18 No.12, December 2018, page 4 right hand side.


alphaMax= 0.3 # with bounds of ideal cpml alpha max, complex frequency shift parameter, Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17


# VARIABLES
kappa_Ex =1
kappa_Hy = 1
psi_Ex =np.zeros(Nz)
psi_Hy = np.zeros(Nz)
alpha_Ex= np.zeros(Nz)
alpha_Hy= np.zeros(Nz)
sigmaEx =np.zeros(Nz)   # specific spatial value of conductivity 
sigmaHy = np.zeros(Nz)
beX =np.zeros(Nz)#
bmY =np.zeros(Nz)#np.exp(-(sigmaHy/(permea_0*kappa_Hy) + alpha_Hy/permea_0 )*delT)
ceX = np.zeros(Nz)
cmY = np.zeros(Nz)

    
    #set up class for FDTD parameters, so mastercontroller is unit testable
    
    
    
    

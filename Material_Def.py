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

period = 1/freq_in

pmlWidth = 40
domainSize = 200
dimen =1
Nz = domainSize +2*dimen*pmlWidth   #Grid size

timeSteps =800
t=np.arange(0, timeSteps, 1)*(delT)  # FOR VERIFICATION PLOTTING, EVALUATE IN CLASS

nzsrcFromPml = 20
nzsrc = nzsrcFromPml + pmlWidth # Position of source 
x1Loc = 80
MaterialDistFromPml = 50
MaterialFrontEdge = MaterialDistFromPml + pmlWidth  # Discrete tile where material begins (array index)
MaterialWidth = 20
MaterialRearEdge = MaterialFrontEdge + MaterialWidth
eLoss =0   # sigma e* delT/2*epsilon
mLoss = 0
eSelfCo = (1-eLoss)/(1+eLoss)#
eHcompsCo = 1/(1+eLoss)
hSelfCo = (1-mLoss)/(1+mLoss)
hEcompsCo = 1/(1+mLoss)



x1ColBe=[[]]*timeSteps 
x1ColAf=[[]]*timeSteps
UpHySelf= np.ones(Nz)
UpHyEcompsCo = np.ones(Nz)
UpExSelf = np.ones(Nz)
UpExHcompsCo =np.ones(Nz)
UpExMat =np.zeros(Nz)
UpHyMat = np.zeros(Nz)
Ex =np.zeros(Nz)#,dtype=complex)
Hy=np.zeros(Nz)#,dtype=complex)
Ex_History= [[]]*timeSteps
Hy_History= [[]]*timeSteps
Psi_Ex_History= [[]]*timeSteps
Psi_Hy_History= [[]]*timeSteps
Hys = []
Exs = []

epsilon = np.ones(Nz)
mu = np.ones(Nz)







###PML STUFF PARAMS

kappaMax =15 # 'Stretching co-ordinate of pml, to minimise numerical dispersion set it as 1' : DOI: 10.22190/FUACR1703229G see conclusion
r_scale = 3# Within ideal bounds see Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17 (scaling power is called 'm' )
r_a_scale=1
sigmaEMax = 0.75 * (0.8*(r_scale+1)/(dz*(permea_0/permit_0)**0.5))#1.1*sigmaOpt # Within ideal bounds for value, : Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
sigmaHMax = sigmaEMax#1.1*sigmaOpt # See International Journal of Computer Science and Network Security, VOL.18 No.12, December 2018, page 4 right hand side.
sigmaOpt  =sigmaEMax*1.1#Optimal value of pml conductivity at far end of pml: DOI: 10.22190/FUACR1703229G see equation 13


alphaMax= 0.024 # with bounds of ideal cpml alpha max, complex frequency shift parameter, Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17


# VARIABLES
kappa_Ex =np.zeros(Nz)
kappa_Hy = np.zeros(Nz)
psi_Ex =np.zeros(Nz)
psi_Hy = np.zeros(Nz)
alpha_Ex= np.zeros(Nz)
alpha_Hy= np.zeros(Nz)
sigma_Ex =np.zeros(Nz)   # specific spatial value of conductivity 
sigma_Hy = np.zeros(Nz)
beX =np.zeros(Nz)#
bmY =np.zeros(Nz)#np.exp(-(sigmaHy/(permea_0*kappa_Hy) + alpha_Hy/permea_0 )*delT)
ceX = np.zeros(Nz)
cmY = np.zeros(Nz)

Ca = np.zeros(Nz)
Cb = np.zeros(Nz)
Cc = np.zeros(Nz)
C1 = np.zeros(Nz)
C2 = np.zeros(Nz)
C3 = np.zeros(Nz)

eLoss_CPML =np.zeros(Nz)   # sigma e* delT/2*epsilon
mLoss_CPML = np.zeros(Nz)
den_Hydz = np.zeros(Nz)
den_Exdz = np.zeros(Nz)    
    #set up class for FDTD parameters, so mastercontroller is unit testable
    
    
    
    

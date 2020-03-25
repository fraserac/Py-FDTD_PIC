# -*- coding: utf-8 -*-
"""
@author: Fraser
"""
import numpy as np
import scipy.constants as sci
import sys
#from MasterController import *


permit_0 = sci.epsilon_0
permea_0 = sci.mu_0
epsRe =9
epsIm = 0
muRe = 1
muIm = 0

c0 = sci.speed_of_light
freq_in = 1e11

### WILL NEED MAX FREQ WHEN HIGHER HARMONICS ARE PRESENT
lamMin = c0/freq_in
Nlam = np.floor(20*np.sqrt(epsRe*muRe))
dz =lamMin/Nlam  
courantNo = 1   # LOOK INTO 2D VERSION
delT = (courantNo*dz)/(c0)
  # LOOK INTO HOW DZ AND DELT ARE USED, COURANT NO?

CharImp =np.sqrt(permea_0)/np.sqrt(permit_0)

period = 1/freq_in

pmlWidth =50 +int(30*freq_in/(1e10))
print('pmlWidth = ' , pmlWidth)
if (pmlWidth >= 500):
    print('pmlWidth too big', pmlWidth)
    sys.exit()
domainSize = 200 +int(5*freq_in/1e10)
print(domainSize, 'domainSize')
dimen =1
nonInt =  False

Nz = domainSize +2*dimen*pmlWidth   #Grid size
minRange = 500
maxRange = 1000
for N in range(minRange,maxRange):
    check = (freq_in*N)/(1/delT)
    #print(check)
    if check/int(check) ==1:
        timeSteps = N 
        break
    elif N == maxRange-1:
       print('Could not find timestep that allowed freq_in to fall on an integer frequency bin index with range provided.') 
       #sys.exit()
       nonInt = True
       
if(nonInt == True):       
    checkNear = (freq_in*minRange)/(1/delT)
    for N in range(minRange, maxRange):
        dummyCheck = (freq_in*N)/(1/delT)
        if dummyCheck/int(dummyCheck) < checkNear/int(checkNear):
            checkNear = dummyCheck
            print(checkNear)
        
    timeSteps = N        
    
print('timesteps: ', timeSteps)

if(timeSteps >= 10000):
    print('timeSteps too large')
    sys.exit()
t=np.arange(0, timeSteps, 1)*(delT)  # FOR VERIFICATION PLOTTING, EVALUATE IN CLASS

nzsrcFromPml = int(lamMin/dz)+1
if nzsrcFromPml >= domainSize/2:
    print(nzsrcFromPml, 'src is too far into domain')
    sys.exit()
    
nzsrc = nzsrcFromPml + pmlWidth # Position of source 
x2LocBehindSrc = 10
if(x2LocBehindSrc >= nzsrcFromPml):
    print('The probe for fft is in the PML region')
    sys.exit()   

MaterialDistFromPml = 70
MaterialFrontEdge = MaterialDistFromPml + pmlWidth + domainSize/4  # Discrete tile where material begins (array index)
MaterialWidth = Nz - MaterialFrontEdge
MaterialRearEdge = MaterialFrontEdge + MaterialWidth
x1Loc = nzsrc+int((MaterialFrontEdge-nzsrc)/2)
x2Loc = nzsrc - (nzsrcFromPml-x2LocBehindSrc)
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

kappaMax =12 # 'Stretching co-ordinate of pml, to minimise numerical dispersion set it as 1' : DOI: 10.22190/FUACR1703229G see conclusion
r_scale = 5# Within ideal bounds see Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17 (scaling power is called 'm' )
r_a_scale=1
sigmaEMax =1.4*(0.8*(r_scale+1)/(dz*(permea_0/permit_0)**0.5))#1.1*sigmaOpt # Within ideal bounds for value, : Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
sigmaHMax = sigmaEMax#1.1*sigmaOpt # See International Journal of Computer Science and Network Security, VOL.18 No.12, December 2018, page 4 right hand side.
sigmaOpt  =sigmaEMax
#Optimal value of pml conductivity at far end of pml: DOI: 10.22190/FUACR1703229G see equation 13


alphaMax= 0.24 # with bounds of ideal cpml alpha max, complex frequency shift parameter, Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17


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
    
"""    
m = 3; ma = 1 ; 
sigZmax =  (0.8*(m+1)/(dxyz*(mu0/eps0*epsR)^0.5)); 
aZmax = 0.05; 
kZmax = 1.0; 
"""
    
    

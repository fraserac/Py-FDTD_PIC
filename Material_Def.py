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
epsRe =1
epsIm = 0
muRe = 1
muIm = 0

c0 = sci.speed_of_light
freq_in = 7.5e8
  

### WILL NEED MAX FREQ WHEN HIGHER HARMONICS ARE PRESENT
lamMin = (c0/freq_in)*10
print("LamMin ", lamMin)
Nlam = 400#np.floor(20*np.sqrt(epsRe*muRe))
dz =lamMin/Nlam  
courantNo = 1   # LOOK INTO 2D VERSION
delT =0.95/(c0*np.sqrt(1/(dz**2) +1/(dz**2)))
  # LOOK INTO HOW DZ AND DELT ARE USED, COURANT NO?

CharImp =np.sqrt(permea_0)/np.sqrt(permit_0)

period = 1/freq_in

pmlWidth =150 +int(30*freq_in/(1e10))
print('pmlWidth = ' , pmlWidth)
if (pmlWidth >= 500):
    print('pmlWidth too big', pmlWidth)
    sys.exit()
domainSize = 1000 +int(5*freq_in/1e9)
print(domainSize, 'domainSize')
dimen =1
nonInt =  False

Nz = domainSize +2*dimen*pmlWidth   #Grid size
minRange = 1000
maxRange = 2000
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
        if (dummyCheck/int(dummyCheck)) < (checkNear/int(checkNear)):
            checkNear = dummyCheck
            print(checkNear)
        
    timeSteps = N        
    
print('timesteps: ', timeSteps)

if(timeSteps >= 10000):
    print('timeSteps too large')
    sys.exit()
t=np.arange(0, timeSteps, 1)*(delT)  # FOR VERIFICATION PLOTTING, EVALUATE IN CLASS

nzsrcFromPml = int(lamMin/dz)+1
print(nzsrcFromPml, domainSize)
if nzsrcFromPml >= 0.75*domainSize:
    print(nzsrcFromPml, 'src is too far into domain')
    sys.exit()
    
nzsrc = nzsrcFromPml + pmlWidth # Position of source 
x2LocBehindSrc = 10
if(x2LocBehindSrc >= nzsrcFromPml):
    print('The probe for fft is in the PML region')
    sys.exit()   

MaterialDistFromPml = 2*int(lamMin/dz)+1
MaterialFrontEdge = MaterialDistFromPml + pmlWidth + domainSize/8  # Discrete tile where material begins (array index)
MaterialWidth = Nz - MaterialFrontEdge - domainSize/6
MaterialRearEdge = MaterialFrontEdge + MaterialWidth
x1Loc = nzsrc+int((MaterialFrontEdge-nzsrc)/2)
x2Loc = nzsrc - (nzsrcFromPml-x2LocBehindSrc)
eLoss =0   # sigma e* delT/2*epsilon
mLoss = 0
eSelfCo = (1-eLoss)/(1+eLoss)#
eHcompsCo = 1/(1+eLoss)
hSelfCo = (1-mLoss)/(1+mLoss)
hEcompsCo = 1/(1+mLoss)



x1ColBe=[[]] 
x1ColAf=[[]]
UpHySelf= np.ones(Nz)
UpHyEcompsCo = np.ones(Nz)
UpExSelf = np.ones(Nz)
UpExHcompsCo =np.ones(Nz)
UpExMat =np.zeros(Nz)
UpHyMat = np.zeros(Nz)
Ex =[]
Hy=[]
Ex_History= [[]]
Hy_History= [[]]
Psi_Ex_History= [[]]
Psi_Hy_History= [[]]
Hys = []
Exs = []

epsilon = []
mu = []

Dx = [[]]
My = [[]]




###PML STUFF PARAMS

kappaMax =12 # 'Stretching co-ordinate of pml, to minimise numerical dispersion set it as 1' : DOI: 10.22190/FUACR1703229G see conclusion
r_scale = 5.4# Within ideal bounds see Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17 (scaling power is called 'm' )
r_a_scale=1
sigmaEMax =1.4*(0.8*(r_scale+1)/(dz*(permea_0/permit_0)**0.5))#1.1*sigmaOpt # Within ideal bounds for value, : Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
sigmaHMax = sigmaEMax#1.1*sigmaOpt # See International Journal of Computer Science and Network Security, VOL.18 No.12, December 2018, page 4 right hand side.
sigmaOpt  =sigmaEMax
#Optimal value of pml conductivity at far end of pml: DOI: 10.22190/FUACR1703229G see equation 13


alphaMax= 0.24# with bounds of ideal cpml alpha max, complex frequency shift parameter, Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17


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
  




def matSetup(V,P, newFreq_in = freq_in):
    P.epsRe =1
    P.epsIm = 0
    P.muRe = 1
    P.muIm = 0
    P.freq_in = newFreq_in
    P.lamMin = (P.c0/P.freq_in)*10
    print("LamMin ", P.lamMin)
    P.Nlam = 400#np.floor(20*np.sqrt(P.epsRe*P.muRe))
    P.dz =P.lamMin/P.Nlam  
    P.courantNo = 1   # LOOK INTO 2D VERSION
    P.delT =  0.95/(P.c0*np.sqrt(1/(P.dz**2) +1/(P.dz**2)))
  

    P.period = 1/P.freq_in

    P.pmlWidth =150 +int(30*P.freq_in/(1e10))
    print('pmlWidth = ' , P.pmlWidth)
    if (P.pmlWidth >= 500):
        print('pmlWidth too big', P.pmlWidth)
        sys.exit()
    P.domainSize = 1000 +int(5*P.freq_in/1e9)
    print(P.domainSize, 'domainSize')
    dimen =1
    nonInt =  False

    P.Nz = domainSize +2*dimen*pmlWidth   #Grid size
    minRange = 1000
    maxRange = 2000
    for N in range(minRange,maxRange):
        check = (P.freq_in*N)/(1/P.delT)
        #print(check)
        if check/int(check) ==1:
            P.timeSteps = N 
            break
        elif N == maxRange-1:
           print('Could not find timestep that allowed freq_in to fall on an integer frequency bin index with range provided.') 
           #sys.exit()
           nonInt = True
       
    if(nonInt == True):       
        checkNear = (P.freq_in*minRange)/(1/P. delT)
        for N in range(minRange, maxRange):
            dummyCheck = (P.freq_in*N)/(1/P.delT)
            if (dummyCheck/int(dummyCheck)) < (checkNear/int(checkNear)):
                checkNear = dummyCheck
                print(checkNear)
        
    P.timeSteps = N        
    
    print('timesteps: ', P.timeSteps)
    
    if(P.timeSteps >= 10000):
        print('timeSteps too large')
        sys.exit()
    t=np.arange(0, P.timeSteps, 1)*(P.delT)  # FOR VERIFICATION PLOTTING, EVALUATE IN CLASS
    
    nzsrcFromPml = int(P.lamMin/P.dz)+1
    if nzsrcFromPml >= P.domainSize/2:
        print(nzsrcFromPml, 'src is too far into domain')
        sys.exit()
        
    P.nzsrc = nzsrcFromPml + pmlWidth # Position of source 
    x2LocBehindSrc = 10
    if(x2LocBehindSrc >= nzsrcFromPml):
        print('The probe for fft is in the PML region')
        sys.exit()   
    
    MaterialDistFromPml = 2*int(P.lamMin/P.dz)+1
    P.MaterialFrontEdge = MaterialDistFromPml + P.pmlWidth + P.domainSize/8  # Discrete tile where material begins (array index)
    MaterialWidth = P.Nz - P.MaterialFrontEdge - P.domainSize/6
    P.MaterialRearEdge = P.MaterialFrontEdge + MaterialWidth
    P.x1Loc = P.nzsrc+int((P.MaterialFrontEdge-P.nzsrc)/2)
    P.x2Loc = P.nzsrc - (nzsrcFromPml-x2LocBehindSrc)    
       

"""
    self.epsRe = epsRe
        self.muRe = muRe
        self.freq_in = freq_in
        self.lamMin = lMin
        self.Nlam = nlm
        self.dz = dz
        self.delT = delT
        self.courantNo= courantNo
        self.materialFrontEdge = matFront
        self.materialRearEdge = matRear
        self.Nz = gridNo
        self.timeSteps = timeSteps
        self.x1Loc = x1Loc
        self.x2Loc = x2Loc
        self.nzsrc = nzsrc
        self.period = period
        self.eLoss = eLoss
        self.eSelfCo = eSelfCo
        self.eHcompsCo = eHcompsCo
        self.mLoss = mLoss
        self.hSelfCo = hSelfCo
        self.hEcompsCo = hEcompsCo
        self.pmlWidth = pmlWidth


        self.kappaMax=kappaMax
        self.sigmaEMax=sigmaEMax
        self.sigmaHMax=sigmaHMax
        self.sigmaOpt=sigmaOpt
        self.alphaMax=alphaMax
        self.r_scale=r_scale
        self.r_a_scale=r_a_scale

"""
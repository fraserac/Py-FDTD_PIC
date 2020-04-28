# -*- coding: utf-8 -*-
"""
This is the start of my FDTD code. So far 1D.
This is currently a 1D work in progress FDTD
This script is the engine that will calculate field updates and plotting functions
The plotting and animation functionality may eventually be moved to a separate script.
This script will eventually just contain the update equations and animation

import scipy as sci
import math
from Material_Def import *

#from MasterController import *
"""

import numpy as np
import math
import matplotlib.pylab as plt
import matplotlib.animation as animation
#from Material_Def import *
from scipy import signal as sign
import sys 
from decimal import *
import itertools as it
from scipy import signal as sign

  ##M MAKE THIS MORE ACCURATE LATER

#Ex =[]#,dtype=complex)
#Hy=[]#,dtype=complex)
#Ex_History= [[]]
#nzsrc = 50#round(Nz/2)
#size =0

"""
Ex =[]#,dtype=complex)
Hy=[]#,dtype=complex)
Ex_History= [[]]
Hy_History= [[]]  # feed in timesteps


"""

"""


def init():
 line.set_ydata([])
 return line,

#MAIN FDTD LOOP BASIC EDITION PRE-PRE-PRE-ALPHA
#for loop over grid up to basic b.c. for update eqns, iterate through Nz with nz

def sourceGen(T):
    print(T)
    pulse = np.sin(2*np.pi*freq_in*delT*2*T)
    #pulse = np.exp(-((t-t0)/tau)**2)
    return pulse

"""

"""  
for k in range(0, MaterialFrontEdge-1):  
    UpExMat[k] =UpExFree
    UpHyMat[k] =UpHyFree

for ii in range(MaterialRearEdge-1, Nz):
    UpExMat[ii] = UpExFree
    UpHyMat[ii] = UpHyFree


"""
"""
for count in range(timeSteps):
    #print(count)
    Hy[Nz-1] = Hy[Nz-2]

    Ex[nzsrc]= Ex[nzsrc] + np.exp(-(count +0.5 -(-0.5)-30)*(count +0.5 -(-0.5)-30)/100) #tf/sf correction Ex
    Ex_History[count] = np.insert(Ex_History[count], 0, Ex)
     
"""
# WHERE DOES SINGLE SINE SOURCE SIT IN LOOP? PROPAGATING? FOURIER TRANSFORM SHOW THE CORRECT FREQUENCY? IF IT DOES, 
# DOES THE TRANSMISSION WORK? IF SO, GET THE RICKER WAVELET TF/SF UP AND RUNNING, THEN SET UP A LORENZ STYLE MATERIAL 
# WITH FREQUENCY DEPENDENCE? NONLINEAR PML? NONLINEAR SUSCEPTIBILITY MATERIAL? THEN OTHER VALIDATION METHODS, THEN
#EXPAND TO 2D after considering memory optimisation, then include particle sutff. 
def sourceGen(T):
    #print(T)
    pulse = np.exp(-(T - 30)*(T-30)/100)# - (50))*(1/CharImp)
    #print(pulse)
    #pulse = np.exp(-((t-t0)/tau)**2)
    return pulse

def sourceGen2(P, T):
    #deltaT =( 50*0.00033273660620797844)/(2*299462945.5871806) - 1.111111111111111e-12/2
    pulse = np.sin(2*np.pi*P.freq_in*P.delT*T)
    #print(pulse)
    #pulse = np.exp(-((t-t0)/tau)**2)
    return pulse



def FieldInit(V,P):
    while (True):
        try:
            if type(P.Nz) != int:
                raise TypeError('Grid spaces must be a positive integer value')
                break
            if type(P.timeSteps) != int:
                raise TypeError('Number of timeSteps must be positive integer valued')
                break
            if P.timeSteps > 2**14:
                raise ValueError('timeSteps max too large')
                break
            if P.Nz > 3000:
                raise ValueError('Grid size too big')
                break
            if P.Nz ==0:
                raise ValueError('Cannot have grid size 0!')
                break
            if P.timeSteps==0:
                raise ValueError('Cannot have zero timeSteps!')
                break
            if P.Nz < 0:
                raise ValueError('Grid size cannot be negative')
                break
            if P.timeSteps < 0:
                raise ValueError('Cannot have negative timeSteps')
                break
            break
        except ValueError as e:
            print(e)
            sys.exit() 
    V.Ex =np.zeros(P.Nz+1)#,dtype=complex)
    V.Hy=np.zeros(P.Nz+1)#,dtype=complex)
    V.Ex_History= [[]]*P.timeSteps
    V.Hy_History= [[]]*P.timeSteps
    V.Psi_Ex_History= [[]]*P.timeSteps
    V.Psi_Hy_History= [[]]*P.timeSteps
    V.Exs = []
    V.Hys = []
    V.polarisationCurr = np.zeros(P.Nz+1)# return later
    V.Dx = np.zeros(P.Nz+1)
    V.tempVarPol = np.zeros(P.Nz+1)
    V.tempTempVarPol = np.zeros(P.Nz+1)
    
    return V.tempTempVarPol, V.polarisationCurr, V.Ex, V.Dx, V.Ex_History, V.Hy, V.Hy_History, V.Psi_Ex_History, V.Psi_Hy_History, V.Exs, V.Hys


def SmoothTurnOn(V,P):
    ppw =  P.c0 /(P.freq_in*P.dz)
    for timer in range(P.timeSteps):
        if(timer*P.delT < P.period):
            V.Exs.append(float(Decimal(np.sin(2.0*np.pi/ppw*(P.courantNo*timer)))))
            V.Hys.append(float(Decimal(np.sin(2.0*np.pi/ppw*(P.courantNo*(timer+1))))))
        elif(timer*P.delT >= P.period):  
            V.Exs.append(0)
            V.Hys.append(0)
    for boo in range(P.timeSteps):
        if(V.Hys[boo] ==0):
          V.Hys[boo-1] =0
        #if(V.Exs[boo] ==0):
         # V.Exs[boo-1] =0
          break    
    return V.Exs, V.Hys   
# FIX TURN OFF JITTER


def EmptySpaceCalc(V,P): # this function will run the FDTD over just the initial media and measure the points at x1 over t
    #material to find transmission and reflection vals
    V.UpHyMat = np.zeros(P.Nz+1) #THIS IS INITIALISER DON'T PASS THROUGH FOR LOOP
    V.UpExMat = np.zeros(P.Nz+1)
    
    UpHyBackground = (1/P.CharImp)*P.courantNo
    UpExBackground = P.CharImp*P.courantNo


    for jl in range(0, P.Nz):  
        V.UpExMat[jl] =UpExBackground
        V.UpHyMat[jl] =UpHyBackground
 
    return V.UpHyMat, V.UpExMat



def Material(V,P):
    V.epsilon = np.ones(P.Nz)
    V.mu = np.ones(P.Nz)
    for kj in range(P.Nz):
            if(kj < int(P.materialFrontEdge)):  
                V.epsilon[kj] = 1
                V.mu[kj] = 1
                V.UpExHcompsCo[kj] =1
                V.UpExSelf[kj] =1
                V.UpHyEcompsCo[kj] =1
                V.UpHySelf[kj] = 1
            if(kj >= int(P.materialFrontEdge) and kj < int(P.materialRearEdge)):
                V.epsilon[kj] = P.epsRe
                V.mu[kj] = P.muRe
                V.UpExHcompsCo[kj] = P.eHcompsCo
                V.UpExSelf[kj] = P.eSelfCo
                V.UpHyEcompsCo[kj] =P.hEcompsCo
                V.UpHySelf[kj] = P.hSelfCo
            if(kj>= P.materialRearEdge):
                V.epsilon[kj] = 1
                V.mu[kj] = 1 
                V.UpExHcompsCo[kj] = 1
                V.UpExSelf[kj] =1
                V.UpHyEcompsCo[kj] =1
                V.UpHySelf[kj] = 1
    return V.epsilon, V.mu, V.UpExHcompsCo, V.UpExSelf, V.UpHyEcompsCo, V.UpHySelf

    


#PASS TO MASTER CONTROLLER, THEN THROUGH TO UPDATECOEF, WHERE THE NEW CO-EFF WILL DIVIDE uPEX AND UPHY at mat, 

def UpdateCoef(V,P):# POTENTIAL ISSUE, COURANT NO AND DOUBLE DEFINITION OF MU EPS.
    #CHECK COURANT NO.
    UpHyBackground = (1/P.CharImp)*P.courantNo
    UpExBackground = P.CharImp*P.courantNo
    UpHyMat = np.zeros(P.Nz+1) #THIS IS INITIALISER DON'T PASS THROUGH FOR LOOP
    UpExMat = np.zeros(P.Nz+1)
    #UpHySelf = np.zeros(P.Nz)
    #UpExSelf = np.zeros(P.Nz)
    
    for k in range(P.Nz):
        UpExMat[k]= UpExBackground/V.epsilon[k]
        UpHyMat[k]= UpHyBackground/V.mu[k]

    return UpHyMat, UpExMat




# FOR HY AND EX update/EZ? feed in eSelfCo and hSelfCo


def HyTfSfCorr(V, P, counts):
     #V.Hy[P.nzsrc-1] -= sourceGen(counts)
     V.Hy[P.nzsrc-1] -= V.Exs[counts]/P.CharImp#*np.exp(-(counter - 30)*(counter-30)/100)
     #link to sourceGen for harmonic or ricker or gaussian etc 
     #np.sin((2*np.pi)/Nlam*(courantNo))
     return V.Hy[P.nzsrc-1]
   

   

def ExTfSfCorr(V,P, counts):
    #V.Ex[P.nzsrc] += sourceGen2(P, counts)
    V.Ex[P.nzsrc] += V.Hys[counts]# *np.exp(-(counter +0.5 -(-0.5)-30)*(counter +0.5 -(-0.5)-30)/100)
    return V.Ex[P.nzsrc]



##### CPML STUFF 
    
def CPML_FieldInit(V,P, C_V, C_P):#INITIALISE FIELD PARAMS 
    C_V.kappa_Ex =np.zeros(P.Nz+1)
    C_V.kappa_Hy = np.zeros(P.Nz+1)
    C_V.psi_Ex =np.zeros(P.Nz+1)
    C_V.psi_Hy = np.zeros(P.Nz+1)
    C_V.alpha_Ex= np.zeros(P.Nz+1)
    C_V.alpha_Hy= np.zeros(P.Nz+1)
    C_V.sigma_Ex =np.zeros(P.Nz+1)   # specific spatial value of conductivity 
    C_V.sigma_Hy = np.zeros(P.Nz+1)
    C_V.beX =np.zeros(P.Nz+1)#np.exp(-(sigma_Ex/(permit_0*kappa_Ex) + alpha_Ex/permit_0 )*delT)
    C_V.bmY =np.zeros(P.Nz+1)#np.exp(-(sigma_Hy/(permea_0*kappa_Hy) + alpha_Hy/permea_0 )*delT)
    C_V.ceX = np.zeros(P.Nz+1)
    C_V.cmY = np.zeros(P.Nz+1)
    C_V.eLoss_CPML = np.zeros(P.Nz+1)
    C_V.mLoss_CPML = np.zeros(P.Nz+1)
    C_V.Ca = np.zeros(P.Nz+1)
    C_V.Cb = np.zeros(P.Nz+1)
    C_V.Cc = np.zeros(P.Nz+1)
    C_V.C1 = np.zeros(P.Nz+1)
    C_V.C2 = np.zeros(P.Nz+1)
    C_V.C3 = np.zeros(P.Nz+1)
    
    return C_V
    
def CPML_ScalingCalc(V, P, C_V, C_P):
    jj=P.pmlWidth
    jjj = P.pmlWidth-1
    for nz in range(P.Nz):   #SCALES VARS AWAY FROM SIM DOMAIN, REVERSE FOR LOOP
        nz1= nz+1
        if (nz1 <=P.pmlWidth): 
            C_V.kappa_Ex[nz] =1+(C_P.kappaMax-1)*((P.pmlWidth-nz1)/(P.pmlWidth-1))**C_P.r_scale
            
            C_V.sigma_Ex[nz] = C_P.sigmaOpt*((P.pmlWidth - nz1)/(P.pmlWidth-1))**C_P.r_scale   #np.abs((((np.abs(nz -0.75))/P.pmlWidth)**(C_P.r_scale))*C_P.sigmaEMax)
            
            C_V.alpha_Ex[nz] = C_P.alphaMax*(nz1/(P.pmlWidth-1))**C_P.r_a_scale #np.abs((1-((counter)/P.pmlWidth)**C_P.r_a_scale)*C_P.alphaMax)
            
            
        elif nz >= P.Nz+2 -(P.pmlWidth):
            C_V.kappa_Ex[nz] = C_V.kappa_Ex[jj]
            
            C_V.sigma_Ex[nz] = C_V.sigma_Ex[jj]
            
            C_V.alpha_Ex[nz] = C_V.alpha_Ex[jj]
            jj-=1
            
        else:
            C_V.kappa_Ex[nz] = 1
            
            C_V.sigma_Ex[nz] = 0
            
            C_V.alpha_Ex[nz] = 0
            
    #for nz in range(P.Nz-1):   
        
        if (nz1 <=P.pmlWidth-1):
            C_V.kappa_Hy[nz] =1+(C_P.kappaMax-1)*((P.pmlWidth-nz1-0.5)/(P.pmlWidth-1))**C_P.r_scale
            C_V.sigma_Hy[nz] = C_P.sigmaOpt*((P.pmlWidth -nz1-0.5)/(P.pmlWidth-1))**C_P.r_scale   #np.abs(((C_P.sigmaEMax)*(np.abs(nz -0.75)/P.pmlWidth)**(C_P.r_scale))*C_P.sigmaHMax)
            C_V.alpha_Hy[nz] = C_P.alphaMax*(nz1-0.5/(P.pmlWidth-1))**C_P.r_a_scale#np.abs(((1)*(1-(counter)/P.pmlWidth)**(C_P.r_a_scale))*C_P.alphaMax)
            
        
        elif nz >= P.Nz +2 - P.pmlWidth:
            C_V.kappa_Hy[nz] = C_V.kappa_Hy[jjj]
            
            C_V.sigma_Hy[nz] = C_V.sigma_Hy[jjj]
            
            C_V.alpha_Hy[nz] = C_V.alpha_Hy[jjj]
            jjj-=1
        else:
            C_V.kappa_Hy[nz] = 1
            
            C_V.sigma_Hy[nz] = 0
            
            C_V.alpha_Hy[nz] = 0 
        
    return C_V.sigma_Ex, C_V.sigma_Hy, C_V.alpha_Ex,  C_V.alpha_Hy, C_V.kappa_Ex, C_V.kappa_Hy

    
def CPML_Ex_RC_Define(V, P, C_V, C_P):
    
    jj = P.pmlWidth-1
    jjj =P.pmlWidth-1
    for nz in range(0, P.Nz):
        if(nz <=P.pmlWidth-1):
            
            C_V.beX[nz] = np.exp(-((C_V.sigma_Ex[nz]*P.delT/C_V.kappa_Ex[nz])+((C_V.alpha_Ex[nz]*P.delT)/P.permit_0)))
        
        elif nz >= P.Nz+2 -P.pmlWidth:
            C_V.beX[nz] = np.exp(-((C_V.sigma_Ex[nz]*P.delT/C_V.kappa_Ex[nz])+((C_V.alpha_Ex[nz]*P.delT)/P.permit_0)))
            jj-=1
        else:
            C_V.beX[nz] =0
            
            
            
        if C_V.sigma_Ex[nz] ==0 and nz==P.pmlWidth-1 and C_V.alpha_Ex[nz]==0:
           C_V.ceX[nz]=0
        elif nz <=P.pmlWidth-1:
            C_V.ceX[nz] = (C_V.beX[nz]-1)*C_V.sigma_Ex[nz]/(C_V.sigma_Ex[nz]+C_V.alpha_Ex[nz]*C_V.kappa_Ex[nz])/C_V.kappa_Ex[nz]
        elif nz >= P.Nz+2 -P.pmlWidth:
            C_V.ceX[nz] = (C_V.beX[jjj]-1)*C_V.sigma_Ex[jjj]/(C_V.sigma_Ex[jjj]+C_V.alpha_Ex[jjj]*C_V.kappa_Ex[jjj])/C_V.kappa_Ex[jjj]
            jjj-=1
        else:
             C_V.ceX[nz] =0
            
    return C_V.beX, C_V.ceX


def CPML_HY_RC_Define(V, P, C_V, C_P):
    jj = P.pmlWidth-2
    jjj =P.pmlWidth-2
   
    for nz in range(0, P.Nz):
        
        if(nz <=P.pmlWidth-2):
            C_V.bmY[nz] = np.exp(-(C_V.sigma_Hy[nz]/(C_V.kappa_Hy[nz]+C_V.alpha_Hy[nz]))*P.delT/P.permea_0)
        elif nz >= P.Nz+2 -P.pmlWidth:
            C_V.bmY[nz] = np.exp(-(C_V.sigma_Hy[jj]/(C_V.kappa_Hy[jj]+C_V.alpha_Hy[jj]))*P.delT/P.permea_0)
            jj-=1
        else:
            C_V.bmY[nz] =0 
            
             
        if C_V.sigma_Hy[nz] ==0 and nz==P.pmlWidth-1 and C_V.alpha_Hy[nz]==0:
           C_V.cmY[nz]=0
        elif nz<=P.pmlWidth-1:
            C_V.cmY[nz] = (C_V.bmY[nz]-1)*C_V.sigma_Hy[nz]/(C_V.sigma_Hy[nz]+C_V.alpha_Hy[nz]*C_V.kappa_Hy[nz])/C_V.kappa_Hy[nz]
        elif nz >= P.Nz+2 -P.pmlWidth:
            C_V.cmY[nz] = (C_V.bmY[jjj]-1)*C_V.sigma_Hy[jjj]/(C_V.sigma_Hy[jjj]+C_V.alpha_Hy[jjj]*C_V.kappa_Hy[jjj])/C_V.kappa_Hy[jjj]
            jjj-=1
        else:
             C_V.cmY[nz] =0    
            
        #C_V.cmY[P.Nz-1] = C_V.cmY[P.Nz-2]
    return C_V.bmY, C_V.cmY

def CPML_Ex_Update_Coef(V,P, C_V, C_P):
    for nz in range(0, P.Nz-1):
        C_V.eLoss_CPML[nz] = (C_V.sigma_Ex[nz]*P.delT)/(2*P.permit_0)
        
        C_V.Ca[nz] = V.UpExSelf[nz]#(1-C_V.eLoss_CPML[nz])/(1+C_V.eLoss_CPML[nz])
        C_V.Cb[nz] =V.UpExHcompsCo[nz]*V.UpExMat[nz]#P.delT/P.permit_0/(1+C_V.eLoss_CPML[nz])
        C_V.Cc[nz] = P.delT/((1+C_V.eLoss_CPML[nz])*P.permit_0)
        
    #for nz in range(P.pmlWidth, P.Nz-P.pmlWidth-1 ):
       # C_V.Ca[nz] = V.UpExSelf[nz]
        #C_V.Cb[nz] = V.UpExHcompsCo[nz]
        
    return C_V.eLoss_CPML, C_V.Ca, C_V.Cb, C_V.Cc    


def CPML_Hy_Update_Coef(V,P, C_V, C_P):
    for nz in range(0, P.Nz-2):
        #C_V.mLoss_CPML[nz] = (C_V.sigma_Hy[nz]*P.delT)/(2*P.permea_0)
        
        C_V.C1[nz] =1#V.UpHySelf[nz] #(1-C_V.mLoss_CPML[nz])/(1+C_V.mLoss_CPML[nz])
        C_V.C2[nz] = -P.delT/P.permea_0# V.UpHyEcompsCo[nz]#P.delT/((1+C_V.mLoss_CPML[nz])*P.permea_0*C_V.kappa_Hy[nz]*P.dz)
        C_V.C3[nz] = P.delT/  ((1+C_V.mLoss_CPML[nz])*P.permea_0)#NOT CURRENTLY USED
    #for nz in range(P.pmlWidth ,P.Nz-P.pmlWidth-1): 
       # C_V.C1[nz] = V.UpHySelf[nz]
       # C_V.C2[nz] = V.UpHyEcompsCo[nz]
        
    return C_V.mLoss_CPML, C_V.C1, C_V.C2, C_V.C3   
    
def denominators(V, P, C_V, C_P):
    jj = P.pmlWidth-2
    for j in range(0,P.Nz): 
        if j <= P.pmlWidth -2:
            C_V.den_Hydz[j]= 1/(C_V.kappa_Hy[j])
        elif j>= P.Nz+2-P.pmlWidth:
            C_V.den_Hydz[j] = 1/(C_V.kappa_Hy[jj])
            
            jj-=1
        else:
            C_V.den_Hydz[j] = 1/(1)
    jj = P.pmlWidth-1
    for j in range(0,P.Nz): 
        if j <= P.pmlWidth-1:
            C_V.den_Exdz[j]= 1/(C_V.kappa_Ex[j])
        elif j>= P.Nz+2-P.pmlWidth:
            C_V.den_Exdz[j] = 1/(C_V.kappa_Ex[jj])
            jj-=1
        else:
            C_V.den_Exdz[j] = 1/(1)        
    return C_V.den_Exdz, C_V.den_Hydz
    
    


def CPML_Psi_e_Update(V,P, C_V, C_P):   # recursive convolution for E field REF
    for nz in range(1, P.Nz-1): 
        C_V.psi_Ex[nz] = C_V.beX[nz]*C_V.psi_Ex[nz] + C_V.ceX[nz]*(V.Hy[nz]-V.Hy[nz-1])
        V.Ex[nz] = V.Ex[nz] + C_V.Cb[nz]*C_V.psi_Ex[nz]
    
    return C_V.psi_Ex, V.Ex 

 
def CPML_Psi_m_Update(V,P, C_V, C_P):   # recursive convolution for H field REF
    for nz in range(0, P.Nz-2): 
        C_V.psi_Hy[nz] = C_V.bmY[nz]*C_V.psi_Hy[nz] + C_V.cmY[nz]*(V.Ex[nz]-V.Ex[nz+1])
        V.Hy[nz] = V.Hy[nz] + C_V.C2[nz]*C_V.psi_Hy[nz]
    
    return C_V.psi_Hy, V.Hy 



def CPML_HyUpdate(V,P, C_V, C_P):
    for nz in range(0, P.Nz-2):
        V.Hy[nz] = V.Hy[nz]*V.UpHySelf[nz] + (V.Ex[nz+1]-V.Ex[nz])*V.UpHyMat[nz]*V.UpHyEcompsCo[nz]#*C_V.den_Hydz[nz]
        #V.Hy[nz]= C_V.C1[nz]*V.Hy[nz]+C_V.C2[nz]*(V.Ex[nz]-V.Ex[nz+1])*C_V.den_Hydz[nz]
    return V.Hy


def CPML_ExUpdate(V,P, C_V, C_P):
    for nz in range(1, P.Nz-1):
        V.Ex[nz] = V.Ex[nz]*V.UpExSelf[nz] + (V.Hy[nz]-V.Hy[nz-1])*V.UpExHcompsCo[nz]*V.UpExMat[nz]#*C_V.den_Exdz[nz]
        #V.Ex[nz]= C_V.Ca[nz]*V.Ex[nz]+C_V.Cb[nz]*(V.Hy[nz-1]-V.Hy[nz])*C_V.den_Exdz[nz]
    return V.Ex


def CPML_PEC(V, P, C_V, C_P):
    V.Ex[0] =0
   # V.Ex[P.Nz-1]=0
    return V.Ex[0], V.Ex[P.Nz-1]

def CPML_PMC(V,P,C_V, C_P):
    V.Hy[P.Nz-1]=0
    return V.Hy[P.Nz-1]


def PLRC(V,P, C_V, C_P):
    #implementation of PLRC --- small novelty?
    
    
    pass

# NEED TO SET UP ADE VARIABLES TO BE ZERO WHEN NOT IN MATERIAL, MAKE ARRAY

def ADE_TempPolCurr(V,P):
     for nz in range(1, P.Nz-1):
         V.tempTempVarPol[nz] = V.tempVarPol[nz]
         V.tempVarPol[nz] = V.polarisationCurr[nz]  
     return V.tempTempVarPol, V.tempVarPol


def ADE_PolarisationCurrent_Ex(V, P, C_V, C_P):
    #take vars from re-arragement and use to update polarisationCurr
    D= (1/P.delT**2)+(V.gammaE/(2*P.delT))
    print("D ", D)
    A = ((2/P.delT**2)-V.omega_0E**2)/D
    print("A", A)
    B = ((V.gammaE/(2*P.delT))-1/P.delT**2)/D
    print(B)
    C = (P.permit_0*(V.plasmaFreqE**2))/D
    print(C)
  
    for nz in range (int(P.materialFrontEdge-1), int(P.materialRearEdge)):### PROBLEM # TEMP VAR POL ISSUE
        V.polarisationCurr[nz] = A*V.tempVarPol[nz]+ B*V.tempTempVarPol[nz] +C*V.Ex[nz]
        #print(V.polarisationCurr[nz])
    return V.polarisationCurr

def ADE_HyUpdate(V, P, C_V, C_P):
    #free space calc:
    """
    for nz in range(0, int(P.materialFrontEdge)):
        V.Hy[nz] = V.Hy[nz]*V.UpHySelf[nz] + (V.Ex[nz+1]-V.Ex[nz])*V.UpHyMat[nz] ## COURANT NO.
     
    if P.materialRearEdge < P.Nz-1:
        for nzz in range(int(P.materialRearEdge-1), P.Nz):
            V.Hy[nz] = V.Hy[nz]*V.UpHySelf[nz] + (V.Ex[nz+1]-V.Ex[nz])*V.UpHyMat[nz]
    #if P.MaterialRearEdge >= P.Nz-1:
    """
    for nz in range(1, P.Nz-1):
        V.Hy[nz] = V.Hy[nz]*V.UpHySelf[nz] + (V.Ex[nz+1]-V.Ex[nz])*V.UpHyMat[nz]
    return V.Hy
        
def ADE_MyUpdate():
    
    pass

def ADE_ExUpdate(V, P, C_V, C_P): #### free space 
    #linear polarization
    
    for nz in range(1, int(P.materialFrontEdge-1)):
        V.Ex[nz] =V.UpExSelf[nz]*V.Ex[nz] + (V.Hy[nz]-V.Hy[nz-1])*V.UpExMat[nz]
    #for nz in range(int(P.materialFrontEdge-1), int(P.materialRearEdge)):
        
    if P.materialRearEdge < P.Nz-1:
        for nzz in range(int(P.materialRearEdge-1), P.Nz):
            V.Ex[nzz] = V.UpExSelf[nzz]*V.Ex[nzz] + (V.Hy[nzz]-V.Hy[nzz-1])*V.UpExMat[nzz]
            
    """
    for nz in range(1, P.materialFrontEdge):
        V.Ex[nz] = V.Ex[nz]*V.UpExSelf[nz] + (V.Hy[nz]-V.Hy[nz-1])*V.UpExMat[nz]
    """   
    return V.Ex#




def ADE_ExCreate(V, P, C_V, C_P):
    for nz in range(int(P.materialFrontEdge-1), int(P.materialRearEdge)):
       V.Ex[nz] =(V.Dx[nz] - V.polarisationCurr[nz])/P.permit_0
       #acting like hard source
    return V.Ex

def ADE_DxUpdate(V, P, C_V, C_P):
    for nz in range(int(P.materialFrontEdge-1), int(P.materialRearEdge)):
        V.Dx[nz] = V.Dx[nz] +(V.Hy[nz] - V.Hy[nz-1])*P.delT/P.dz
        V.Ex[nz] =(V.Dx[nz] - V.polarisationCurr[nz])/P.permit_0
    return V.Dx, V.Ex

#SET UP TEMP POLLCURR VAR, POLLCURR q+ 1 = A* POLLCURR  q + B*pollcurrTemp + C*E q




def ADE_NonLinMyUpdate():
    pass

def ADE_NonLinPxUpdate():
    pass

def SpatialFiltering():
    
    pass

def SymbolicRegression():
    pass

"""
Issue: feed fields back and forth. RETURN uphy etc etc , call
"""
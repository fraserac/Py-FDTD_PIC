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
import scipy as sci
import math
import matplotlib.pylab as plt
import matplotlib.animation as animation
#from Material_Def import *
from scipy import signal as sign
import sys 
from decimal import *
import itertools as it

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
    pulse = np.sin(2*np.pi*freq_in*delT*T)#np.exp(-(T - 30)*(T-30)/100)# - (50))*(1/CharImp)
    #print(pulse)
    #pulse = np.exp(-((t-t0)/tau)**2)
    return pulse

def sourceGen2(T):
    #deltaT =( 50*0.00033273660620797844)/(2*299462945.5871806) - 1.111111111111111e-12/2
    pulse = np.sin(2*np.pi*freq_in*delT*T - (50+1))
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
            if P.Nz > 1000:
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
    V.Ex =np.zeros(P.Nz)#,dtype=complex)
    V.Hy=np.zeros(P.Nz)#,dtype=complex)
    V.Ex_History= [[]]*P.timeSteps
    V.Hy_History= [[]]*P.timeSteps
    V.Psi_Ex_History= [[]]*P.timeSteps
    V.Psi_Hy_History= [[]]*P.timeSteps
    return V.Ex, V.Ex_History, V.Hy, V.Hy_History, V.Psi_Ex_History, V.Psi_Hy_History


def SmoothTurnOn(V,P):
    ppw =  P.c0 /(P.freq_in*P.dz)
    for timer in range(P.timeSteps):
        if(timer*P.delT < P.period):
            V.Exs.append(float(Decimal(np.sin(2.0*np.pi/ppw*(P.courantNo*timer)))))
            V.Hys.append(float(Decimal(np.sin(2.0*np.pi/ppw*(P.courantNo*(timer+1))))))
        elif(timer*P.delT >= P.period):  
            V.Exs.append(0)
            V.Hys.append(0)
    return V.Exs, V.Hys   
# FIX TURN OFF JITTER


def EmptySpaceCalc(V,P): # this function will run the FDTD over just the initial media and measure the points at x1 over t
    #material to find transmission and reflection vals
    V.UpHyMat = np.zeros(P.Nz) #THIS IS INITIALISER DON'T PASS THROUGH FOR LOOP
    V.UpExMat = np.zeros(P.Nz)
    
    UpHyBackground = (1/P.CharImp)*P.courantNo
    UpExBackground = P.CharImp*P.courantNo


    for jl in range(0, P.Nz):  
        V.UpExMat[jl] =UpExBackground
        V.UpHyMat[jl] =UpHyBackground
 
    return V.UpHyMat, V.UpExMat



def Material(V,P):
    for kj in range(P.Nz):
            if(kj < P.materialFrontEdge):  
                V.epsilon[kj] = 1
                V.mu[kj] = 1
                V.UpExHcompsCo[kj] = 1
                V.UpExSelf[kj] =1
                V.UpHyEcompsCo[kj] =1
                V.UpHySelf[kj] = 1
            if(kj >= P.materialFrontEdge and kj < P.materialRearEdge):
                V.epsilon[kj] = P.epsRe
                V.mu[kj] = P.muRe
                V.UpExHcompsCo[kj] = 1
                V.UpExSelf[kj] = 1
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
    UpHyMat = np.zeros(P.Nz) #THIS IS INITIALISER DON'T PASS THROUGH FOR LOOP
    UpExMat = np.zeros(P.Nz)
    #UpHySelf = np.zeros(P.Nz)
    #UpExSelf = np.zeros(P.Nz)
    
    for k in range(P.Nz):
        UpExMat[k]= UpExBackground/V.epsilon[k]
        UpHyMat[k]= UpHyBackground/V.mu[k]

    return UpHyMat, UpExMat



def HyBC(V,P):
    V.Hy[P.Nz-1] = V.Hy[P.Nz-2]
    return V.Hy[P.Nz-1]
   
# FOR HY AND EX update/EZ? feed in eSelfCo and hSelfCo
def HyUpdate(V,P):
    for nz in range(P.pmlWidth, P.Nz-1-P.pmlWidth):   #START AFTER CPML AND FINISH BEFORE
        V.Hy[nz] = V.Hy[nz]*V.UpHySelf[nz] + (V.Ex[nz+1]-V.Ex[nz])*V.UpHyMat[nz]*V.UpHyEcompsCo[nz]
    return V.Hy[P.pmlWidth:P.Nz-1-P.pmlWidth]

def HyTfSfCorr(V, P, counts):
     V.Hy[P.nzsrc-1] -= V.Exs[counts]/P.CharImp#*np.exp(-(counter - 30)*(counter-30)/100)
     #link to sourceGen for harmonic or ricker or gaussian etc 
     #np.sin((2*np.pi)/Nlam*(courantNo))
     return V.Hy[P.nzsrc-1]
   
def ExBC(V, P):
    V.Ex[0] = V.Ex[1]
    V.Ex[P.Nz-1]=  V.Ex[P.Nz-2]
    return V.Ex[0], V.Ex[P.Nz-1]
   

def ExUpdate(V, P):
    for nz in range(P.pmlWidth, P.Nz-1-P.pmlWidth):
        V.Ex[nz] = V.Ex[nz]*V.UpExSelf[nz] + (V.Hy[nz]-V.Hy[nz-1])*V.UpExMat[nz]*V.UpExHcompsCo[nz]#*UpExMat[nz]
    return V.Ex[P.pmlWidth: P.Nz-1-P.pmlWidth]    


def ExTfSfCorr(V,P, counts):
   # ExTfsf= ExTfsf + np.exp(-(counter +0.5 -(-0.5)-30)*(counter +0.5 -(-0.5)-30)/100)
    V.Ex[P.nzsrc] += V.Hys[counts]# *np.exp(-(counter +0.5 -(-0.5)-30)*(counter +0.5 -(-0.5)-30)/100)
    return V.Ex[P.nzsrc]



##### CPML STUFF 
    
def CPML_FieldInit(V,P, C_V, C_P):#INITIALISE FIELD PARAMS 
    C_V.kappa_Ex =1
    C_V.kappa_Hy = 1
    C_V.psi_Ex =np.zeros(P.Nz)
    C_V.psi_Hy = np.zeros(P.Nz)
    C_V.alpha_Ex= np.zeros(P.Nz)
    C_V.alpha_Hy= np.zeros(P.Nz)
    C_V.sigma_Ex =np.zeros(P.Nz)   # specific spatial value of conductivity 
    C_V.sigma_Hy = np.zeros(P.Nz)
    C_V.beX =np.zeros(P.Nz)#np.exp(-(sigma_Ex/(permit_0*kappa_Ex) + alpha_Ex/permit_0 )*delT)
    C_V.bmY =np.zeros(P.Nz)#np.exp(-(sigma_Hy/(permea_0*kappa_Hy) + alpha_Hy/permea_0 )*delT)
    C_V.ceX = np.zeros(P.Nz)
    C_V.cmY = np.zeros(P.Nz)
    C_V.eLoss_CPML = np.zeros(P.Nz)
    C_V.mLoss_CPML = np.zeros(P.Nz)
    C_V.Ca = np.zeros(P.Nz)
    C_V.Cb = np.zeros(P.Nz)
    C_V.Cc = np.zeros(P.Nz)
    C_V.C1 = np.zeros(P.Nz)
    C_V.C2 = np.zeros(P.Nz)
    C_V.C3 = np.zeros(P.Nz)
    
    return C_V
    
def CPML_ScalingCalc(V, P, C_V, C_P):
    counter =1
    for nz in range(P.pmlWidth-1, -1, -1 ):   #SCALES VARS AWAY FROM SIM DOMAIN, REVERSE FOR LOOP
        C_V.sigma_Ex[nz] =np.abs((((counter-0.5*P.dz)/P.pmlWidth)**(C_P.r_scale))*C_P.sigmaEMax)
        C_V.sigma_Hy[nz] =np.abs(((P.CharImp**2)*((counter)/P.pmlWidth)**(C_P.r_scale))*C_P.sigmaHMax)
        C_V.alpha_Ex[nz] =np.abs((1-((counter-0.5*P.dz)/P.pmlWidth)**C_P.r_a_scale)*C_P.alphaMax)
        C_V.alpha_Hy[nz] =np.abs(((P.CharImp**2)*(1-(counter)/P.pmlWidth)**(C_P.r_a_scale))*C_P.alphaMax)
        counter +=1
    nz =0
    counter =1
    for nz in range(P.Nz-P.pmlWidth-1, P.Nz): 
        C_V.sigma_Ex[nz] =np.abs((((counter-0.5*P.dz)/P.pmlWidth)**(C_P.r_scale))*C_P.sigmaEMax)
        C_V.sigma_Hy[nz] =np.abs(((P.CharImp**2)*((counter)/P.pmlWidth)**(C_P.r_scale))*C_P.sigmaHMax)
        C_V.alpha_Ex[nz] = np.abs((1-((counter-0.5*P.dz)/P.pmlWidth)**C_P.r_a_scale)*C_P.alphaMax)
        C_V.alpha_Hy[nz] =np.abs(((P.CharImp**2)*(1-(counter)/P.pmlWidth)**(C_P.r_a_scale))*C_P.alphaMax)
        counter+=1
    return C_V.sigma_Ex, C_V.sigma_Hy, C_V.alpha_Ex,  C_V.alpha_Hy
    
    
def CPML_Ex_RC_Define(V, P, C_V, C_P):
    for nz in it.chain(range(0, P.pmlWidth), range(P.Nz-P.pmlWidth, P.Nz)):
        C_V.beX[nz] = np.exp((-(C_V.sigma_Ex[nz]/(P.permit_0*C_V.kappa_Ex)) + C_V.alpha_Ex[nz]/P.permit_0)*P.delT)
        C_V.ceX[nz] = (C_V.beX[nz]-1)*(C_V.sigma_Ex[nz]/(C_V.kappa_Ex*(C_V.sigma_Ex[nz]+C_V.alpha_Ex[nz]*C_V.kappa_Ex)))
    return C_V.beX, C_V.ceX

def CPML_HY_RC_Define(V, P, C_V, C_P):
    for nz in it.chain(range(0, P.pmlWidth), range(P.Nz-P.pmlWidth, P.Nz)):
        C_V.bmY[nz] = np.exp((-(C_V.sigma_Hy[nz]/(P.permea_0*C_V.kappa_Hy)) + C_V.alpha_Hy[nz]/P.permea_0)*P.delT)
        C_V.cmY[nz] = (C_V.bmY[nz]-1)*(C_V.sigma_Hy[nz]/(C_V.kappa_Hy*(C_V.sigma_Hy[nz]+C_V.alpha_Hy[nz]*C_V.kappa_Hy)))
    return C_V.bmY, C_V.cmY

def CPML_Ex_Update_Coef(V,P, C_V, C_P):
    for nz in it.chain(range(0, P.pmlWidth), range(P.Nz-P.pmlWidth, P.Nz)):
        C_V.eLoss_CPML[nz] = (C_V.sigma_Ex[nz]*P.delT)/(2*P.permit_0)
        if(C_V.eLoss_CPML[nz]>= 1):
            C_V.Ca[nz] = 0.01
        else:
            C_V.Ca[nz] = (1-C_V.eLoss_CPML[nz])/(1+C_V.eLoss_CPML[nz])
        C_V.Cb[nz] = -P.delT/((1+C_V.eLoss_CPML[nz])*P.permit_0*C_V.kappa_Ex*P.dz)
        C_V.Cc[nz] = -P.delT/((1+C_V.eLoss_CPML[nz])*P.permit_0)
    return C_V.eLoss_CPML, C_V.Ca, C_V.Cb, C_V.Cc    


def CPML_Hy_Update_Coef(V,P, C_V, C_P):
    for nz in it.chain(range(0, P.pmlWidth), range(P.Nz-P.pmlWidth, P.Nz)):
        C_V.mLoss_CPML[nz] = (C_V.sigma_Hy[nz]*P.delT)/(2*P.permea_0)
        if(C_V.mLoss_CPML[nz]>= 1):
            C_V.C1[nz] = 0.01
        else:
            C_V.C1[nz] = (1-C_V.mLoss_CPML[nz])/(1+C_V.mLoss_CPML[nz])
        C_V.C2[nz] = P.delT/((1+C_V.mLoss_CPML[nz])*P.permea_0*C_V.kappa_Hy*P.dz)
        C_V.C3[nz] = P.delT/((1+C_V.mLoss_CPML[nz])*P.permea_0)
    return C_V.mLoss_CPML, C_V.C1, C_V.C2, C_V.C3   
    
    
def CPML_Psi_e_Update(V,P, C_V, C_P):   # recursive convolution for E field REF
    for nz in it.chain(range(0, P.pmlWidth), range(P.Nz-P.pmlWidth, P.Nz)):
        C_V.psi_Ex[nz] = C_V.beX[nz]*C_V.psi_Ex[nz] + C_V.ceX[nz]*(V.Hy[nz]-V.Hy[nz-1])
    return C_V.psi_Ex    

def CPML_Psi_m_Update(V,P, C_V, C_P):   # recursive convolution for H field REF
    for nz in it.chain(range(0, P.pmlWidth), range(P.Nz-1-P.pmlWidth, P.Nz-1)):
        C_V.psi_Hy[nz] = C_V.bmY[nz]*C_V.psi_Hy[nz] + C_V.cmY[nz]*(V.Ex[nz+1]-V.Ex[nz])
    return C_V.psi_Hy  


def CPML_HyUpdate(V,P, C_V, C_P):
    for nz in it.chain(range(0, P.pmlWidth), range(P.Nz-1-P.pmlWidth, P.Nz-1)):
        V.Hy[nz] = V.Hy[nz]*C_V.C1[nz] + (V.Ex[nz+1]-V.Ex[nz])*C_V.C2[nz] + C_V.psi_Hy[nz]*C_V.C3[nz]# + Cc Psi
    return V.Hy[0:P.pmlWidth], V.Hy[P.Nz-1-P.pmlWidth: P.Nz-1]


def CPML_ExUpdate(V,P, C_V, C_P):
    for nz in it.chain(range(1, P.pmlWidth), range(P.Nz-1-P.pmlWidth, P.Nz-1)):
        V.Ex[nz] = V.Ex[nz]*C_V.Ca[nz] + (V.Hy[nz]-V.Hy[nz-1])*C_V.Cb[nz] + C_V.psi_Ex[nz]*C_V.Cc[nz]# + Cc Psi
    return V.Ex[1:P.pmlWidth], V.Ex[P.Nz-1-P.pmlWidth: P.Nz-1]

def CPML_PEC(V, P, C_V, C_P):
    V.Hy[P.Nz-1] = V.Hy[P.Nz-1]*C_V.C1[P.Nz-1] + (0-V.Ex[P.Nz-1])*C_V.C2[P.Nz-1] + C_V.psi_Hy[P.Nz-1]*C_V.C3[P.Nz-1]
    V.Ex[0] = V.Ex[0]*C_V.Ca[0] + (V.Hy[0]-0)*C_V.Cb[0] + C_V.psi_Ex[0]*C_V.Cc[0]
    return V.Ex[0], V.Hy[P.Nz-1]

"""
Issue: feed fields back and forth. RETURN uphy etc etc , call
"""
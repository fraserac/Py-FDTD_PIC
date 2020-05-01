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

def sourceGen(T):
    pulse = np.exp(-(T - 30)*(T-30)/100)
    return pulse

def sourceGen2(P, T):
    pulse = np.sin(2*np.pi*P.freq_in*P.delT*T)
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
    V.Ex =np.zeros(P.Nz+1)
    V.Hy=np.zeros(P.Nz+1)
    V.Ex_History= [[]]*P.timeSteps
    V.Hy_History= [[]]*P.timeSteps
    V.Psi_Ex_History= [[]]*P.timeSteps
    V.Psi_Hy_History= [[]]*P.timeSteps
    V.Exs = []
    V.Hys = []
    V.polarisationCurr = np.zeros(P.Nz+1)
    V.Dx = np.zeros(P.Nz+1)
    V.tempVarPol = np.zeros(P.Nz+1)
    V.tempTempVarPol = np.zeros(P.Nz+1)
    
    return V.tempTempVarPol, V.polarisationCurr, V.Ex, V.Dx, V.Ex_History, V.Hy, V.Hy_History, V.Psi_Ex_History, V.Psi_Hy_History, V.Exs, V.Hys


def SmoothTurnOn(V,P):
    V.Exs = np.zeros(P.timeSteps)
    V.Hys = np.zeros(P.timeSteps)
    ppw =  P.c0 /(P.freq_in*P.dz)
    phase1 = False
    phase2 = False
    p1Ind = 0
    p2Ind = 0
    for timer in range(P.timeSteps):
        V.Exs[timer] = (np.sin(2.0*np.pi/ppw*(P.courantNo*timer)))
        V.Hys[timer] = (np.sin(2.0*np.pi/ppw*(P.courantNo*(timer+1))))
        if(V.Exs[timer] <0 and phase1 == False):
             p1Ind = timer 
             phase1 = True
             #print(p1Ind, "p1Ind")
        if(V.Exs[timer] >0 and phase1 == True and phase2 == False):
            p2Ind = timer 
            phase2 =True
            #print(p2Ind, "p2Ind")
        if(phase2 == True) and phase1 == True:
            V.Exs[timer] = 0.0
            V.Hys[timer] =0.0
        if(V.Hys[p2Ind] != 0 and phase2== True):
            V.Hys[p2Ind] = 0.0;
    return V.Exs, V.Hys   


def EmptySpaceCalc(V,P): 
    V.UpHyMat = np.zeros(P.Nz+1) 
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

    


 

def UpdateCoef(V,P):
    UpHyBackground = (1/P.CharImp)*P.courantNo
    UpExBackground = P.CharImp*P.courantNo
    UpHyMat = np.zeros(P.Nz+1) 
    UpExMat = np.zeros(P.Nz+1)
    
    for k in range(P.Nz):
        UpExMat[k]= UpExBackground/V.epsilon[k]
        UpHyMat[k]= UpHyBackground/V.mu[k]

    return UpHyMat, UpExMat



def HyTfSfCorr(V, P, counts):
     V.Hy[P.nzsrc-1] -= V.Exs[counts]/P.CharImp
     return V.Hy[P.nzsrc-1]

def ExTfSfCorr(V,P, counts):
    V.Ex[P.nzsrc] += V.Hys[counts]
    return V.Ex[P.nzsrc]

def CPML_FieldInit(V,P, C_V, C_P): 
    C_V.kappa_Ex =np.zeros(P.Nz+1)
    C_V.kappa_Hy = np.zeros(P.Nz+1)
    C_V.psi_Ex =np.zeros(P.Nz+1)
    C_V.psi_Hy = np.zeros(P.Nz+1)
    C_V.alpha_Ex= np.zeros(P.Nz+1)
    C_V.alpha_Hy= np.zeros(P.Nz+1)
    C_V.sigma_Ex =np.zeros(P.Nz+1)    
    C_V.sigma_Hy = np.zeros(P.Nz+1)
    C_V.beX =np.zeros(P.Nz+1)
    C_V.bmY =np.zeros(P.Nz+1)
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
    for nz in range(P.Nz):
        nz1= nz+1
        if (nz1 <=P.pmlWidth): 
            C_V.kappa_Ex[nz] =1+(C_P.kappaMax-1)*((P.pmlWidth-nz1)/(P.pmlWidth-1))**C_P.r_scale
            C_V.sigma_Ex[nz] = C_P.sigmaOpt*((P.pmlWidth - nz1)/(P.pmlWidth-1))**C_P.r_scale   
            C_V.alpha_Ex[nz] = C_P.alphaMax*(nz1/(P.pmlWidth-1))**C_P.r_a_scale 

        elif nz >= P.Nz+2 -(P.pmlWidth):
            C_V.kappa_Ex[nz] = C_V.kappa_Ex[jj]
            C_V.sigma_Ex[nz] = C_V.sigma_Ex[jj]
            C_V.alpha_Ex[nz] = C_V.alpha_Ex[jj]
            jj-=1
            
        else:
            C_V.kappa_Ex[nz] = 1
            C_V.sigma_Ex[nz] = 0
            C_V.alpha_Ex[nz] = 0

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
    return C_V.bmY, C_V.cmY

def CPML_Ex_Update_Coef(V,P, C_V, C_P):
    for nz in range(0, P.Nz-1):
        C_V.eLoss_CPML[nz] = (C_V.sigma_Ex[nz]*P.delT)/(2*P.permit_0)
        
        C_V.Ca[nz] = V.UpExSelf[nz]
        C_V.Cb[nz] =V.UpExHcompsCo[nz]*V.UpExMat[nz]
        C_V.Cc[nz] = P.delT/((1+C_V.eLoss_CPML[nz])*P.permit_0)
    return C_V.eLoss_CPML, C_V.Ca, C_V.Cb, C_V.Cc    


def CPML_Hy_Update_Coef(V,P, C_V, C_P):
    for nz in range(0, P.Nz-2):
        C_V.C1[nz] =1
        C_V.C2[nz] = -P.delT/P.permea_0
        C_V.C3[nz] = P.delT/  ((1+C_V.mLoss_CPML[nz])*P.permea_0)
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
    
    


def CPML_Psi_e_Update(V,P, C_V, C_P):  
    for nz in range(1, P.Nz-1): 
        C_V.psi_Ex[nz] = C_V.beX[nz]*C_V.psi_Ex[nz] + C_V.ceX[nz]*(V.Hy[nz]-V.Hy[nz-1])
        V.Ex[nz] = V.Ex[nz] + C_V.Cb[nz]*C_V.psi_Ex[nz]
    
    return C_V.psi_Ex, V.Ex 

 
def CPML_Psi_m_Update(V,P, C_V, C_P):  
    for nz in range(0, P.Nz-2): 
        C_V.psi_Hy[nz] = C_V.bmY[nz]*C_V.psi_Hy[nz] + C_V.cmY[nz]*(V.Ex[nz]-V.Ex[nz+1])
        V.Hy[nz] = V.Hy[nz] + C_V.C2[nz]*C_V.psi_Hy[nz]
    
    return C_V.psi_Hy, V.Hy 



def CPML_HyUpdate(V,P, C_V, C_P):
    for nz in range(0, P.Nz-2):
        V.Hy[nz] = V.Hy[nz]*V.UpHySelf[nz] + (V.Ex[nz+1]-V.Ex[nz])*V.UpHyMat[nz]*V.UpHyEcompsCo[nz]#*C_V.den_Hydz[nz]
    return V.Hy


def CPML_ExUpdate(V,P, C_V, C_P):
    for nz in range(1, P.Nz-1):
        V.Ex[nz] = V.Ex[nz]*V.UpExSelf[nz] + (V.Hy[nz]-V.Hy[nz-1])*V.UpExHcompsCo[nz]*V.UpExMat[nz]#*C_V.den_Exdz[nz]
    return V.Ex


def CPML_PEC(V, P, C_V, C_P):
    V.Ex[0] =0
    return V.Ex[0], V.Ex[P.Nz-1]

def CPML_PMC(V,P,C_V, C_P):
    V.Hy[P.Nz-1]=0
    return V.Hy[P.Nz-1]

def ADE_TempPolCurr(V,P):
     for nz in range(1, P.Nz-1):
         V.tempTempVarPol[nz] = V.tempVarPol[nz]
         V.tempVarPol[nz] = V.polarisationCurr[nz]  
     return V.tempTempVarPol, V.tempVarPol


def ADE_PolarisationCurrent_Ex(V, P, C_V, C_P):
    D= (1/P.delT**2)+(V.gammaE/(2*P.delT))
    #print("D ", D)
    A = ((2/P.delT**2)-V.omega_0E**2)/D
    #print("A", A)
    B = ((V.gammaE/(2*P.delT))-1/P.delT**2)/D
    #print(B)
    C = (P.permit_0*(V.plasmaFreqE**2))/D
    #print(C)
  
    for nz in range (int(P.materialFrontEdge-1), int(P.materialRearEdge)):
        V.polarisationCurr[nz] = A*V.tempVarPol[nz]+ B*V.tempTempVarPol[nz] +C*V.Ex[nz]
    return V.polarisationCurr

def ADE_HyUpdate(V, P, C_V, C_P):
    for nz in range(1, P.Nz-1):
        V.Hy[nz] = V.Hy[nz]*V.UpHySelf[nz] + (V.Ex[nz+1]-V.Ex[nz])*V.UpHyMat[nz]#*C_V.den_Hydz[nz]
    return V.Hy
        
def ADE_MyUpdate():
    
    pass

def ADE_ExUpdate(V, P, C_V, C_P): 
    for nz in range(1, int(P.materialFrontEdge-1)):
        V.Ex[nz] =V.UpExSelf[nz]*V.Ex[nz] + (V.Hy[nz]-V.Hy[nz-1])*V.UpExMat[nz]*C_V.den_Exdz[nz]#*C_V.den_Exdz[nz]
        
    if P.materialRearEdge < P.Nz-1:
        for nzz in range(int(P.materialRearEdge-1), P.Nz):
            V.Ex[nzz] = V.UpExSelf[nzz]*V.Ex[nzz] + (V.Hy[nzz]-V.Hy[nzz-1])*V.UpExMat[nzz]#*C_V.den_Exdz[nz]
    return V.Ex

def ADE_ExCreate(V, P, C_V, C_P):
    for nz in range(int(P.materialFrontEdge-1), int(P.materialRearEdge)):
       V.Ex[nz] =(V.Dx[nz] - V.polarisationCurr[nz])/P.permit_0
    return V.Ex

def ADE_DxUpdate(V, P, C_V, C_P):
    for nz in range(int(P.materialFrontEdge-1), int(P.materialRearEdge)):
        V.Dx[nz] = V.Dx[nz] +(V.Hy[nz] - V.Hy[nz-1])*P.delT/P.dz
        V.Ex[nz] =(V.Dx[nz] - V.polarisationCurr[nz])/P.permit_0
    return V.Dx, V.Ex


def ADE_NonLinMyUpdate():
    pass

def ADE_NonLinPxUpdate():
    pass

def AnalyticalReflectionE(V, P):
    epsNum = 1+((V.plasmaFreqE)**2)
    epsDom = (V.omega_0E**2-2*np.pi*P.freq_in**2 + 1j*V.gammaE*2*np.pi*P.freq_in)
    eps0 = P.permit_0   
    epsilon = (epsNum/epsDom)
    reflection = (np.sqrt(eps0*epsilon) - np.sqrt(eps0))/(np.sqrt(eps0*epsilon) + np.sqrt(eps0))
    trans = 2*np.sqrt(epsilon*eps0)/(np.sqrt(eps0*epsilon) + np.sqrt(eps0))
    trans1 =abs(trans)/(abs(trans)+abs(reflection))
    reflection1 = abs(reflection)/(abs(trans)+abs(reflection))
   # print(trans1)
    #print(reflection1)
    #print(epsilon)
    #print(epsNum)
    #print(epsDom)
    #print(trans1+reflection1)
    return reflection1

"""
pseudocode for ade cpml.

auxiliary var -polarisationCurr, 

Update  equation for polCurr has to create polcurr too, 
added in psi 

"""

def SpatialFiltering():
    
    pass

def SymbolicRegression():
    pass

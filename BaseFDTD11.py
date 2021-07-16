# -*- coding: utf-8 -*-
"""
This is the start of my FDTD code. So far 1D.
This is currently a 1D work in progress FDTD
This script is the engine that will calculate field updates and plotting functions
The plotting and animation functionality may eventually be moved to a separate script.
This script will eventually just contain the update equations and animation



"""
import time

import numpy as np
import matplotlib.pylab as plt
from scipy import signal as sign
import sys 
import numba
from numba import njit as nj
from numba import jit,prange

from numba import int32, float32, int64, float64, boolean, complex128
from Tests.Integration_Tester import testerFuncVector as tfv
import CubicEquationSolver


### CONSIDER IMPLEMENTING DIFFERENT SOURCES RF, gaussian, ricker, as well as sine pulse
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
            if P.timeSteps > 2**17:
                raise ValueError('timeSteps max too large')
                break
            if P.Nz > 25000:
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
    #V.Ex_History= [[]]*P.timeSteps
    #V.Hy_History= [[]]*P.timeSteps
    #V.Psi_Ex_History= [[]]*P.timeSteps
    #V.Psi_Hy_History= [[]]*P.timeSteps
    #V.Exs = []
    #V.Hys = []
    V.polarisationCurr = np.zeros(P.Nz+1)
    V.Dx = np.zeros(P.Nz+1)
    V.tempVarPol = np.zeros(P.Nz+1)
    V.tempTempVarPol = np.zeros(P.Nz+1)
    V.tempVarE = np.zeros(P.Nz+1)
    V.tempTempVarE = np.zeros(P.Nz+1)
    
    return V.tempVarPol, V.tempTempVarE, V.tempVarE, V.tempTempVarPol, V.polarisationCurr, V.Ex, V.Dx, V.Hy



def Gaussian(V,P):
    #use signal scipy?
    t = np.arange(P.timeSteps)
    fc = 200
    tau = fc*2.2
    
    gaussArg = ((t-tau)*(t-tau))/(fc*fc)*np.cos(2*np.pi*P.freq_in*(t-tau))
    out = np.exp(-gaussArg)*2
    out = P.Amplitude*out/P.courantNo
    #this creates Exs, functions like smoothturnon for TFSF
    return out

def SineCont(V,P, rep, interv):
    pass

def Ricker(V,P):
    pass

def SmoothTurnOn(V,P, tempfreq=0):   # prevents discontinuity in source pattern from causing instability
    if tempfreq != 0:
        frq = tempfreq
    else:
        frq = P.freq_in
    ppw =  P.c0 /(frq*P.dz)
    Exs =[]
    Hys = []

    for timer in range(P.timeSteps):
        if(timer*P.delT < P.period*P.Periods):
            Exs.append(float(np.sin(2.0*np.pi/ppw*(P.courantNo*timer))))
            Hys.append(float(np.sin(2.0*np.pi/ppw*(P.courantNo*(timer+1)))))
        elif(timer*P.delT >= P.period*P.Periods):  
            Exs.append(0)
            Hys.append(0)
    return Exs, Hys   
    

###@nj
def EmptySpaceCalc(V,P): 
    V.UpHyMat = np.ones(P.Nz+1) 
    V.UpExMat = np.ones(P.Nz+1)
    UpHyBackground = (1/P.CharImp)*P.courantNo
    UpExBackground = P.CharImp*P.courantNo
    V.UpExMat *=UpExBackground
    V.UpHyMat *=UpHyBackground
 
    return V.UpHyMat, V.UpExMat



def Material(V,P):  # This was used originally before dispersive media ADE, is now redundant
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

    
### CHECK COEFFS ABOVE DO NOT SHOW UP IN CODE

 

def UpdateCoef(V,P):  # 
    UpHyBackground = (1/P.CharImp)*P.courantNo
    UpExBackground = P.CharImp*P.courantNo
    UpHyMat = np.ones(P.Nz+1) 
    UpExMat = np.ones(P.Nz+1)
    
    #for k in range(P.Nz):
    UpExMat*=UpExBackground
    UpHyMat*=UpHyBackground
    for k in range(P.materialFrontEdge, P.materialRearEdge):
        UpExMat[k] = UpExMat[k]/V.epsilon[k]
        UpHyMat[k] = UpHyMat[k]/V.mu[k]
        

    return UpHyMat, UpExMat

forlocals ={'Exs' : float64[:], 'counts' : int32}  #Numba need predefined typing
@jit(nopython=True, locals= forlocals)
def HyTfSfCorr(V, P, counts, Exs):
     V.Hy[P.nzsrc-1] -= Exs[counts]/P.CharImp
     return V.Hy[P.nzsrc-1]#
 
forlocals2 ={'Hys' : float64[:], 'counts' : int32} ### for tf/sf
@jit(nopython=True, locals= forlocals2)
def ExTfSfCorr(V,P, counts, Hys):
    V.Ex[P.nzsrc] += Hys[counts]
    return V.Ex[P.nzsrc]
@nj
def CPML_FieldInit(V,P, C_V, C_P):    #initialise cpml 
    C_V.kappa_Ex =np.ones(P.Nz+1)
    C_V.kappa_Hy = np.ones(P.Nz+1)
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
    C_V.den_Exdz = np.ones(len(V.Ex))
    C_V.den_Hydz = np.ones(len(V.Ex))
    return C_V


@nj
def CPML_ScalingCalc(V, P, C_V, C_P):
    kappa_Ex_n = np.ones(P.pmlWidth)
    sigma_Ex_n = np.zeros(P.pmlWidth)
    alpha_Ex_n = np.zeros(P.pmlWidth)
    kappa_Ex_p=np.ones(P.pmlWidth)
    sigma_Ex_p = np.zeros(P.pmlWidth)
    alpha_Ex_p = np.zeros(P.pmlWidth)
    kappa_Hy_n = np.ones(P.pmlWidth)
    sigma_Hy_n = np.zeros(P.pmlWidth)
    alpha_Hy_n = np.zeros(P.pmlWidth)
    kappa_Hy_p=np.ones(P.pmlWidth)
    sigma_Hy_p = np.zeros(P.pmlWidth)
    alpha_Hy_p = np.zeros(P.pmlWidth)
    
    nz =np.arange(0, P.pmlWidth,1)
    kappa_Ex_n[0:P.pmlWidth] = 1+(C_P.kappaMax-1)*((P.pmlWidth-nz)/(P.pmlWidth))**C_P.r_scale
    sigma_Ex_n[0:P.pmlWidth] = C_P.sigmaOpt*((P.pmlWidth-nz)/(P.pmlWidth))**C_P.r_scale
    alpha_Ex_n[0:P.pmlWidth] =  C_P.alphaMax*((nz+1)/(P.pmlWidth))**C_P.r_a_scale 
    
    
    kappa_Hy_n[0:P.pmlWidth] = kappa_Ex_n[0:P.pmlWidth]
    sigma_Hy_n[0:P.pmlWidth] = sigma_Ex_n[0:P.pmlWidth]
    alpha_Hy_n[0:P.pmlWidth] = alpha_Ex_n[0:P.pmlWidth]
    
    #right side scaling vars
    kappa_Ex_p[0:P.pmlWidth] = kappa_Ex_n[::-1] 
    sigma_Ex_p[0:P.pmlWidth] = sigma_Ex_n[::-1]
    alpha_Ex_p[0:P.pmlWidth] = alpha_Ex_n[::-1]
    
    kappa_Hy_p[0:P.pmlWidth] = kappa_Hy_n[::-1]
    sigma_Hy_p[0:P.pmlWidth] = sigma_Hy_n[::-1]
    alpha_Hy_p[0:P.pmlWidth] = alpha_Hy_n[::-1]
    
    #set up C_V
    C_V.kappa_Ex[:P.pmlWidth] = kappa_Ex_n 
    C_V.kappa_Ex[len(V.Ex)-P.pmlWidth: len(V.Ex)] = kappa_Ex_p
    
    C_V.sigma_Ex[:P.pmlWidth] = sigma_Ex_n 
    C_V.sigma_Ex[len(V.Ex)-P.pmlWidth: len(V.Ex)] = sigma_Ex_p
    
    C_V.alpha_Ex[:P.pmlWidth] = alpha_Ex_n 
    C_V.alpha_Ex[len(V.Ex)-P.pmlWidth: len(V.Ex)] = alpha_Ex_p
    
    C_V.kappa_Hy = C_V.kappa_Ex
    C_V.sigma_Hy = C_V.sigma_Ex
    C_V.alpha_Hy = C_V.alpha_Ex

    
        
    return C_V.sigma_Ex, C_V.sigma_Hy, C_V.alpha_Ex,  C_V.alpha_Hy, C_V.kappa_Ex, C_V.kappa_Hy

@nj
def CPML_Ex_RC_Define(V, P, C_V, C_P):
    #window = sign.hann(P.Nz+1)
    
    C_V.beX[:P.pmlWidth] = np.exp(-((C_V.sigma_Ex[:P.pmlWidth]*P.delT/(C_V.kappa_Ex[:P.pmlWidth]*P.permit_0))+((C_V.alpha_Ex[:P.pmlWidth]*P.delT)/P.permit_0))) 
    C_V.beX[len(V.Ex)-P.pmlWidth:] = np.exp(-((C_V.sigma_Ex[len(V.Ex)-P.pmlWidth:]*P.delT/(C_V.kappa_Ex[len(V.Ex)-P.pmlWidth:]*P.permit_0))+((C_V.alpha_Ex[len(V.Ex)-P.pmlWidth:]*P.delT)/P.permit_0))) 
    
    C_V.ceX[:P.pmlWidth] = (C_V.beX[:P.pmlWidth]-1)*C_V.sigma_Ex[:P.pmlWidth]/((C_V.sigma_Ex[:P.pmlWidth]*C_V.kappa_Ex[:P.pmlWidth]+C_V.alpha_Ex[:P.pmlWidth]*C_V.kappa_Ex[:P.pmlWidth]*C_V.kappa_Ex[:P.pmlWidth]))
    C_V.ceX[len(V.Ex)-P.pmlWidth:] = (C_V.beX[len(V.Ex)-P.pmlWidth:]-1)*C_V.sigma_Ex[len(V.Ex)-P.pmlWidth:]/((C_V.sigma_Ex[len(V.Ex)-P.pmlWidth:]*C_V.kappa_Ex[len(V.Ex)-P.pmlWidth:]+C_V.alpha_Ex[len(V.Ex)-P.pmlWidth:]*C_V.kappa_Ex[len(V.Ex)-P.pmlWidth:]*C_V.kappa_Ex[len(V.Ex)-P.pmlWidth:]))

        
   # C_V.beX = sign.convolve(C_V.beX, window, mode= 'same')/np.sum(window)  
    return C_V.beX, C_V.ceX

@nj
def CPML_HY_RC_Define(V, P, C_V, C_P):
    C_V.bmY[:P.pmlWidth] = np.exp(-((C_V.sigma_Hy[:P.pmlWidth]*P.delT/(C_V.kappa_Hy[:P.pmlWidth]*P.permit_0))+((C_V.alpha_Hy[:P.pmlWidth]*P.delT)/P.permit_0))) 
    C_V.bmY[len(V.Hy)-P.pmlWidth:] = np.exp(-((C_V.sigma_Hy[len(V.Hy)-P.pmlWidth:]*P.delT/(C_V.kappa_Hy[len(V.Hy)-P.pmlWidth:]*P.permit_0))+((C_V.alpha_Hy[len(V.Hy)-P.pmlWidth:]*P.delT)/P.permit_0))) 
    
    C_V.cmY[:P.pmlWidth] = (C_V.bmY[:P.pmlWidth]-1)*C_V.sigma_Hy[:P.pmlWidth]/((C_V.sigma_Hy[:P.pmlWidth]*C_V.kappa_Hy[:P.pmlWidth]+C_V.alpha_Hy[:P.pmlWidth]*C_V.kappa_Hy[:P.pmlWidth]*C_V.kappa_Hy[:P.pmlWidth])*P.dz)
    C_V.cmY[len(V.Hy)-P.pmlWidth:] = (C_V.bmY[len(V.Hy)-P.pmlWidth:]-1)*C_V.sigma_Hy[len(V.Hy)-P.pmlWidth:]/((C_V.sigma_Hy[len(V.Hy)-P.pmlWidth:]*C_V.kappa_Hy[len(V.Hy)-P.pmlWidth:]+C_V.alpha_Hy[len(V.Hy)-P.pmlWidth:]*C_V.kappa_Hy[len(V.Hy)-P.pmlWidth:]*C_V.kappa_Hy[len(V.Hy)-P.pmlWidth:])*P.dz)

    return C_V.bmY, C_V.cmY


def CPML_Ex_Update_Coef(V,P, C_V, C_P):
    
    betaE = (0.5*V.plasmaFreqE*V.plasmaFreqE*P.permit_0*P.delT)/(1+0.5*V.gammaE*P.delT)  # This is only called once 
    kapE = (1-0.5*V.gammaE*P.delT)/(1+0.5*V.gammaE*P.delT)
    a= ((2*P.permit_0-betaE*P.delT)/(2*P.permit_0+betaE*P.delT))
    b =  ((2*P.delT)/(2*P.permit_0+betaE*P.delT))*(1/P.courantNo)
    #breakpoint()
    for nz in range(0, P.pmlWidth):
        C_V.eLoss_CPML[nz] = (C_V.sigma_Ex[nz]*P.delT)/(2*P.permit_0)
        
        C_V.Ca[nz] = a
        C_V.Cb[nz] = V.UpExHcompsCo[nz]*V.UpExMat[nz]
        C_V.Cc[nz] = P.delT/((1+C_V.eLoss_CPML[nz])*P.permit_0)
    for nz in range(len(V.Hy)-1, len(V.Hy)-P.pmlWidth, -1):
        C_V.eLoss_CPML[nz] = (C_V.sigma_Ex[nz]*P.delT)/(2*P.permit_0)
        
        C_V.Ca[nz] = a
        C_V.Cb[nz] = V.UpExHcompsCo[nz]*V.UpExMat[nz]
        C_V.Cc[nz] = P.delT/((1+C_V.eLoss_CPML[nz])*P.permit_0)
    return C_V.eLoss_CPML, C_V.Ca, C_V.Cb, C_V.Cc    

#@nj
def CPML_Hy_Update_Coef(V,P, C_V, C_P):
    for nz in range(0, P.pmlWidth):  # Some of these could be redundant check this 
        C_V.C1[nz] =1
        C_V.C2[nz] = P.delT/P.permea_0
        C_V.C3[nz] = P.delT/  ((1+C_V.mLoss_CPML[nz])*P.permea_0)
    for nz in range(len(V.Hy)-1, len(V.Hy)-P.pmlWidth, -1):
        C_V.C1[nz] =1
        C_V.C2[nz] = P.delT/P.permea_0
        C_V.C3[nz] = P.delT/  ((1+C_V.mLoss_CPML[nz])*P.permea_0)
    return C_V.mLoss_CPML, C_V.C1, C_V.C2, C_V.C3   

@nj
def denominators(V, P, C_V, C_P):
    jj = P.pmlWidth
       # set denom as vector of ones default
       
    
    for j in range(0,len(V.Hy)): 
        if j <= P.pmlWidth  and P.CPMLXm ==True:
            C_V.den_Hydz[j]= 1/(C_V.kappa_Hy[j])
        elif j>= len(V.Hy)-P.pmlWidth and P.CPMLXp == True:
            C_V.den_Hydz[j] = 1/(C_V.kappa_Hy[jj])
            
            jj-=1
        else:
            C_V.den_Hydz[j] = 1/(1)
    jj = P.pmlWidth-1
    for j in range(0,len(V.Ex)): 
        if j <= P.pmlWidth and P.CPMLXm ==True:
            C_V.den_Exdz[j]= 1/(C_V.kappa_Ex[j])
        elif j>=len(V.Ex)-P.pmlWidth and P.CPMLXp ==True:
            C_V.den_Exdz[j] = 1/(C_V.kappa_Ex[jj])
            jj-=1
        else:
            C_V.den_Exdz[j] = 1/(1)       
    
    return C_V.den_Exdz, C_V.den_Hydz
    
    

forLocals = {'zeta0': float64[:], 'zeta1':float64[:]}

#@jit(nopython = True, locals=forLocals, debug =True)
@nj
def CPML_Psi_e_Update(V,P, C_V, C_P): 
   #zeta0 =(-C_V.sigma_Ex/(C_V.alpha_Ex*C_V.kappa_Ex*C_V.kappa_Ex+C_V.sigma_Ex*C_V.kappa_Ex))*(1-np.exp(-(C_V.alpha_Ex*P.delT)))
   #zeta1 = (-C_V.sigma_Ex/(C_V.alpha_Ex*C_V.kappa_Ex*C_V.kappa_Ex+C_V.sigma_Ex*C_V.kappa_Ex))*(C_V.alpha_Ex/P.delT)*(1-(1+C_V.alpha_Ex*P.delT)*np.exp(-(C_V.alpha_Ex*P.delT)))
   if P.CPMLXm ==True:
       for nz in range(1, P.pmlWidth): 
            C_V.psi_Ex[nz] = C_V.beX[nz]*C_V.psi_Ex[nz] +C_V.ceX[nz]*(V.Hy[nz]-V.Hy[nz-1])
            V.Ex[nz] = V.Ex[nz] - C_V.Cb[nz]*C_V.psi_Ex[nz]
   if P.CPMLXp ==True:
       for nz in range(len(V.Ex)- P.pmlWidth, len(V.Ex)): 
            C_V.psi_Ex[nz] = C_V.beX[nz]*C_V.psi_Ex[nz] + C_V.ceX[nz]*(V.Hy[nz]-V.Hy[nz-1])
            V.Ex[nz] = V.Ex[nz] - C_V.Cb[nz]*C_V.psi_Ex[nz]
   return C_V.psi_Ex, V.Ex


########################### DELIBERATELY BROKEN, FIX THIS TOMORROW!
#@jit(nopython = True, locals=forLocals, debug =True)
@nj
def CPML_Psi_m_Update(V,P, C_V, C_P): 
    #zeta0 =(-C_V.sigma_Ex/(C_V.alpha_Ex*C_V.kappa_Ex*C_V.kappa_Ex+C_V.sigma_Ex*C_V.kappa_Ex))*(1-np.exp(-(C_V.alpha_Ex*P.delT)))
    #zeta1 = (-C_V.sigma_Ex/(C_V.alpha_Ex*C_V.kappa_Ex*C_V.kappa_Ex+C_V.sigma_Ex*C_V.kappa_Ex))*(C_V.alpha_Ex/P.delT)*(1-(1+C_V.alpha_Ex*P.delT)*np.exp(-(C_V.alpha_Ex*P.delT)))
    if P.CPMLXm ==True:
       for nz in range(1, P.pmlWidth): 
        C_V.psi_Hy[nz] = C_V.bmY[nz]*C_V.psi_Hy[nz]+ C_V.cmY[nz]*(V.Ex[nz+1]-V.Ex[nz])
        V.Hy[nz] = V.Hy[nz] + C_V.C2[nz]*C_V.psi_Hy[nz]
    if P.CPMLXp ==True:
       for nz in range(len(V.Ex)- P.pmlWidth, len(V.Ex)-1): 
            C_V.psi_Hy[nz] = C_V.bmY[nz]*C_V.psi_Hy[nz] + C_V.cmY[nz]*(V.Ex[nz+1]-V.Ex[nz])
            V.Hy[nz] = V.Hy[nz] +C_V.C2[nz]*C_V.psi_Hy[nz]
    return C_V.psi_Hy, V.Hy


@nj
def CPML_HyUpdate(V,P, C_V, C_P):
    for nz in range(0, len(V.Hy)):
        V.Hy[nz] = V.Hy[nz] + (V.Ex[nz+1]-V.Ex[nz])*(1/P.courantNo)
    return V.Hy


@nj
def CPML_ExUpdate(V,P, C_V, C_P):
    
    V.Ex[1:-1] = V.Ex[1:-1]*V.UpExSelf[1:-1] + (V.Hy[1:-1]-V.Hy[:-2])*V.UpExHcompsCo[1:-1]*V.UpExMat[1:-1]#V.Jx[nz])*V.UpExHcompsCo[nz]*V.UpExMat[nz]#*C_V.den_Exdz[nz]
    if P.CPMLXm ==True:
        V.Ex[1:P.pmlWidth] = 1*V.Ex[1:P.pmlWidth]*V.UpExSelf[1:P.pmlWidth] + (V.Hy[1:P.pmlWidth]-V.Hy[:P.pmlWidth-1])*V.UpExHcompsCo[1:P.pmlWidth]*V.UpExMat[1:P.pmlWidth]
    if P.CPMLXp ==True:
        V.Ex[len(V.Ex)-P.pmlWidth+1:-1] = 1*V.Ex[len(V.Ex)-P.pmlWidth+1:-1]*V.UpExSelf[len(V.Ex)-P.pmlWidth+1:-1] + (V.Hy[len(V.Ex)-P.pmlWidth+1:-1]-V.Hy[len(V.Ex)-P.pmlWidth:-2])*V.UpExHcompsCo[len(V.Ex)-P.pmlWidth+1:-1]*V.UpExMat[len(V.Ex)-P.pmlWidth+1:-1]
    return V.Ex

@nj
def CPML_PEC(V, P, C_V, C_P):
    V.Ex[0] =0
    return V.Ex[0], V.Ex[-1]
@nj
def CPML_PMC(V,P,C_V, C_P):
    V.Hy[-1]=0
    return V.Hy[-1]


def CPML_Plot_Scaling(V, P, C_V, C_P):
    ## plot all scaling vars, write to folder same directory as video so each run outputs all graphs for debugging. 
    ## Also consider outputting all interesting variables in plots. 
    #Sigma E, Sigma H, plot entire C_V, C_P classes. Iterate through all? Reflection?
    fig, ax = plt.subplots()
    ax.plot(C_V.kappa_Ex)
    ax.plot(C_V.sigma_Ex)
    ax.plot(C_V.alpha_Ex)
    ax.set_title("Scaling parameters Ex, CPML")
    ax.legend(["kappa Ex","sigma Ex", "alpha Ex"])
    
    fig2, ax2 = plt.subplots()
    ax2.plot(C_V.kappa_Hy)
    ax2.plot(C_V.sigma_Hy)
    ax2.plot(C_V.alpha_Hy)
    ax2.set_title("Scaling parameters Hy, CPML")
    ax2.legend(["kappa Hy","sigma Hy", "alpha Hy"])
    
    """
     self.kappaMax =1# 'Stretching co-ordinate of pml, to minimise numerical dispersion set it as 1' : DOI: 10.22190/FUACR1703229G see conclusion
        self.r_scale =4 #Within ideal bounds see Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17 (scaling power is called 'm' )
        self.r_a_scale=1
        self.sigmaEMax=1*(0.8*(1)/(dz*(sci.constants.mu_0/sci.constants.epsilon_0)**0.5))#1.1*sigmaOpt # Within ideal bounds for value, : Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
        self.sigmaHMax =1*(0.8*(1)/(dz*(sci.constants.mu_0/sci.constants.epsilon_0)**0.5))#1.1*sigmaOpt # See International Journal of Computer Science and Network Security, VOL.18 No.12, December 2018, page 4 right hand side.
        self.sigmaOpt  =1*(0.8*(1)/(dz*(sci.constants.mu_0/sci.constants.epsilon_0)**0.5))
    #Optimal value of pml conductivity at far end of pml: DOI: 10.22190/FUACR1703229G see equation 13
        self.alphaMax=0.02# with bounds of ideal cpm
        
        self.kappa_Ex = np.zeros(Nz)
        self.kappa_Hy = np.zeros(Nz)
        self.psi_Ex =  np.zeros(Nz)
        self.psi_Ex_Probe = np.zeros(timeSteps)
        self.psi_Hy =  np.zeros(Nz)
        self.psi_Hy_Probe = np.zeros(timeSteps)
        self.alpha_Ex = np.zeros(Nz)
        self.alpha_Hy = np.zeros(Nz)
        self.sigma_Ex =np.zeros(Nz)   # specific spatial value of conductivity 
        self.sigma_Hy = np.zeros(Nz)
        self.beX =np.zeros(Nz)
        self.bmY =np.zeros(Nz)#np.exp(-(sigmaHy/(permea_0*kappa_Hy) + alpha_Hy/permea_0 )*delT)
        self.ceX = np.zeros(Nz)
        self.cmY = np.zeros(Nz)
        self.Ca = np.zeros(Nz)
        self.Cb = np.zeros(Nz)
        self.Cc = np.zeros(Nz)
        self.C1 = np.zeros(Nz)
        self.C2 = np.zeros(Nz)
        self.C3 = np.zeros(Nz)
        self.eLoss_CPML =np.zeros(Nz)   # sigma e* delT/2*epsilon
        self.mLoss_CPML = np.zeros(Nz)
        self.den_Hydz = np.zeros(Nz)
        self.den_Exdz = np.zeros(Nz) 
        self.tempTempVarPsiEx = np.zeros(Nz)
        self.tempVarPsiEx = np.zeros(Nz)
        self.tempTempVarPsiHy = np.zeros(Nz)
        self.tempVarPsiHy = np.zeros(Nz)
        self.psi_Ex_Old= np.zeros(timeSteps)
        self.psi_Hy_Old = np.zeros(timeSteps)
        
    
    """
    pass

#@nj
def ADE_TempPolCurr(V,P, C_V, C_P):
    # V.tempTempTest = V.tempTest
    # V.tempTest = V.test
     """
     strange issue with ndarrays in for loop, can't have temporary previous 
     field stored for some reason makes current and previous fields equal?
     Works with as expected as list however, so convert to list here, 
     then back to ndarray before returning.
     This func can't be numba func because of dynamic typing.
     """
     tempTempVarPol = V.tempTempVarPol
     tempTempVarPol = tempTempVarPol.tolist()
     tempVarPol = V.tempVarPol
     tempVarPol = tempVarPol.tolist()
     polarisationCurr = V.polarisationCurr
     polarisationCurr = polarisationCurr.tolist()
     tempTempVarE = V.tempTempVarE
     tempTempVarE= tempTempVarE.tolist()
     tempVarE =V.tempVarE
     tempVarE = tempVarE.tolist()
     Ex = V.Ex
     Ex = Ex.tolist()
     ##########
     
     tempTempVarPol = tempVarPol
     tempVarPol = polarisationCurr         
     V.tempTempVarDx = V.tempVarDx
     V.tempVarDx = V.Dx
     tempTempVarE = tempVarE 
     tempVarE = Ex 
     V.tempTempVarHy = V.tempVarHy
     V.tempVarHy=V.Hy
     V.tempTempVarJx = V.tempVarJx
     V.tempVarJx = V.Jx
     C_V.tempTempVarPsiEx = C_V.tempVarPsiEx
     C_V.tempVarPsiEx = C_V.psi_Ex
     C_V.tempTempVarPsiHy = C_V.tempVarPsiHy
     C_V.tempVarPsiHy = C_V.psi_Hy
     
     
     #### reconvert 
     
     V.tempTempVarPol = np.asarray(tempTempVarPol)
     V.tempVarPol = np.asarray(tempVarPol)
     V.polarisationCurr = np.asarray(polarisationCurr)
     V.tempTempVarE = np.asarray(tempTempVarE)
     V.tempVarE =np.asarray(tempVarE)
     V.Ex = np.asarray(Ex)
     if np.max(np.abs(V.polarisationCurr)) >0:
         if np.max(np.abs(V.polarisationCurr - V.tempTempVarPol)) ==0:
             breakpoint()
     return V.tempTempVarPol, V.tempVarPol, V.tempVarE, V.tempTempVarE, V.tempTempVarHy, V.tempVarHy, V.tempTempVarJx, V.tempVarJx, C_V.tempTempVarPsiEx, C_V.tempVarPsiEx, C_V.tempTempVarPsiHy, C_V.tempVarPsiHy

forlocalsJ ={'coef' : float64, 'V.Jx' : float64[:]} 

@jit(nopython=True, locals=forlocalsJ)
def ADE_JxUpdate(V,P, C_V,C_P): #timestep t+1/2, FM?    NOT USING THIS IN CURRENT RENDITION
  
   
   # betaE = (0.5*V.plasmaFreqE**2*P.permit_0*P.delT)/(1+ 0.5*V.gammaE*P.delT)
    #kapE = (1-0.5*V.gammaE*P.delT)/(1+0.5*V.gammaE*P.delT)
    
    #selfCo = (2*P.permit_0-betaE*P.delT)/(2*P.permit_0+betaE*P.delT)
    
    #HCurlCo= (2*P.delT)/(2*P.permit_0+betaE*P.delT)
    
    #beeBee =0.696
    #alphaE = 2-(V.plasmaFreqE*P.delT**2)
    #gammGamm =P.permit_0*beeBee*V.plasmaFreqE**2*P.delT**2
    
   # print("types: ", type(matLoc), type(gammaE), type(plas), type(betaE), type(kapE), type(selfCo), type(HCurlCo))
    #print(m1JxNum/m1JxDen, "co ef 1", m2PxNum/m1JxDen, "second coef", m3ExNum/m1JxDen)
    coef= (1/(P.permit_0*V.plasmaFreqE*V.omega_0E*V.omega_0))
    #print(coef)
    for nz in range(int(P.materialFrontEdge-1), int(P.materialRearEdge)):
       V.Jx[nz] =coef*(V.omega_0E*V.omega_0E*V.tempVarPol[nz] + (V.gammaE/P.delT)*(V.polarisationCurr[nz]-V.tempVarPol[nz]) + (V.polarisationCurr[nz]-2*V.tempVarPol[nz] +V.tempTempVarPol[nz])/(P.delT*P.delT))#( kapE*V.Jx[nz] + betaE*(V.Ex[nz] + V.tempTempVarE[nz]))*C_V.den_Exdz[nz]
    return V.Jx
    # find paper

#@nj
def ADE_NonLin_Pol_Ex_Pbar(V, P): #Charles Varin: Explicit formulation of second and third order optical nonlinearity in the FDTD framework 
    #This function creates J linear,J_{inst Kerr}, J_{Raman}
    #Keep current polCurr method as it seems to work for lin lorentz. displacement D? keep same, but treat nonlinear with their J method, extra step Ex - Jnl
    if np.max(np.abs(V.Pbar3))> 1e-20:
        breakpoint() # COULD BE A PROBLEM WITH ARRAY ELEMENTS VS ARRAY AS A WHOLE MAYBE WITH FOR LOOP?
    for nz in range (int(P.materialFrontEdge), int(P.materialRearEdge)): 
        V.Pbar3[nz] = P.permit_0* (V.chi1Stat*V.Ex[nz] + V.chi3Stat*(V.alpha3* V.Ex[nz]*V.Ex[nz]*V.Ex[nz] + (1-V.alpha3)*V.Qx3[nz]*V.Ex[nz])) 
    
    return V.Pbar3


#@nj
def ADE_Lin_Curr_And_Pol_Varin(V, P):
    # Linear lorentz dispersion skeleton from Varin paper
    
    Gamma = (V.gammaE*P.delT)/2
    A = 1- Gamma
    D = 1+Gamma
    B = V.omega_0E*V.omega_0E*P.delT
    #if np.max(np.abs(V.Jx))> 0:
       # breakpoint()
    for nz in range (int(P.materialFrontEdge), int(P.materialRearEdge)): 
        V.Jx[nz] = (A/D)*V.Jx[nz] + (B/D)*(V.Pbar3[nz] -V.polarisationCurr[nz])
        V.polarisationCurr[nz] = V.polarisationCurr[nz] + P.delT *V.Jx[nz]
    
    
    return V.Jx, V.polarisationCurr

#@nj
def ADE_Nonlin_Q_and_G(V, P):
    
    Gamma = (V.nonLin3gammaE*P.delT)/2
    e = 1- Gamma
    f = 1+Gamma
    h = V.nonLin3Omega_0E*V.nonLin3Omega_0E*P.delT
   # if np.max(np.abs(V.Gx3))> 0:
     #   breakpoint()
    for nz in range (int(P.materialFrontEdge), int(P.materialRearEdge)): 
        V.Gx3[nz] = (e/f)*V.Gx3[nz] + (h/f)*(V.Ex[nz]*V.Ex[nz] - V.Qx3[nz])
        V.Qx3[nz] = V.Qx3[nz] + P.delT*V.Gx3[nz]
    
    return V.Gx3, V.Qx3, V.Ex
@nj
def ADE_PolarisationCurrent_Ex(V, P, C_V, C_P, counts):   #FIND ADE PAPER!
    """
    s0=(1/delt^2)+(gamae/(2*delt))
   s1=((2/delt^2)-woe^2)/s0
   s2=((gamae/(2*delt))-1/delt^2)/s0;
   s3=(eps0*wpe^2)/s0;
   foe=2*pi*0.1591e9; woe=2*pi*foe;fom=foe; wom=woe;
fpe=1.1027e9; wpe=2*pi*fpe;fpm=fpe;  wpm=wpe;
gamae=0; gamam=gamae;
    """
    D= (1/P.delT**2)+(V.gammaE/(2*P.delT))#1+V.gammaE*P.delT*V.omega_0E
    #print("D ", D)
    A = ((2/P.delT**2)-V.omega_0E**2)/D#(2-V.omega_0E*V.omega_0E*P.delT*P.delT)/D
    #print("A", A)
    B =((V.gammaE/(2*P.delT))-1/P.delT**2)/D#(V.gammaE*V.omega_0E*P.delT-1)/D
    #print(B)
    C = (P.permit_0*V.plasmaFreqE**2)/D
    
   
  
    for nz in range (int(P.materialFrontEdge), int(P.materialRearEdge)): # Does this need to be a for loop?
        V.polarisationCurr[nz] = A*V.polarisationCurr[nz]+ B*V.tempTempVarPol[nz] +C*V.Ex[nz]
  
    return V.polarisationCurr

# textbook for linear polarisation
#Elsherbeni, Atef Z. Demir, Veysel. (2016). Finite-Difference Time-Domain Method for Electromagnetics with MATLAB® Simulations (2nd Edition) - 13.1.3 Modeling Drude Medium Using ADE Technique. Institution of Engineering and Technology. Retrieved from
#https://app.knovel.com/hotlink/pdf/id:kt010WVJ54/finite-difference-time/modeling-drude-medium


@nj
def ADE_HyUpdate(V, P, C_V, C_P):
    
    for nz in range(1, P.Nz):
        V.Hy[nz] = V.Hy[nz]*V.UpHySelf[nz] + (V.Ex[nz+1]-V.Ex[nz])*V.UpHyMat[nz]*C_V.den_Hydz[nz] #- (P.delT/P.permea_0)*C_V.psi_Hy[nz] 
        #V.Hy[nz] = V.Hy[nz] + (V.Ex[nz+1]-V.Ex[nz])*(1/P.courantNo)#*C_V.den_Hydz[nz]
        #if(np.isnan(V.Hy[nz]) or V.Hy[nz] > 10):
         #    print("Hy IS wrong", V.Hy[nz], nz)
             
             #sys.exit()#*C_V.den_Hydz[nz]
    """
    if np.sum(np.isnan(V.Hy))>0:
        breakpoint()
    if np.sum(np.isinf(V.Hy))>0:
        breakpoint()
    """
    return V.Hy
        
def ADE_MyUpdate():
    
    pass

#forlocalsAEX= { 'betaE': float64, 'kapE' : float64, 'counts':int32 }
@nj
def ADE_ExUpdate(V, P, C_V, C_P, counts):
    for nz in range(1, len(V.Ex)):
         
        V.Ex[nz]=V.Ex[nz] +(V.Hy[nz]-V.Hy[nz-1]-V.Jx[nz])*V.UpExMat[nz]*C_V.den_Exdz[nz]#-(1/(epsInf))*(V.polarisationCurr[nz]-V.tempTempVarPol[nz])#*C_V.den_Exdz[nz] #-(P.delT/P.permit_0)*C_V.psi_Ex[nz]

    return V.Ex




def NonlinAnalytic(V, P, C_V, C_P):#
    #manley rowe 
    #phase match condition to determine dominant harmonic?
    
    pass

"""

Tomorrows work: Fill in nonlinear deets and measure harmonics just to see if they exist 
and are differentiable from noise


measure harmonics, relative amplitudes of harmonics variation along z of amplitude 
to find dAi/dz  Two adjacent grids? Integrate?
Create new var V.nonLin2PolCurr


Expand code to have nonlinearity and lorentz in magnetic field. Bring over info from 
COMSOL/Matlab stuff, run through code and might be usable in report.

Solitons? 
 
 
Some kind of validation to show periodic metamaterial acts like continuous material?
papers? Effective params. 

Expand fields to 2D, incorporate stl module, build set up in comsol and export stl

Incorporate basic evolving charge distribution

Some verifications for 2D set up? Particle dist? ensemble system?

Consider data science analysis of numerical data? GPU ENHANCEMENTS? 

Be careful about time steps and spatial steps of updates ? Go over and check as writing
up equations in thesis 
"""

@nj
def ADE_ExCreate(V, P, C_V, C_P):
    
    #Dxn = (V.Dx-V.tempTempVarDx)/2# ??
    for nz in range(P.materialFrontEdge, P.materialRearEdge):
        
       V.Ex[nz] =(V.Dx[nz] - V.polarisationCurr[nz] )/(P.permit_0) 
     #  if(np.isnan(V.Ex[nz]) or V.Ex[nz] > 10):
      #       print("Ex IS wrong create", V.Ex[nz])
             #sys.exit()
    #if(np.max(np.abs(V.Ex[P.materialFrontEdge: P.materialRearEdge])))>2:
     #   breakpoint()
     
    return V.Ex

@nj
def ADE_ExNonlin3Create(V, P, C_V, C_P, counts):
    
    #Dxn = (V.Dx-V.tempTempVarDx)/2# ??
    ######NOTE
    
    ###### NOTE
    
    
    ######NOTE
    #print("epsilon 0 multiplies epsilon! Include later")
    for nz in range(P.materialFrontEdge, P.materialRearEdge):
       
       
       V.Ex[nz] = V.Dx[nz] /(P.permit_0*1+P.permit_0*V.alpha3*V.chi3Stat*np.abs(V.Ex[nz])*np.abs(V.Ex[nz]))
     #  if(np.isnan(V.Ex[nz]) or V.Ex[nz] > 10):
      #       print("Ex IS wrong create", V.Ex[nz])
             #sys.exit()
    #if(np.max(np.abs(V.Ex[P.materialFrontEdge: P.materialRearEdge])))>2:
     #   breakpoint()
     
    return V.Ex

@nj
def ADE_DxUpdate(V, P, C_V, C_P):
    for nz in range(P.materialFrontEdge, P.materialRearEdge):
        V.Dx[nz] =V.Dx[nz] + (V.Hy[nz]-V.Hy[nz-1])*(P.delT/(P.dz))*C_V.den_Exdz[nz]
        #print(V.Dx[nz])
       # V.Ex[nz] =(V.Dx[nz] - V.polarisationCurr[nz])/P.permit_0
        #if abs(V.Dx[nz]) >1e-4:
            #print("Dx", V.Dx[nz], nz)
    #if(np.max(np.abs(V.Dx))) >0:
     #breakpoint()
    return V.Dx

@nj
def KerrNonlin(V,P, counts):
   # if np.max(V.JxKerr >100):
    #    breakpoint()
    V.JxKerr = ((V.alpha3*P.permit_0*V.chi3Stat)/P.delT)*(np.abs(V.Ex)**2*V.Ex - np.abs(V.tempTempVarE)**2*V.tempTempVarE)
    return V.JxKerr

def MUR1DEx(V,P, C_V, C_P):
    # convert into for loop for forward and backwards waves
    murMult = (P.c0*P.delT-P.dz)/(P.c0*P.delT+P.dz)
    #nz=1
    
    #nz = 2
    #V.Ex[nz] = V.tempTempVarE[nz+1] + murMult*(V.Ex[nz+1] - V.tempTempVarE[nz])
    #backwards for loop
    for nz in range(1, 5):
        V.Ex[nz] = V.tempTempVarE[nz+1] + murMult*(V.Ex[nz+1] - V.tempTempVarE[nz])
    #if V.Ex[nz] !=0:
      #  print("Ex is nonzero: ", V.Ex[nz])
    for nz in range(P.Nz-1, P.Nz-6, -1):
         V.Ex[nz] = V.tempTempVarE[nz-1] + murMult*(V.Ex[nz-1] - V.tempTempVarE[nz])
    #nz = P.Nz-1
   
    #if V.Ex[nz] !=0:
     #   print("Ex is nonzero: ", V.Ex[nz])
    
    return V.Ex

#if below works clean out redundant code

#Andrey Sukhorukov, Comparative study of FDTD adopted...
locs = {"cub": float64, "qua":float64, "one": float64, "nz": int32}
@nj(locals=locs)
def Nonlin_Eqn_Setup(V, P, cub, qua, one, nz):
    V.cubPoly[nz] = np.array([cub, qua, one, -np.abs(V.Dx[nz]/P.permit_0)**2])
    #print(out.dtype)
    #for i in range(len(out)):
       # V.cubPoly[nz][i] =
    return V.cubPoly[nz]

#@nj
def AcubicFinder(V,P):
    epsNum = (V.plasmaFreqE*V.plasmaFreqE)
    epsDom = (V.omega_0E*V.omega_0E-(2*np.pi*P.freq_in*2*np.pi*P.freq_in) + 1j*V.gammaE*2*np.pi*P.freq_in)
    eps0 = P.permit_0   
    epsilon = 1 + epsNum/epsDom
    cub = (V.alpha3*V.chi3Stat)**2
    qua = 2*np.real(V.alpha3*epsilon*V.chi3Stat)
    one = np.abs(epsilon)**2
    tim1 = time.perf_counter()
    for nz in prange(P.materialFrontEdge, P.materialRearEdge):
        V.cubPoly[nz] = Nonlin_Eqn_Setup(V, P, cub, qua, one, nz)

    #outp = Nonlin_Cubic_Solver(V, P)
    #breakpoint()

    V.Acubic[P.materialFrontEdge:P.materialRearEdge] = np.real(Nonlin_Cubic_Solver(V,P)[P.materialFrontEdge:P.materialRearEdge])

    return V.Acubic
        
# run repeatedly from E update each time A is desired.
#@nj(locals={"out": complex128[:]})
def Nonlin_Cubic_Solver(V, P):
    out = np.zeros(P.Nz)
    for jj in range(P.materialFrontEdge, P.materialRearEdge):
        #print("before roots")
        a = V.cubPoly[jj][0]
        b = V.cubPoly[jj][1]
        c = V.cubPoly[jj][2]
        d = V.cubPoly[jj][3]


        if abs(d)> 1e-8:
            #print("shouldn't be here yet")
            CS= CubicEquationSolver.CubicSolver(a,b,c,d)
            V.roots = CubicEquationSolver.solve(CS)
            out[jj] = V.roots[0]
    """
            for i in prange(len(V.roots)):
                if abs(np.real(V.roots[i])) >=0:
                    comp = np.imag(V.roots[i])
                    if abs(comp) <1e-40:
                        out[jj]= np.real(V.roots[i])
                    else:
                        out[jj]= 0
                else:
                    out[jj]= 0   # Returns V.Acubic[nz]
        else:
            out[jj] =0
    """

    return out

#forLocs ={'check' : bool}
#@nj(locals = forLocs, nogil=True)
#@jit(locals ={"check":  np.bool_})
@nj
def NonLinExUpdate(V,P):
    #epsilon =
    #epsNum = (V.plasmaFreqE*V.plasmaFreqE)
    #epsDom = (V.omega_0E*V.omega_0E-(2*np.pi*P.freq_in*2*np.pi*P.freq_in) + 1j*V.gammaE*2*np.pi*P.freq_in)
    eps0 = P.permit_0   
    epsilon = np.sqrt(1.2)#1 + epsNum/epsDom
    #matching: A simple and rigorous verification technique for nonlinearFDTD algorithms by optical parametric four-wave mixing
    prev = V.Ex
   # check: np.bool_ = np.max(abs(V.Ex[P.materialRearEdge]))>0
    for nz in range(P.materialFrontEdge, P.materialRearEdge):
        V.Ex[nz] = V.Dx[nz]/(eps0*np.real(epsilon) +eps0*V.chi3Stat*V.Acubic[nz])
    aft = V.Ex
    if abs(aft.sum()-prev.sum()) >0:
        print("After and previous are different")
    #if P.testMode:
     #   dictRep = {}
      #  dictRep = tfv(prev, aft)

    return V.Ex




def AnalyticalReflectionE(V, P):
    pi = np.pi
    
    #CHOSEN GAMMA HAS 1 IN FRONT NOT 2!!
    epsNum = (V.plasmaFreqE*V.plasmaFreqE)
    epsDom = (V.omega_0E*V.omega_0E-(2*pi*P.freq_in*2*pi*P.freq_in) + 1j*V.gammaE*2*pi*P.freq_in)
    eps0 = P.permit_0   
    epsilon = 1 + epsNum/epsDom
    #V.epsilon = epsilon
    # INTERESTING BUG WITH ARCSIN INVALID VALUE, UNIT TEST 
    mu =1
    refr = 1 # factored out 377 ohms
    refr2 = np.real(np.sqrt(epsilon))
    trans = abs((2*refr2)/(refr+refr2))
    #print(abs(trans), "transmission")
    reflection = abs((refr2-refr)/(refr+refr2))
    #print(abs(reflection), "reflection analytical")
    #print(abs(trans)+abs(reflection), "sum of trans and reflection")
    realV = P.c0/(np.sqrt(abs(np.real(epsilon))))
    n2 =  realV/P.c0
    n1 = 1
    reflectionNormInc = 1-abs((n1-n2)/(n1+n2))**2
    print(reflection, "NEW ANALYTICAL REFLECTION USING FRESNEL NORMAL INCIDENCE.")
    dispPhaseVelNum = (((2*np.pi*P.freq_in)*P.dz)/2)
    dispPhaseVelDenArg= ((P.dz)/(realV*P.delT))*np.sin((2*np.pi*P.freq_in*P.delT)/2)
    dispPhaseVel = dispPhaseVelNum/np.arcsin(dispPhaseVelDenArg)
   # print(trans1)
   # print(reflection1, " analytical reflection")
    print(epsilon ,"eps for pf: ", V.plasmaFreqE)
    print("Refractive index: ", refr2)
    alphaAtten =( 2*(np.imag(refr2))*P.freq_in*2*np.pi)/P.c0
    print(alphaAtten) #file:///C:/Users/Fraser/Downloads/MIT6_007S11_lorentz.pdf
    stability = np.real(refr2/(np.sqrt(1/P.dz*P.dz)))
    #plot stability and delT on same plot, to make sure stability does not exceed delT
    
    #print(epsNum)
    #print(epsDom)
    #print(trans1+reflection1)
    
   
    
    return reflection




def CPML(V,P, C_V, C_P):
    
    pass



def SpatialFiltering():
    
    pass

def SymbolicRegression():
    pass

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
            if P.Nz > 250:
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
    return V.Ex, V.Ex_History, V.Hy, V.Hy_History


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
    UpHyMat = np.zeros(P.Nz) #THIS IS INITIALISER DON'T PASS THROUGH FOR LOOP
    UpExMat = np.zeros(P.Nz)
    
    UpHyBackground = (1/P.CharImp)*P.courantNo
    UpExBackground = P.CharImp*P.courantNo


    for jl in range(0, P.Nz):  
        UpExMat[jl] =UpExBackground
        UpHyMat[jl] =UpHyBackground
 
    return UpHyMat, UpExMat



def Material(V,P):
    for kj in range(P.Nz):
            if(kj < P.materialFrontEdge):
                V.epsilon[kj] = 1
                V.mu[kj] = 1
                V.UpExHcompsCo[kj] = 1
                V.UpExSelf[kj] =1
            if(kj >= P.materialFrontEdge and kj < P.materialRearEdge):
                V.epsilon[kj] = P.epsRe
                V.mu[kj] = P.muRe
                V.UpExHcompsCo[kj] = P.eHcompsCo
                V.UpExSelf[kj] = P.eSelfCo
            if(kj>= P.materialRearEdge):
                V.epsilon[kj] = 1
                V.mu[kj] = 1 
                V.UpExHcompsCo[kj] = 1
                V.UpExSelf[kj] =1
    return V.epsilon, V.mu, V.UpExHcompsCo, V.UpExSelf

    


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
    for nz in range(0, P.Nz-1):
        V.Hy[nz] = V.Hy[nz] + (V.Ex[nz+1]-V.Ex[nz])*V.UpHyMat[nz]
    return V.Hy[0:P.Nz-2]

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
    for nz in range(1, P.Nz-1):
        V.Ex[nz] = V.Ex[nz]*V.UpExSelf[nz] + (V.Hy[nz]-V.Hy[nz-1])*V.UpExMat[nz]*V.UpExHcompsCo[nz]#*UpExMat[nz]
    return V.Ex[1:P.Nz-2]    


def ExTfSfCorr(V,P, counts):
   # ExTfsf= ExTfsf + np.exp(-(counter +0.5 -(-0.5)-30)*(counter +0.5 -(-0.5)-30)/100)
    V.Ex[P.nzsrc] += V.Hys[counts]# *np.exp(-(counter +0.5 -(-0.5)-30)*(counter +0.5 -(-0.5)-30)/100)
    return V.Ex[P.nzsrc]



"""
Issue: feed fields back and forth. RETURN uphy etc etc , call
"""
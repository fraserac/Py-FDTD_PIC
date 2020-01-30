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
from Material_Def import *
from scipy import signal as sign
import sys 

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



def FieldInit(Size, timeJumps):
    while (True):
        try:
            if type(Size) != int:
                raise TypeError('Grid spaces must be a positive integer value')
                break
            if type(timeJumps) != int:
                raise TypeError('Number of timeSteps must be positive integer valued')
                break
            if timeJumps > 2**14:
                raise ValueError('timeSteps max too large')
                break
            if Size > 250:
                raise ValueError('Grid size too big')
                break
            if Size ==0:
                raise ValueError('Cannot have grid size 0!')
                break
            if timeJumps==0:
                raise ValueError('Cannot have zero timeSteps!')
                break
            if Size < 0:
                raise ValueError('Grid size cannot be negative')
                break
            if timeJumps < 0:
                raise ValueError('Cannot have negative timeSteps')
                break
            break
        except ValueError as e:
            print(e)
            sys.exit()
        
    Ex =np.zeros(Size)#,dtype=complex)
    Hy=np.zeros(Size)#,dtype=complex)
    Ex_History= [[]]*timeJumps
    Hy_History= [[]]*timeJumps
    Hys = []*timeJumps
    Ezs = []*timeJumps
    return Ex, Ex_History, Hy, Hy_History,Hys, Ezs
   
    
def SourceCalc(UpHyMat, UpExMat, Nz): # this function will run the FDTD over just the initial media and measure the points at x1 over t
    #material to find transmission and reflection vals
    UpHyMat = np.zeros(Nz) #THIS IS INITIALISER DON'T PASS THROUGH FOR LOOP
    UpExMat = np.zeros(Nz)
    
    UpHyBackground = (1/CharImp)*courantNo
    UpExBackground = CharImp*courantNo


    for jl in range(0, Nz):  
        UpExMat[jl] =UpExBackground
        UpHyMat[jl] =UpHyBackground
        
    x1Loc = 80 # be careful with thin regions
    return UpHyMat, UpExMat, x1Loc
  
def lossyMat():
    #Calculate change to co-efficients
    # eLoss = sigma_e dt, mLoss = sigma_m dt 
    #for now explicit values
    eLoss =0.001
    mLoss = 0
    eSelfCo = (1-eLoss)/(1+eLoss)#
    eHcompsCo = 1+eLoss
    return eSelfCo, EHcompsCo

#PASS TO MASTER CONTROLLER, THEN THROUGH TO UPDATECOEF, WHERE THE NEW CO-EFF WILL DIVIDE uPEX AND UPHY at mat, 

def UpdateCoef(UpHyMat, UpExMat, Nz):# POTENTIAL ISSUE, COURANT NO AND DOUBLE DEFINITION OF MU EPS.
    #CHECK COURANT NO.
    UpHyBackground = (1/CharImp)*courantNo
    UpExBackground = CharImp*courantNo
    print(UpExBackground)
    UpHyMat = np.zeros(Nz) #THIS IS INITIALISER DON'T PASS THROUGH FOR LOOP
    UpExMat = np.zeros(Nz)
    for k in range(0, MaterialFrontEdge-1):  
        UpExMat[k] =UpExBackground
        UpHyMat[k] =UpHyBackground
    for jj in range(MaterialFrontEdge-1, MaterialRearEdge-1):
        UpExMat[jj] = (UpExBackground/MatEps)
        UpHyMat[jj] = (UpHyBackground/MatMu)
    for ii in range(MaterialRearEdge-1, Nz):
        UpExMat[ii] = UpExBackground
        UpHyMat[ii] = UpHyBackground
   
    return UpHyMat, UpExMat  



def HyBC(Hy, size):
    Hy[size-1] = Hy[size-2]
    return Hy[size-1]
   
# FOR HY AND EX update/EZ? feed in eSelfCo and hSelfCo
def HyUpdate(Hy, Ex, UpHyMat, size):
    for nz in range(0, size-1):
        Hy[nz] = Hy[nz] + (Ex[nz+1]-Ex[nz])*UpHyMat[nz]
    return Hy[0:size-2]

def HyTfSfCorr(HyTfsf, counter, UpHyMatTfsf, Ezs):
     HyTfsf -= Ezs[counter]*UpHyMatTfsf#*np.exp(-(counter - 30)*(counter-30)/100)
     #link to sourceGen for harmonic or ricker or gaussian etc 
     #np.sin((2*np.pi)/Nlam*(courantNo))
     return HyTfsf
   
def ExBC(Ex, size):
    Ex[0] = Ex[1]
    Ex[size-1]=  Ex[size-2]
    return Ex[0], Ex[size-1]
   

def ExUpdate(Ex, UpExMat, Hy,  size):
    for nz in range(1, size-1):
        Ex[nz] = Ex[nz] + (Hy[nz]-Hy[nz-1])*UpExMat[nz]#*UpExMat[nz]
    return Ex[1:size-2]    


def ExTfSfCorr(ExTfsf, counter, nzsrc, UpExMatTfsf, Hys):
   # ExTfsf= ExTfsf + np.exp(-(counter +0.5 -(-0.5)-30)*(counter +0.5 -(-0.5)-30)/100)
    ExTfsf += Hys[counter]# *np.exp(-(counter +0.5 -(-0.5)-30)*(counter +0.5 -(-0.5)-30)/100)
    return ExTfsf



"""
Issue: feed fields back and forth. RETURN uphy etc etc , call
"""
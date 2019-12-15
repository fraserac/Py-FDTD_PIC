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


c0 = 1/(np.sqrt(permit_0*permea_0))   ##M MAKE THIS MORE ACCURATE LATER
freq_in = 5e9
maxFreq = 10e9
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

def sourceGen(T):
    print(T)
    pulse = np.sin(2*np.pi*freq_in*delT*2*T)
    #pulse = np.exp(-((t-t0)/tau)**2)
    return pulse


def FieldInit(Size, timeJumps):
    size = Size
    Ex =np.zeros(size)#,dtype=complex)
    Hy=np.zeros(size)#,dtype=complex)
    Ex_History= [[]]*timeJumps
    Hy_History= [[]]*timeJumps
   
    return Ex, Ex_History, Hy, Hy_History
   
def UpdateCoef(UpHyMat, UpExMat, Nz):#  Pass these back to master, pass in too 
    UpHyMat = np.zeros(Nz) #THIS IS INITIALISER DON'T PASS THROUGH FOR LOOP
    UpExMat = np.zeros(Nz)

    UpHyBackground = ((1/mu)/CharImp)*courantNo
    UpExBackground = ((1/epsilon)*CharImp)*courantNo

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
   

def HyUpdate(Hy, Ex, UpHyMat, size):
    for nz in range(0, size-1):
        Hy[nz] = Hy[nz] + UpHyMat[nz]*(Ex[nz+1]-Ex[nz])
    return Hy[0:size-2]

def HyTfSfCorr(HyTfsf, counter, UpHyMatTfsf):
     HyTfsf= HyTfsf - UpHyMatTfsf*np.exp(-(counter - 30)*(counter-30)/100)
     return HyTfsf
   
def ExBC(Ex, size):
    Ex[0] = Ex[1]
    Ex[size-1]=  Ex[size-2]
    return Ex[0], Ex[size-1]
   

def ExUpdate(Ex, UpExMat, Hy,  size):
    for nz in range(1, size-1):
        Ex[nz] = Ex[nz] + UpExMat[nz]*(Hy[nz]-Hy[nz-1])
    return Ex[1:size-2]    


def ExTfSfCorr(ExTfsf, counter):
    ExTfsf= ExTfsf + np.exp(-(counter +0.5 -(-0.5)-30)*(counter +0.5 -(-0.5)-30)/100)
    return ExTfsf





"""
Issue: feed fields back and forth. RETURN uphy etc etc , call
"""
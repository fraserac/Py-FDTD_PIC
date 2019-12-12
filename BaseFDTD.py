# -*- coding: utf-8 -*-
"""
This is currently a 1D work in progress FDTD 

This script is the engine that will calculate field updates and plotting functions

The plotting and animation functionality may eventually be moved to a separate script.

This script will eventually just contain the update equations and animation

Courant number is 1, C delT = delZ

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sci
import math 
from Material_Def import *
#from MasterController import *
"""
c0 = 1/(np.sqrt(permit_0*permea_0))   ##M MAKE THIS MORE ACCURATE LATER
freq_in = 5e9
maxFreq = 10e9
lamMin = c0/(np.sqrt(abs(epsRe)*abs(muRe))*maxFreq)
Nlam = 30
MaxGridSize = 1e5
dz =lamMin/Nlam    
delT = (dz)/(2*c0)
NzSize =200
Nz = NzSize

UpHyFree = (1/mu)/CharImp
UpExFree = (1/epsilon)*CharImp 
timeSteps =449#  (sim runs for timeStep*delT)
# redefine E as Sqrt(epsilon0/mu0), leads to update coefficients being 0.5

if timeSteps > 5000:
    print('time stepping too damn fine')
    stahp



nzsrc = 50#round(Nz/2)
"""
Ex =[]#,dtype=complex)
Hy=[]#,dtype=complex)
Ex_History= [[]]
Hy_History= [[]]  # feed in timesteps 

fig, ax = plt.subplots()
line, = ax.plot(Ex)
"""
UpHyMat = np.zeros(Nz)#,dtype=complex)
UpExMat = np.zeros(Nz)#,dtype=complex)
"""



#MAIN FDTD LOOP BASIC EDITION PRE-PRE-PRE-ALPHA
#for loop over grid up to basic b.c. for update eqns, iterate through Nz with nz


"""
for k in range(0, MaterialFrontEdge-1):   
    UpExMat[k] =UpExFree 
    UpHyMat[k] =UpHyFree
for jj in range(MaterialFrontEdge-1, MaterialRearEdge-1):
    UpExMat[jj] = (UpExFree/MatEps)
    UpHyMat[jj] = (UpHyFree/MatMu)
for ii in range(MaterialRearEdge-1, Nz):
    UpExMat[ii] = UpExFree
    UpHyMat[ii] = UpHyFree
"""
"""
for count in range(timeSteps):
    #print(count)
    Hy[Nz-1] = Hy[Nz-2]
    for nz in range(0, Nz-1):
        Hy[nz] = Hy[nz] + UpHyMat[nz]*(Ex[nz+1]-Ex[nz])
     # PERFECT REFLECTING BOUNDARY   
   #Ex[Nz-1] = Ex[Nz-1] + UpEx*(0-Hy[Nz-1])
   #Hy[1] = Hy[1] + UpHy*(Ex[1]-0)
    Hy[nzsrc-1]= Hy[nzsrc-1] - UpHyMat[nzsrc-1]*np.exp(-(count - 30)*(count-30)/100)  #tf/sf correction Hy
      #tf/sf
    
    
    Ex[0] = Ex[1]
    Ex[Nz-1]=  Ex[Nz-2]
    
    for nz in range(1, Nz-1):
        Ex[nz] = Ex[nz] + UpExMat[nz]*(Hy[nz]-Hy[nz-1])
    
     #make T0 and tau into variables
    Ex[nzsrc]= Ex[nzsrc] + np.exp(-(count +0.5 -(-0.5)-30)*(count +0.5 -(-0.5)-30)/100) #tf/sf correction Ex
    Ex_History[count] = np.insert(Ex_History[count], 0, Ex)
      
"""

def sourceGen(T):
    print(T)
    pulse = np.sin(2*np.pi*freq_in*delT*2*T)
    #pulse = np.exp(-((t-t0)/tau)**2)
    return pulse 


def FieldInit(size, timeJumps):
    Ex =np.zeros(size)#,dtype=complex)
    Hy=np.zeros(size)#,dtype=complex)
    Ex_History= [[]]*timeJumps
    Hy_History= [[]]*timeJumps



def HyBC(Size):
    Hy[Size-1] = Hy[Size-2]
    return Hy[Size-1]

def HyUpdate():
    pass

def HyTfSfCorr():
    pass

def ExBC():
    pass

def ExUpdate():
    pass

def ExTfSfCorr():
    pass

def PrepAnim():
    pass

def init():
 line.set_ydata([])
 return line,

def animate(i):
    line.set_ydata(Ex_History[i])  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=15, blit=True)


plt.axis([0, Nz, -2, 2])
ax.axvspan(MaterialFrontEdge, MaterialRearEdge, alpha=0.5, color='red')
plt.show()



# -*- coding: utf-8 -*-
"""
This is the start of my FDTD code. So far 1D.

This script will eventually just contain the update equations and animation


DOES NOT CURRENTLY WORK FOR EPSILON NEGATIVE OR MU NEGATIVE MATERIALS OR NONLINEAR MATERIALS or complex 
materials

Courant number is 1, C delT = delZ

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sci
import math 

c0 = 3e8   ##M MAKE THIS MORE ACCURATE LATER
freq_in = 5e9
refInd =1
maxFreq = 10e9
conduct = 0.5
permit_0 = 8.85e-12
permea_0 = 1.26e-6
epsRe = 1
epsIm = 1
epsilon = 1#complex(epsRe,epsIm)
muRe = 1
muIm = 1
mu = 1#complex(muRe, muIm)
lamMin = c0/(np.sqrt(abs(epsRe)*abs(muRe))*maxFreq)
Nlam = 30
MaxGridSize = 1e5
dz =lamMin/Nlam    
delT = (dz)/(2*c0)
NzSize =200
Nz = NzSize

MatEps = 3
MatMu = 1


CharImp =np.sqrt(permea_0)/np.sqrt(permit_0)#c0*delT/(dz)   ## En normalised to sqrt(epsion0/mu0)*E
UpHyFree = (1/mu)/CharImp
UpExFree = (1/epsilon)*CharImp 
timeSteps =449#  (sim runs for timeStep*delT)
# redefine E as Sqrt(epsilon0/mu0), leads to update coefficients being 0.5

if timeSteps > 5000:
    print('time stepping too damn fine')
    stahp


Ex =np.zeros(Nz)#,dtype=complex)
Hy=np.zeros(Nz)#,dtype=complex)
Ex_History= [[]]*timeSteps
nzsrc = 50#round(Nz/2)
Z = np.arange(0, Nz, dz)
fig, ax = plt.subplots()
line, = ax.plot(Ex)
Ex_low_m2 =0.
Ex_low_m1 =0.
Ex_high_m1 = 0.
Ex_high_m2 = 0.
MaterialFrontEdge = 100  # Discrete tile where material begins (array index) 
MaterialRearEdge = 120  # discrete tile where material ends 
UpHyMat = np.zeros(Nz)#,dtype=complex)
UpExMat = np.zeros(Nz)#,dtype=complex)

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
for hh in range(Nz-1):
    Ex[hh] =0
for hhh in range(Nz-2):
    Hy[hhh] =0
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


for count in range(timeSteps):
    #print(count)
    Hy[Nz-1] = Hy[Nz-2]
    for nz in range(0, Nz-1):
        Hy[nz] = Hy[nz] + UpHyMat[nz]*(Ex[nz+1]-Ex[nz])
     # PERFECT REFLECTING BOUNDARY   
   #Ex[Nz-1] = Ex[Nz-1] + UpEx*(0-Hy[Nz-1])
   #Hy[1] = Hy[1] + UpHy*(Ex[1]-0)
    Hy[49]= Hy[49] - UpHyMat[49]*np.exp(-(count - 30)*(count-30)/100)  #tf/sf correction Hy
      #tf/sf
    
    #Absorbing boundary condition
    """
    Ex[0] = Ex_low_m2
    Ex_low_m2 = Ex_low_m1
    Ex_low_m1 = Ex[1]
    
    Ex[Nz-1] = Ex_high_m2
    Ex_high_m2 = Ex_high_m1
    Ex_high_m1 = Ex[Nz-2]
    """
    Ex[0] = Ex[1]
    Ex[Nz-1]=  Ex[Nz-2]
    
    for nz in range(1, Nz-1):
        Ex[nz] = Ex[nz] + UpExMat[nz]*(Hy[nz]-Hy[nz-1])
    
     #make T0 and tau into variables
    Ex[50]= Ex[50] + np.exp(-(count +0.5 -(-0.5)-30)*(count +0.5 -(-0.5)-30)/100) #tf/sf correction Ex
    Ex_History[count] = np.insert(Ex_History[count], 0, Ex)
      


def animate(i):
    line.set_ydata(Ex_History[i])  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=15, blit=True)


plt.axis([0, Nz, -2, 2])
ax.axvspan(MaterialFrontEdge, MaterialRearEdge, alpha=0.5, color='red')
plt.show()



# -*- coding: utf-8 -*-
"""
This is the start of my FDTD code. So far 1D.

This script will eventually just contain the update equations and animation

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import scipy as sci
import math 

c0 = 3e8
refInd =1
maxFreq = 10e9
lamMin = c0/maxFreq
Nlam = 30
MaxGridSize = 1e5
dz =0.01#lamMin/Nlam
delT = (dz)/(2*c0)
NzSize =100
Nz = 50#np.arange(0, NzSize, dz)
permit_0 = 8.85e-12
permea_0 = 1.26e-6
epsilon_r = 5
UpHy =0.5 #c0*delT/dz#0.5#(c0*delT)/permea_0
UpEx =0.5#c0*delT/dz# 0.5#(c0*delT)/permit_0
timeSteps =500# math.ceil((12*tau_0+5*PropTime)/delT)
# redefine E as Sqrt(epsilon0/mu0), leads to update coefficients being 0.5

if timeSteps > 1000:
    print('time stepping too damn fine')
    stahp


Ex =np.zeros(Nz)
Hy=np.zeros(Nz)
Ex_History= [[]]*timeSteps
nzsrc = 10#round(Nz/2)
Z = np.arange(0, Nz, dz)
fig, ax = plt.subplots()
line, = ax.plot(Ex)
Ex_low_m2 =0.
Ex_low_m1 =0.
Ex_high_m1 = 0.
Ex_high_m2 = 0.
MaterialFrontEdge = 30  # Discrete tile where material begins (array index) 
MaterialRearEdge = 35  # discrete tile where material ends 

UpExMat = np.zeros(Nz)

def init():
 line.set_ydata([])
 return line,

for k in range(0, MaterialFrontEdge-1):
    UpExMat[k] = 0.5
for jj in range(MaterialFrontEdge-1, MaterialRearEdge-1):
    UpExMat[jj] = 0.5/epsilon_r   
for ii in range(MaterialRearEdge-1, Nz-1):
        UpExMat[ii] = 0.5

#MAIN FDTD LOOP BASIC EDITION PRE-PRE-PRE-ALPHA
#for loop over grid up to basic b.c. for update eqns, iterate through Nz with nz

def sourceGen(T):
    print(T)
    pulse = sin(2*pi*freq_in*delT*T/10)
    #pulse = np.exp(-((t-t0)/tau)**2)
    return pulse 

freq_in = 5e9

for count in range(timeSteps):
    print(count)
    for nz in range(1, Nz-1):
        Ex[nz] = Ex[nz] + UpExMat[nz]*(Hy[nz-1]-Hy[nz])
        
     # PERFECT REFLECTING BOUNDARY   
   #Ex[Nz-1] = Ex[Nz-1] + UpEx*(0-Hy[Nz-1])
   #Hy[1] = Hy[1] + UpHy*(Ex[1]-0)
    
    Ex[nzsrc]= Ex[nzsrc] + np.exp(-((count-40)/12)**2)
    
    #Absorbing boundary condition
    
    Ex[0] = Ex_low_m2
    Ex_low_m2 = Ex_low_m1
    Ex_low_m1 = Ex[1]
    
    Ex[Nz-1] = Ex_high_m2
    Ex_high_m2 = Ex_high_m1
    Ex_high_m1 = Ex[Nz-2]
    
    for nz in range(0, Nz-1):
        Hy[nz] = Hy[nz] + UpHy*(Ex[nz]-Ex[nz+1])
        
    Ex_History[count] = np.insert(Ex_History[count], 0, Ex)
      


def animate(i):
    line.set_ydata(Ex_History[i])  # update the data.
    return line,


ani = animation.FuncAnimation(
    fig, animate, init_func=init, interval=15, blit=True)


plt.axis([0, Nz, -2, 2])
ax.axvspan(MaterialFrontEdge, MaterialRearEdge, alpha=0.5, color='red')
plt.show()



# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 17:59:40 2020

@author: Owner

DISPERSIVE FDTD SANDBOX TO FIND ISSUES 

Complete lorentzian FDTD with sinusoidal source
procedural:
    
H field 

E field 
E source 

no pml

"""
import numpy as np
import matplotlib.pylab as plt

domain = 14000 #BIGGER DOMAIN FOR SMALLER FREQUENCY/LARGER WAVELENGTH)
tim = 5000
Ex = np.zeros(domain)
Hy = np.zeros(domain)
freq =0.08e9 # IS THIS IN RADIAN OR HZ?
lam = (3e8/freq)*10
nl = 700  # POINTS PER WAVELENGTH
dz = (3e8/(freq))*(1/nl)
dt = (dz/3e8)
cour = (3e8*dt)/dz
src = 2100
matFront = 8000
matRear = domain-1
fig, ax = plt.subplots()
ax.set_ylim(-2, 2)
ax.set_xlim(0, domain)
ax.axvspan(matFront, matRear, alpha=0.5, color='green')
tempE = np.zeros(domain)
tempEOld = np.zeros(domain)
Jx = np.zeros(domain)
polX = np.zeros(domain)
gammaE = 0
omega0 = 5e9
plasmaE = 2*np.pi*1e9
currCo = (2-gammaE*dt)/(2+gammaE*dt)
polCurr = (2*omega0**2*dt)/(2+gammaE*dt)
ExCo = (2*0.14*dt*plasmaE**2)/(2+gammaE*dt)
Dx = np.zeros(domain)
perm0 = 8.854e-12
betaE = (0.5*plasmaE**2*perm0*dt)/(1+0.5*gammaE*dt)
kapE = (1-0.5*gammaE*dt)/(1+0.5*gammaE*dt)

ppw =  3e8/(freq*dz)
Exs =[]
Hys = []
for timer in range(tim):
    if(timer*dt < 1/freq):
        Exs.append(float(np.sin(2.0*np.pi/ppw*(cour*timer))))
        Hys.append(float(np.sin(2.0*np.pi/ppw*(cour*(timer+1)))))
    elif(timer*dt >= 1/freq):  
        Exs.append(0)
        Hys.append(0)
for boo in range(tim):
    if(Hys[boo] ==0):
        Hys[boo-1] =0
    #if(Exs[boo] ==0):
     #   Exs[boo-1] =0
    #break
    
for i in range(0,tim):
    
    for nz in range(0, domain-1):
        Hy[nz] = Hy[nz] + (Ex[nz+1] -Ex[nz])*(1/cour)
    
    #Hy[src-1] -= Exs[i]
    
    for nz in range(matFront, matRear):
        Jx[nz] =( kapE*Jx[nz] + betaE*(Ex[nz] + tempEOld[nz]))*(1/cour)
    #for nz in range(matFront, matRear):
     #       polX[nz] = polX[nz] + dt*Jx[nz]*cour
    
    #for nz in range(0, domain):
     #   Dx[nz] = Dx[nz] + (Hy[nz] - Hy[nz-1])*cour
    
        
    
   
    
    for nz in range(0,domain):
        tempEOld[nz] = tempE[nz]
        tempE[nz] = Ex[nz]
        Ex[nz] =((2*perm0-betaE*dt)/(2*perm0+betaE*dt))*Ex[nz] + (Hy[nz] - Hy[nz-1] -0.5*(1+kapE)*Jx[nz])*((2*dt)/(2*perm0+betaE*dt))*(1/cour) 
        
    Ex[src] = Hys[i]
   
    
    #for nz in range(matFront, matRear):
     #    Ex[nz] =  (Dx[nz] - polX[nz])
    
    #for nz in range(matRear,domain):
      #   Ex[nz] =Ex[nz] + (Hy[nz] - Hy[nz-1])*cour  
    #if i == 50:
        #ax.plot(Ex)
    #if i == 200:
        #ax.plot(Ex)
    if i == tim-1:
        ax.plot(Ex)
    
        
        
        
        
        
 # -*- coding: utf-8 -*-
"""
sandbox 


TO DO LIST:
    
    
TRANSFER OVER TO BASEFDTD, RE-ARRANGE BASE FDTD SO THAT EPS AND MU ARE MATRICES, 
ADD A PML/BETTER ABSORBING BOUNDARY CAPABLE OF HANDLING NONLINEAR STUFF 
PEC/PMC FOR NONLINEAR? 
CORRECT SIZES VIA CEM LECTURES OF DIELECTRIC 
FFTw APPLY UPDATE CO-EFFICIENTS IN FREQ DOMAIN? RE-ARRANGE COMPLEX PARAMETERS TO LOSS TERMS IN UPDATE?
HOW TO GET NEGATIVE REFRACTIVE INDEX MATERIALS WORKING? ADE? NONLINEAR MATERIALS?
VALIDATION SCRIPT, AS MANY AS POSSIBLE
DISPERSION OF NONLINEAR COMPONENTS?


UNIT TESTING IMPLEMENTATION

GRID ANISOTROPY?

OPTIMISATION?

UPGRADE TO 2D 

INCORPORATE PARTICLE SOURCE (MONTE CARLO GUN)
PARTICLE DYNAMICS

LAGRANGIAN TYPE STUFF? 

END RESULT: GRAPHS BOTH VALIDATION AND EXPERIMENTAL OUTPUT, AROUND 50 GRAPHS. 

IF TIME PERMITS, TWO POSSIBLE ADDITIONS:
    
    APPLICATIONS OF TECHNIQUES TO REAL WORLD DESIGN SITUATIONS WITH EXAMPLES
    NUMERICAL CONSTRUCTION AND SOLVING OF DAE's DESCRIBING CIRCUIT LIKE BEHAVIOUR OF META
    ATOM TO ALLOW PREDICTIONS OF NON-LINEAR PROPERTIES (LINEAR PROPERTIES FOUND VIA MLC LOW VOLTAGE 
    APPROXIMATION IN MATLAB/COMSOL USING NRW METHOD).
    
    CHANGE EX TO EZ ON BASEFDTD 
    
    
    NEXT CHALLENGE: FIX DISPERSION IN MATERIAL.
    

"""
import numpy as np
from scipy import constants as sci
from scipy import signal as sign
import matplotlib.pylab as plt
import os
import sys
import shutil
import cv2
import natsort
from decimal import *
from math import *

timeSteps = 500
Nz = 200
eps0 = sci.epsilon_0
mu0 = sci.mu_0
c0=float(Decimal(1/(eps0*mu0)).sqrt())
freq_in = 1e9
waveLen =  c0/freq_in 
CharImp = float(Decimal(mu0/eps0).sqrt())
EpsRe = 9
MuRe = 1
numPerWave = 20*(float(Decimal(EpsRe*MuRe).sqrt()))
dz = (waveLen/numPerWave)
courantNo = 1
dt = courantNo*dz/c0

Ez =np.zeros(Nz)
Hy =np.zeros(Nz)
Ez_History=[np.zeros(Nz)]*timeSteps
probe=np.zeros(timeSteps)
probeLoc = 75
t = np.arange(0, timeSteps, 1)*dt
fig, ax = plt.subplots()
srcLoc =20
deltaT = (srcLoc*dz)/(2*c0)- dt/2
materialFrontEdge = 70 # Discrete tile where material begins (array index)
materialRearEdge = 120
period = float(Decimal(1/(freq_in)))
probeLoc2 = 160



Ezs = []*timeSteps
Hys = []*timeSteps
tau = 2*np.pi
omega = tau*freq_in
ppw = (c0/(dz*freq_in))

def verification(matEps, matMu, assertion):## dispersion checks
    kNumerical = 2/(dz)*(np.arcsin(float(Decimal((dz/(c0*dt))**2*np.sin(np.pi*freq_in*dt)*np.sin(np.pi*freq_in*dt)).sqrt())))
    freeSpaceVp = (2*np.pi*freq_in)/(kNumerical)
    freeSpaceVpinC0 = freeSpaceVp/c0
    betaNumMat = (tau/(ppw*dz))*(float(Decimal(matEps*matMu).sqrt()))
    matVpinC0 = (omega/betaNumMat)*(1/c0)
    actualVpinC0 = (1/(float(Decimal(matEps*matMu).sqrt())))
    discrep = actualVpinC0/matVpinC0
    impedMat = np.sqrt(matMu/matEps)
    denomChar = CharImp+CharImp*impedMat
    analyticalTran1 = (2*(CharImp*impedMat))/denomChar
    analyticalTran2 = (2*(CharImp))/denomChar
    newAmp1 = analyticalTran1*1
    newAmp2 = newAmp1*analyticalTran2
    if(assertion):
        return newAmp1
    print("Analytical Transmission from region 1-2, new amplitude: ", newAmp1)
    print("Analytical Transmission from region 2-3, new amplitude: ", newAmp2)
    print("Free space dispersion: " , Decimal(freeSpaceVpinC0))
    print("Material dispersion: " ,Decimal(discrep))
    print("matVpInC0: ",Decimal(matVpinC0)  )
    print("actualVPinC0 : ",Decimal(actualVpinC0))


winHann = sign.hann(timeSteps) 
winBlack = sign.blackman(timeSteps)
winHamm = sign.hamming(timeSteps)
winFlat = sign.flattop(timeSteps)

def smoothTurnOn():
    
    ppw = c0/(freq_in*dz)
    for timer in range(timeSteps):
        if(timer*dt < period):
            Ezs.append(float(Decimal(np.sin(2.0*np.pi/ppw*(courantNo*timer)))))
            Hys.append(float(Decimal(np.sin(2.0*np.pi/ppw*(courantNo*(timer+1))))))
        elif(timer*dt >= period):  
            Ezs.append(0)
            Hys.append(0)
            
        for boo in range(timeSteps):
            if(Hys[boo] ==0):
                Hys[boo-1] =0
                break    
    return Ezs, Hys   
# FIX TURN OFF JITTER
Ezs, Hys = smoothTurnOn()






verification(EpsRe,MuRe, false)




def material(matEps, matMu):
    for kj in range(Nz):
            if(kj < materialFrontEdge):
                Epsilon[kj] = 1
                Mu[kj] = 1
            if(kj >= materialFrontEdge and kj <= materialRearEdge):
                Epsilon[kj] = matEps
                Mu[kj] = matMu
            if(kj>= materialRearEdge+2):
                Epsilon[kj] = 1
                Mu[kj] = 1
                
    return Epsilon, Mu

def updates():
    Ez =np.zeros(Nz)#,dtype=complex)
    Hy=np.zeros(Nz)#,dtype=complex)
    Ez_History= [[]]*timeSteps
    Hy_History= [[]]*timeSteps
    for timer in range(timeSteps):
        for k in range(Nz-1):
            Hy[k] = Hy[k] + (Ez[k+1]-Ez[k])/CharImp/Mu[k]   #*courantno # H[NEW TIME] = H OLD AT SAME SPATIAL+ 
        Hy[srcLoc] -= Ezs[timer]/CharImp
        Ez[srcLoc+1] += Hys[timer]
        Ez[0] = Ez[1]
        Ez[Nz-1] = Ez[Nz-2]
        for j in range(1, Nz-1):
            Ez[j] = Ez[j] + (Hy[j]-Hy[j-1])*(CharImp)/Epsilon[j]
        Ez_History[timer] = np.insert(Ez_History[timer], 0, Ez)
    return Hy, Ez, Ez_History        
       
probe = np.zeros(timeSteps)
probe2 = np.zeros(timeSteps)
probeAf = np.zeros(timeSteps)
probeAf2 = np.zeros(timeSteps)
Epsilon = np.ones(Nz)
Mu = np.ones(Nz)
Epsilon, Mu = material(1,1)
Hy, Ez, Ez_History= updates()
for ii in range(timeSteps):
    probe[ii] = Ez_History[ii][probeLoc]
    probeAf[ii] = Ez_History[ii][probeLoc2]
    


Epsilon = np.ones(Nz)
Mu = np.ones(Nz)
Epsilon, Mu = material(EpsRe,MuRe)
Hy, Ez, Ez_History= updates()
for ij in range(timeSteps):
    probe2[ij] = Ez_History[ij][probeLoc]
    probeAf2[ij] = Ez_History[ij][probeLoc2]




interval = 5
my_path = os.getcwd() 
newDir = "Ez fields"

path = os.path.join(my_path, newDir)

try:     ##### BE VERY CAREFUL! THIS DELETES THE FOLDER AT THE PATH DESTINATION!!!!
    shutil.rmtree(path)
except OSError as e:
    print ("Tried to delete folder Error: no such directory Ezists")
    
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" %path)
else:
    print ("Successfully created the directory %s " %path)
    
    

#NEzt up we iterate through the time steps of Ez_History (an array of arrays containing all y data from each time step)
#and create a plot for each time step, including the dielectric material. 

#these plots are converted to png files and saved in the new folder in the working directory

for i in range(0, timeSteps, interval):
    print(str.format('{0:.2f}', (100/(timeSteps/(i+1)))),"% complete")
    ax.clear()
    ax.plot(Ez_History[i])    
    ax.set_ylim(-2, 2)
    ax.set_xlim(0, Nz)
    ax.axvspan(materialFrontEdge, materialRearEdge, alpha=0.5, color='green')
    plt.savefig(path + "/" + str(i) + ".png")



#NEzt we collect all the images in the new directory and sort them numerically, then use OpenCV to create a 24fps video


image_folder = path
video_name = 'Ez in 1D FDTD sandbox.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

images = natsort.natsorted(images)  # without this python does a weird alphabetical sort that doesn't work
    

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
framesPerSec = 8
video = cv2.VideoWriter(video_name, 0, framesPerSec, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    #print(image)

cv2.destroyAllWindows()
video.release()
plt.close()

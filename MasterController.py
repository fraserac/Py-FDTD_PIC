# -*- coding: utf-8 -*-
"""
Fields controller, this script is the master that guides all the processes,
calling the field updates and eventually particle updaters and field interpolater
as well as any memory saving methods like sparse matrices etc.
BC functions called as well
If GUI is incorporated it will act as a direct interface with this script.
"""

import numpy as np
import scipy as sci
import matplotlib.pylab as plt
#import matplotlib.animation as animation
import math
from BaseFDTD import FieldInit, HyBC, FieldInit, HyBC, HyUpdate, HyTfSfCorr, ExBC, ExUpdate, ExTfSfCorr, UpdateCoef, SourceCalc
import BaseFDTD as bsfdtd
from Material_Def import *
#from moviepy.editor import VideoClip
#from moviepy.video.io.bindings import mplfig_to_npimage
#import moviepy.editor as mv
import os
import sys
import shutil
import cv2
import natsort
from TransformHandler import FourierTrans



class Variables(object):
    def __init__(self, UpHyMat, UpExMat, Ex, Hy, Ex_History, Hy_History, Hys, Exs):
        self.UpHyMat = UpHyMat
        self.UpExMat = UpExMat
        self.Ex = Ex
        self.Hy = Hy
        self.Ex_History = Ex_History
        self.Hy_History = Hy_History
        self.Exs = Exs
        self.Hys = Hys
        
    def __str__(self):
        return 'Contains data that will change during sim'
    
    def __repr__(self):
        return (f'{self.__class__.__name__}', ": Contains field variables that change during sim")
       
        
    # methods to handle user input errors during instantiation.
    
class Params(object):
    def __init__(self, epsRe, muRe, f_in, lMin, nlm, dz, dt, crntNo, matRear, matFront, gridNo, timeSteps):
        self.permit_0 = 8.854187817620389e-12#
        self.permea_0 =1.2566370614359173e-06
        self.c0 = 299792458.0
        self.epsRe = epsRe
        self.muRe = muRe
        self.freq_in = f_in
        self.lamMin = lMin
        self.Nlam = Nlm
        self.dz = dz
        self.delT = dt
        self.courantNo= crntNo
        self.MaterialFrontEdge = matFront
        self.MaterialRearEdge = matRear
        self.Nz = gridNo
        self.timeSteps = timeSteps
    
    def __repr__(self):
        return (f'{self.__class__.__name__}(', "Real Epsilon:",f'{self.epsRe!r}, "Real Mu", 'f'{self.muRe!r})
    
    def __str__(self):
        return 'Class containing all values that remain constant throughout a sim' 

#method to handle user input errors.
        
#instantiate objects Params and Vars 
        

    
#convert mast cont to def and feed in instances, adjust code accordingly on baseFDTD and mast
#set up unit tests, get code functioning again. 
        
    
    
    

  # MATERIAL PARAMETERS WILL CHANGE WITH POWER CHANGE BUT FOR NOW KEEP CONSTANT


"""
#Next, loop the FDTD over the time domain range (in integer steps, specific time would be delT*timeStep include this on plots later?)

"""

#FUNCTION THAT LOADS IN MATERIAL DEF, CAN BE PASSED IN AS A FIRST CLASS FUNCTION, RETURNS ALL
#PARAMETERS.
def Controller(P, V):  #Needs dot syntax
    V.Ex, V.Ex_History, V.Hy, V.Hy_History, V.Hys, V.Ezs= FieldInit(Nz, timeSteps)
    P.UpHyMat, P.UpExMat, x1Loc = SourceCalc(UpHyMat, UpExMat, Nz)
    
    for counts in range(timeSteps):   ### for media one transmission
       Hy[Nz-1] = HyBC(Hy, Nz)
       Hy[0:Nz-2] = HyUpdate(Hy, Ex, UpHyMat, Nz)
       Hy[nzsrc-1] = HyTfSfCorr(Hy[nzsrc-1], counts, UpHyMat[nzsrc-1])
       Ex[0], Ex[Nz-1] = ExBC(Ex, Nz)
       Ex[1:Nz-2]= ExUpdate(Ex,UpExMat, Hy,  Nz)
       Ex[nzsrc] = ExTfSfCorr(Ex[nzsrc], counts, nzsrc, UpExMat[nzsrc], Hys)
       Ex_History[counts] = np.insert(Ex_History[counts], 0, Ex)
       x1ColBe[counts] = Ex_History[counts][x1Loc] ##  X1 SHOULD BE ONE POINT! SPECIFY WITH E HISTORY ADDITIONAL INDEX.
    
    
    Ex, Ex_History, Hy, Hy_History, Hys, Ezs= FieldInit(Nz, timeSteps)
    UpHyMat, UpExMat = UpdateCoef(UpHyMat, UpExMat, Nz)
    
    for count in range(timeSteps):   
       Hy[Nz-1] = HyBC(Hy, Nz)
       Hy[0:Nz-2] = HyUpdate(Hy, Ex, UpHyMat, Nz)
       Hy[nzsrc-1] = HyTfSfCorr(Hy[nzsrc-1], count, UpHyMat[nzsrc-1])
       Ex[0], Ex[Nz-1] = ExBC(Ex, Nz)
       Ex[1:Nz-2]= ExUpdate(Ex,UpExMat, Hy,  Nz)
       Ex[nzsrc] = ExTfSfCorr(Ex[nzsrc], count, nzsrc, UpExMat[nzsrc], Hys)  # DON'T FEED IN NZSRC ONCE COMPLETE
       Ex_History[count] = np.insert(Ex_History[count], 0, Ex)
       x1ColAf[count]= Ex_History[count][x1Loc]
       #Hy_History[count] = np.insert(Hy_History[count], 0, Hy)
       
       
    #FFT x1ColBe and x1ColAf? 
    
    transWithExp, sig1Freq, sig2Freq, sample_freq = FourierTrans(x1ColBe, x1ColAf, x1Loc, t, delT)
# should have constant val of transmission over all freq range of source, will need harmonic source?   


params = Params(epsRe, muRe, freq_in, lamMin, Nlam, dz, delT, courantNo, MaterialRearEdge, MaterialFrontEdge, Nz, timeSteps)    
variables = Variables(UpHyMat, UpExMat, Ex, Hy, Ex_History, Hy_History)
Controller(params, variables)
"""
#Now we prepare to make the video including I/O stuff like setting up a new directory in the current working directory and 

#deleting the old directory from previous run and overwriting.
"""


###############
##########################
############################### MAYBE MOVE STUFF BELOW TO A NEW SCRIPT?#
"""
fig, ax = plt.subplots()
interval = 10
my_path = os.getcwd() 
newDir = "Ex fields"

path = os.path.join(my_path, newDir)

try:     ##### BE VERY CAREFUL! THIS DELETES THE FOLDER AT THE PATH DESTINATION!!!!
    shutil.rmtree(path)
except OSError as e:
    print ("Tried to delete folder Error: no such directory exists")
    
    
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed" %path)
else:
    print ("Successfully created the directory %s " %path)
    
    
"""
#Next up we iterate through the time steps of Ex_History (an array of arrays containing all y data from each time step)
#and create a plot for each time step, including the dielectric material. 

#these plots are converted to png files and saved in the new folder in the working directory
"""
for i in range(0, timeSteps, interval):
    print(str.format('{0:.2f}', (100/(timeSteps/(i+1)))),"% complete")
    ax.clear()
    ax.plot(Ex_History[i])    
    ax.set_ylim(-2, 2)
    ax.set_xlim(0, Nz)
    ax.axvspan(MaterialFrontEdge, MaterialRearEdge, alpha=0.5, color='green')
    plt.savefig(path + "/" + str(i) + ".png")


"""
#Next we collect all the images in the new directory and sort them numerically, then use OpenCV to create a 24fps video
"""

image_folder = path
video_name = 'Ex in 1D FDTD.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

images = natsort.natsorted(images)  # without this python does a weird alphabetical sort that doesn't work
    

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 12, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    #print(image)

cv2.destroyAllWindows()
video.release()
plt.close()
"""
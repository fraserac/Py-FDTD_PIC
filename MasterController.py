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
from BaseFDTD import FieldInit, HyBC, FieldInit, HyBC, HyUpdate, HyTfSfCorr, ExBC, ExUpdate, ExTfSfCorr, UpdateCoef
import BaseFDTD as bsfdtd
from Material_Def import *
from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import moviepy.editor as mv
import os
import sys
import shutil
import cv2
import natsort


Nz = 100

timeSteps =200

Ex =[]#,dtype=complex)
Hy=[]#,dtype=complex)
Ex_History= [[]]
Hy_History=[[]]
nzsrc = 50#round(Nz/2)
fig, ax = plt.subplots()
#line, = ax.plot(Ex, color = 'b')

UpHyMat =[]
UpExMat =[]

if timeSteps > 200:
    print('time stepping too damn fine')
    stahp


"""
First call initialiser functions
"""
   
Ex, Ex_History, Hy, Hy_History= FieldInit(Nz, timeSteps)
UpHyMat, UpExMat = UpdateCoef(UpHyMat, UpExMat, Nz)  # MATERIAL PARAMETERS WILL CHANGE WITH POWER CHANGE BUT FOR NOW KEEP CONSTANT


"""
Next, loop the FDTD over the time domain range (in integer steps, specific time would be delT*timeStep include this on plots later?)

"""

for count in range(timeSteps):   
   Hy[Nz-1] = HyBC(Hy, Nz)
   Hy[0:Nz-2] = HyUpdate(Hy, Ex, UpHyMat, Nz)
   Hy[nzsrc-1] = HyTfSfCorr(Hy[nzsrc-1], count, UpHyMat[nzsrc-1])
   Ex[0], Ex[Nz-1] = ExBC(Ex, Nz)
   Ex[1:Nz-2]= ExUpdate(Ex,UpExMat, Hy,  Nz)
   Ex[nzsrc] = ExTfSfCorr(Ex[nzsrc], count)
   Ex_History[count] = np.insert(Ex_History[count], 0, Ex)
   #Hy_History[count] = np.insert(Hy_History[count], 0, Hy)

"""
Now we prepare to make the video including I/O stuff like setting up a new directory in the current working directory and 
deleting the old directory from previous run and overwriting.

"""
###############
##########################
############################### MAYBE MOVE STUFF BELOW TO A NEW SCRIPT?
interval = 2
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
Next up we iterate through the time steps of Ex_History (an array of arrays containing all y data from each time step)
and create a plot for each time step, including the dielectric material. 

these plots are converted to png files and saved in the new folder in the working directory
"""
for i in range(0, timeSteps, interval):
    print(i)
    ax.clear()
    ax.plot(Ex_History[i])    
    ax.set_ylim(-2, 2)
    ax.set_xlim(0, Nz)
    ax.axvspan(MaterialFrontEdge, MaterialRearEdge, alpha=0.5, color='green')
    plt.savefig(path + "/" + str(i) + ".png")


"""
Next we collect all the images in the new directory and sort them numerically, then use OpenCV to create a 24fps video
"""

image_folder = path
video_name = 'Ex in 1D FDTD.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

images = natsort.natsorted(images)  # without this python does a weird alphabetical sort that doesn't work
    

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 24, (width,height))
for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))
    #print(image)

cv2.destroyAllWindows()
video.release()
plt.close()
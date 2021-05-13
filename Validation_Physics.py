
"""
Validation_Physics

This script will monitor the progress of the simulation and compare results to check the physics is working
appropriately
"""
import os
import sys
import shutil
import cv2
import natsort
import numpy as np
import scipy as sci
from tqdm import tqdm
import matplotlib.pylab as plt



def VideoMaker(P,V):
    fig, ax = plt.subplots()
    interval =100
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
    for i in tqdm(range(0, len(V.Ex_History))):
      ##breakpoint()
        ax.clear()
        ax.set_xlim(0, P.Nz)
        ax.set_ylim(-2,2)
        if P.FreeSpace == False:
            ax.axvspan(P.materialFrontEdge, P.materialRearEdge , alpha=0.5, color='green')
        if P.CPMLXm == True:
            ax.axvspan(0, P.pmlWidth-1 , alpha=0.2, color='blue')
        if P.CPMLXp == True:
            ax.axvspan(P.Nz-1-P.pmlWidth, P.Nz-1 , alpha=0.2, color='blue')
        ax.axvspan(P.x1Loc, P.x1Loc+2 , alpha=1, color='black')
        ax.axvspan(P.x2Loc, P.x2Loc+2 , alpha=1, color='red')
        
        ax.plot(np.real(V.Ex_History[i]))    
        stringTit = "Ex @ timeStep", str(i*P.vidInterval)
        plt.title(stringTit)
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
    framesPerSec =5
    video = cv2.VideoWriter(video_name, 0, framesPerSec, (width,height))
    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))
        #print(image)
    
    cv2.destroyAllWindows()
    video.release()
    plt.close()
    
# GRAPHS AND OTHER THINGS HERE. 
"""
def verification(P, V, assertion):## dispersion checks
    kNumerical = 2/(P.dz)*np.sqrt((np.arcsin(dz/(c0*dt))**2*np.sin(np.pi*freq_in*dt)*np.sin(np.pi*freq_in*dt)))
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
"""
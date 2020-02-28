# -*- coding: utf-8 -*-
"""
This script will contain all the validation tests and plots and will be called 
at end of MasterController NOT UNIT TEST, PHYSICS INTEGRATIVE TESTS.
"""

import numpy as np
import os
import sys
import shutil
import cv2
import natsort


def video():   ## FEED PARAMS AND VARS IN 
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

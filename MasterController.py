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
from BaseFDTD import FieldInit, Material, SmoothTurnOn, HyBC, FieldInit, HyBC, HyUpdate, HyTfSfCorr, ExBC, ExUpdate, ExTfSfCorr, UpdateCoef, EmptySpaceCalc
import BaseFDTD as bsfdtd
from Material_Def import *
from Validation_Physics import VideoMaker 
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
    def __init__(self, UpHyMat, UpExMat, Ex, Hy, Ex_History, Hy_History, Hys, Exs, x1ColBe, x1ColAf, epsilon, mu, UpExHcompsCo, UpExSelf):
        self.UpHyMat = UpHyMat
        self.UpExMat = UpExMat
        self.UpExHcompsCo = UpExHcompsCo
        self.UpExSelf = UpExSelf
        self.Ex = Ex
        self.Hy = Hy
        self.Ex_History = Ex_History
        self.Hy_History = Hy_History
        self.Exs = Exs
        self.Hys = Hys
        self.x1ColBe = x1ColBe
        self.x1ColAf = x1ColAf
        self.epsilon = epsilon
        self.mu = mu
        
    def __str__(self):
        return 'Contains data that will change during sim'
    
    def __repr__(self):
        return (f'{self.__class__.__name__}', ": Contains field variables that change during sim")
       
        
    # methods to handle user input errors during instantiation.
    
class Params(object):
    permit_0 =8.854187817620389e-12
    permea_0 =1.2566370614359173e-06
    CharImp =np.sqrt(permea_0/permit_0)
    c0 = 299792458.0
    
    def __init__(self, epsRe, muRe, f_in, lMin, nlm, dz, delT, courantNo, matRear, matFront, gridNo, timeSteps, x1Loc, nzsrc, period, eLoss, eSelfCo, eHcompsCo):
        self.epsRe = epsRe
        self.muRe = muRe
        self.freq_in = f_in
        self.lamMin = lMin
        self.Nlam = nlm
        self.dz = dz
        self.delT = delT
        self.courantNo= courantNo
        self.materialFrontEdge = matFront
        self.materialRearEdge = matRear
        self.Nz = gridNo
        self.timeSteps = timeSteps
        self.x1Loc = x1Loc
        self.nzsrc = nzsrc
        self.period = period
        self.eLoss = eLoss
        self.eSelfCo = eSelfCo
        self.eHcompsCo = eHcompsCo
    
    def __repr__(self):
        return (f'{self.__class__.__name__}'(f'{self.epsRe!r}, {self.muRe!r}'))
    
    def __str__(self):
        return 'Class containing all values that remain constant throughout a sim' 

"""
#Next, loop the FDTD over the time domain range (in integer steps, specific time would be delT*timeStep include this on plots later?)

"""

#FUNCTION THAT LOADS IN MATERIAL DEF, CAN BE PASSED IN AS A FIRST CLASS FUNCTION, RETURNS ALL
#PARAMETERS.
def Controller(P, V):  #Needs dot syntax
    V.Ex, V.Ex_History, V.Hy, V.Hy_History = FieldInit(V,P)
    V.Exs, V.Hys = SmoothTurnOn(V,P)
    
    V.UpHyMat, V.UpExMat = EmptySpaceCalc(V,P)   #RENAME EMPTY SPACE CALC
    
    for counts in range(P.timeSteps):   ### for media one transmission
       #V.Hy[P.Nz-1] = HyBC(V,P)
       V.Hy[0:P.Nz-2] = HyUpdate(V,P)
       V.Hy[P.nzsrc-1] = HyTfSfCorr(V,P, counts)
       V.Ex[P.nzsrc] = ExTfSfCorr(V,P, counts)
       V.Ex[0], V.Ex[P.Nz-1] = ExBC(V,P)
       V.Ex[1:P.Nz-2]= ExUpdate(V,P) 
       V.Ex_History[counts] = np.insert(V.Ex_History[counts], 0, V.Ex)
       V.x1ColBe[counts] = V.Ex_History[counts][P.x1Loc] ##  X1 SHOULD BE ONE POINT! SPECIFY WITH E HISTORY ADDITIONAL INDEX.
    
    V.epsilon, V.mu, V.UpExHcompsCo, V.UpExSelf  = Material(V,P)
    V.Ex, V.Ex_History, V.Hy, V.Hy_History= FieldInit(V,P)
    V.Exs, V.Hys = SmoothTurnOn(V,P)
    V.UpHyMat, V.UpExMat = UpdateCoef(V,P)
    
    for count in range(P.timeSteps):   ### for media one transmission
       #V.Hy[P.Nz-1] = HyBC(V,P)
       V.Hy[0:P.Nz-2] = HyUpdate(V,P)
       V.Hy[P.nzsrc-1] = HyTfSfCorr(V,P, count)
       V.Ex[P.nzsrc] = ExTfSfCorr(V,P, count)
       V.Ex[0], V.Ex[P.Nz-1] = ExBC(V,P)
       V.Ex[1:P.Nz-2]= ExUpdate(V,P)
       V.Ex_History[count] = np.insert(V.Ex_History[counts], 0, V.Ex)
       V.x1ColAf[count]= V.Ex_History[count][P.x1Loc]
       #Hy_History[count] = np.insert(Hy_History[count], 0, Hy)
       
       
    #FFT x1ColBe and x1ColAf? 
    
   # transWithExp, sig1Freq, sig2Freq, sample_freq = FourierTrans(x1ColBe, x1ColAf, x1Loc, t, delT)
# should have constant val of transmission over all freq range of source, will need harmonic source?   

    return P, V

P = Params(epsRe, muRe, freq_in, lamMin, Nlam, dz, delT, courantNo, MaterialRearEdge, MaterialFrontEdge, Nz, timeSteps, x1Loc, nzsrc, period, eLoss, eSelfCo, eHcompsCo)    
V = Variables(UpHyMat, UpExMat, Ex, Hy, Ex_History, Hy_History, Hys, Exs, x1ColBe, x1ColAf, epsilon, mu, UpExHcompsCo, UpExSelf)
P, V = Controller(P, V)

#Now we prepare to make the video including I/O stuff like setting up a new directory in the current working directory and 

#deleting the old directory from previous run and overwriting.

VideoMaker(P, V)


####variable exposes

UpHyMat = V.UpHyMat
UpExMat = V.UpExMat
Ex = V.Ex
Hy = V.Hy
Ex_History = V.Ex_History
Hy_History = V.Hy_History
Exs = V.Exs
Hys = V.Hys
x1ColBe = V.x1ColBe
x1ColAf = V.x1ColAf
epsilon = V.epsilon
mu = V.mu


###parameters

permit_0 =P.permit_0
permea_0 = P.permea_0
CharImp =P.CharImp
c0 = P.c0
epsRe = P.epsRe
muRe =  P.muRe
freq_in =  P.freq_in
lamMin = P.lamMin
Nlam = P.Nlam
dz = P.dz
delT =  P.delT
courantNo=  P.courantNo
materialFrontEdge = P.materialFrontEdge
materialRearEdge = P.materialRearEdge
Nz = P.Nz
timeSteps = P.timeSteps
x1Loc = P.x1Loc
nzsrc = P.nzsrc
period = P.period

######







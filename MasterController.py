3 # -*- coding: utf-8 -*-
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
#from BaseFDTD import FieldInit, Material, SmoothTurnOn, HyBC, FieldInit, HyBC, HyUpdate, HyTfSfCorr, ExBC, ExUpdate, ExTfSfCorr, UpdateCoef, EmptySpaceCalc, 
import BaseFDTD
import Material_Def as matDef
from Validation_Physics import VideoMaker 
import pdb 
import os 
import sys
import shutil
import cv2
import natsort
from TransformHandler import FourierTrans, ReflectionCalc
import winsound
import pyttsx3
import timeit
import numba
import cython 
import numba as num
import time as tim
 

duration = 1000  # milliseconds
freq = 440  # Hz


class Variables(object):
    UpHyMat = []
    UpExMat = []
    UpExHcompsCo = []
    UpExSelf = []
    UpHySelf = []
    UpHyEcompsCo = [] 
    Ex = []
    Hy = []
    Dx = []
   # self.By = By
    Ex_History = []
    Hy_History = []
    Psi_Ex_History= []
    Psi_Hy_History = []
    Exs = []
    Hys = []
    x1ColBe = []
    x1ColAf = []
    epsilon = []
    mu = []
    inputVec =[]
    outputPlots = []
    polarisationCurr = []
    plasmaFreqE = 2*np.pi*1e9
    gammaE = 1.2e8
    omega_0E = 5.15e9
    tempVarPol =[]
    tempTempVarPol =[]
    tempVarE =[]
    tempTempVarE =[]
    def __str__(self):
        return 'Contains data that will change during sim'
    
    def __repr__(self):
        return (f'{self.__class__.__name__}', ": Contains field variables that change during sim")
       
        
    # methods to handle user input errors during instantiation.
    
class Params(object):
    permit_0 = 0.0
    permea_0 = 0.0
    CharImp =376.730313668
    c0 = 299792458.0
    freq_in = 0.0
    epsRe =0.0
    epsIm = 0.0
    muRe = 0.0
    muIm = 0.0
    lamMin = 0.0
    Nlam = 0.0
    dz = 0.0
    delT = 0.0
    courantNo= 0.0
    materialFrontEdge = 0
    materialRearEdge = 0
    Nz = 0
    timeSteps = 0
    x1Loc =0
    x2Loc = 0
    nzsrc = 0
    period = 0.0
    eLoss = 0.0
    eSelfCo =0.0
    eHcompsCo = 0.0
    mLoss =0.0
    hSelfCo = 0.0
    hEcompsCo = 0.0
    pmlWidth = 0
    domainSize = 0
        
    def __repr__(self):
        return (f'{self.__class__.__name__}'(f'{self.epsRe!r}, {self.muRe!r}'))
    
    def __str__(self):
        return 'Class containing all values that remain constant throughout a sim' 


class CPML_Params(object):
    kappaMax =12 # 'Stretching co-ordinate of pml, to minimise numerical dispersion set it as 1' : DOI: 10.22190/FUACR1703229G see conclusion
    r_scale = 5.4# Within ideal bounds see Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17 (scaling power is called 'm' )
    r_a_scale=1
    sigmaEMax=0.0#1.1*sigmaOpt # Within ideal bounds for value, : Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
    sigmaHMax =sigmaEMax#1.1*sigmaOpt # See International Journal of Computer Science and Network Security, VOL.18 No.12, December 2018, page 4 right hand side.
    sigmaOpt  =sigmaEMax
    #Optimal value of pml conductivity at far end of pml: DOI: 10.22190/FUACR1703229G see equation 13
    
    
    alphaMax= 0.05# with bounds of ideal cpml alpha max, complex frequency shift parameter, Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
    
    def __repr__(self):
        return (f'{self.__class__.__name__}')
    
    def __str__(self):
        return 'Class containing all CPML values that remain constant throughout a sim' 
    
    
class CPML_Variables(object):
    kappa_Ex =[]
    kappa_Hy = []
    psi_Ex =[]
    psi_Hy = []
    alpha_Ex= []
    alpha_Hy= []
    sigma_Ex =[]   # specific spatial value of conductivity 
    sigma_Hy = []
    beX =[]#
    bmY =[]#np.exp(-(sigmaHy/(permea_0*kappa_Hy) + alpha_Hy/permea_0 )*delT)
    ceX = []
    cmY = []
    Ca = []
    Cb = []
    Cc = []
    C1 = []
    C2 = []
    C3 = []
    
    eLoss_CPML =[]   # sigma e* delT/2*epsilon
    mLoss_CPML = []
    den_Hydz = []
    den_Exdz = [] 
    
   
    def __repr__(self):
        return (f'{self.__class__.__name__}')
    
    def __str__(self):
        return 'Class containing all CPML values that vary throughout a sim' 
#@num.jit
        
def Controller(V, P, C_V, C_P):
    plt.close('all')
    V.x1ColBe = [[]]*(P.timeSteps)
    V.x1ColAf = [[]]*(P.timeSteps)
    for i in range(0,2):
        V.tempVarPol, V.tempTempVarE, V.tempVarE, V.tempTempVarPol, V.polarisationCurr, V.Ex, V.Dx, V.Ex_History, V.Hy, V.Hy_History, V.Psi_Ex_History, V.Psi_Hy_History,V.Exs, V.Hys = BaseFDTD.FieldInit(V,P)
        V.Exs, V.Hys = BaseFDTD.SmoothTurnOn(V,P)
        #print(i)
        V.UpHyMat, V.UpExMat = BaseFDTD.EmptySpaceCalc(V,P)   
        C_V = BaseFDTD.CPML_FieldInit(V,P, C_V, C_P)
        counts = 0
        for counts in range(0,P.timeSteps):
               
        
               if counts == 0:
                   tic = tim.perf_counter()
               C_V.sigma_Ex, C_V.sigma_Hy, C_V.alpha_Ex,  C_V.alpha_Hy, C_V.kappa_Ex, C_V.kappa_Hy= BaseFDTD.CPML_ScalingCalc(V, P, C_V, C_P)
               C_V.beX, C_V.ceX = BaseFDTD.CPML_Ex_RC_Define(V, P, C_V, C_P)
               C_V.bmY, C_V.cmY = BaseFDTD.CPML_HY_RC_Define(V, P, C_V, C_P)
               C_V.eLoss_CPML, C_V.Ca, C_V.Cb, C_V.Cc = BaseFDTD.CPML_Ex_Update_Coef(V,P, C_V, C_P)
               C_V.mLoss_CPML, C_V.C1, C_V.C2, C_V.C3 = BaseFDTD.CPML_Hy_Update_Coef(V,P, C_V, C_P)
               #C_V.den_Exdz, C_V.den_Hydz = BaseFDTD.denominators(V, P, C_V, C_P)
               if counts == 0 :
                       toc= tim.perf_counter()
                       print(toc - tic, "cpml batch ran in..." )
               if i == 1:
                       #print(i)
                       if counts == 0:
                           tic = tim.perf_counter()
                       V.tempTempVarPol, V.tempVarPol, V.tempVarE, V.tempTempVarE = BaseFDTD.ADE_TempPolCurr(V,P)
                       V.polarisationCurr = BaseFDTD.ADE_PolarisationCurrent_Ex(V, P, C_V, C_P)
                       if counts == 0 :
                           toc= tim.perf_counter()
                           print(toc - tic, "polarisation batch ran in..." )
               V.Hy[P.nzsrc-1] = BaseFDTD.HyTfSfCorr(V,P, counts) 
               if counts == 0:
                           tic = tim.perf_counter()
               V.Ex =BaseFDTD.ADE_ExUpdate(V, P, C_V, C_P)
               if counts == 0 :
                           toc= tim.perf_counter()
                           print(toc - tic, "Ex ran in..." )
               #V.Ex =BaseFDTD.CPML_ExUpdate(V, P, C_V, C_P)
               C_V.psi_Ex, V.Ex  = BaseFDTD.CPML_Psi_e_Update(V,P, C_V, C_P)
               V.Ex[P.nzsrc] = BaseFDTD.ExTfSfCorr(V,P, counts)
               
               
               
               
              
               if i == 1: 
                      #  V.epsilon, V.mu, V.UpExHcompsCo, V.UpExSelf, V.UpHyEcompsCo, V.UpHySelf = BaseFDTD.Material(V,P)
                       # V.UpHyMat, V.UpExMat = BaseFDTD.UpdateCoef(V,P)
                        #
                   if counts == 0:
                           tic = tim.perf_counter()     #
                   V.Dx = BaseFDTD.ADE_DxUpdate(V, P, C_V, C_P)
                   if counts == 0 :
                           toc= tim.perf_counter()
                           print(toc - tic, "Dx ran in..." )
                   V.Ex= BaseFDTD.ADE_ExCreate(V, P, C_V, C_P)
               #V.Hy =BaseFDTD.CPML_HyUpdate(V, P, C_V, C_P)
               V.Hy = BaseFDTD.ADE_HyUpdate(V, P, C_V, C_P)
               C_V.psi_Hy, V.Hy  = BaseFDTD.CPML_Psi_m_Update(V,P, C_V, C_P)
               #
               #
               
              
               
               V.Ex_History[counts] = np.insert(V.Ex_History[counts], 0, V.Ex)
               #V.Psi_Ex_History[counts] = np.insert(V.Psi_Ex_History[counts], 0, C_V.psi_Ex)
               #V.Psi_Hy_History[counts] = np.insert(V.Psi_Hy_History[counts], 0, C_V.psi_Hy)
               V.Hy_History[counts] = np.insert(V.Hy_History[counts], 0, V.Hy)
               
               

               if i ==0:
                   V.x1ColBe[counts] = V.Ex_History[counts][P.x1Loc]
      
           
               if i ==1:
                   V.x1ColAf[counts] = V.Ex_History[counts][P.x2Loc]
                 
                  	
    return V, P, C_V, C_P


P= Params()
V=Variables()
C_P = CPML_Params()
C_V = CPML_Variables()

def results(V, P, C_V, C_P, time_Vec, loop = False, RefCo = False, FFT = False, AnalRefCo = False):
    if RefCo == True:
        #print(V.x1ColBe)
        #print(V.x1ColAf)
        #
          
        #for iii in range(len(V.x1ColBe)):
         #   if V.x1ColBe[iii] <= 1e-10:  # be careful with harmonics later
          #      V.x1ColBe[iii] = 0.0
           # if V.x1ColAf[iii] <= 1e-10:
            #    V.x1ColAf[iii] = 0.0
                
                 
        #transm, sig_fft1, sig_fft2, sample_freq, timePadded = 
        sig_fft, sample_freq= FourierTrans(P, V, V.x1ColBe, V.x1ColAf, time_Vec, P.delT)
        sig_fft2, sample_freq= FourierTrans(P, V, V.x1ColBe, V.x1ColAf, time_Vec, P.delT)
        #plt.plot(sample_freq,sig_fft1) 
        
       # plt.plot(sample_freq,abs(sig_fft2))
        #breakpoint()
        reflectCo = ReflectionCalc(P, V, sample_freq, sig_fft, sig_fft2)
        #print(reflectCo, "REF")
        
        return reflectCo
    if AnalRefCo==True:
        analReflectCo = BaseFDTD.AnalyticalReflectionE(V,P)
        return analReflectCo

def plotter(xAxisData, yAxisData1, yAxisData2, yAxisLim = 1, xAxisLabel = " ", yAxisLabel = " ", legend = " ", title= " "):
    fig, ax = plt.subplots()
    ax.clear()
    ax.plot(xAxisData, yAxisData1)
    ax.plot(xAxisData, yAxisData2)
    ax.set_ylim(-yAxisLim, yAxisLim)    
    
  
InputSweepSwitch = {"Input frequency sweep": results,
                "test function" : results }   

def LoopedSim(V,P,C_V, C_P, stringparamSweep = "Input frequency sweep", loop =False,  Low = 1e9, Interval = 1e8, RefCoBool = True):
    #print("loop = ", loop)
    if loop == True:
        points = 5
        dataRange = np.arange(Low, points*Interval+Low, Interval) 
        print(len(dataRange))
        freqDomYAxisRef =np.zeros(points)
        timeDomYAxis = np.zeros(points)       
        freqDomYAxis2AnalRef= np.zeros(points) 
        if stringparamSweep == "Input frequency sweep":
            for loop in range(points): 
                matDef.matSetup(V,P,C_P,C_V, dataRange[loop] ) 
                V, P, C_V, C_P= Controller(V, P, C_V, C_P)
                t =np.arange(0,len(V.x1ColBe))
                
                freqDomYAxisRef[loop] = results(V, P, C_V, C_P, t, loop, True) # One refco y point
                freqDomYAxis2AnalRef[loop] =results(V, P, C_V, C_P, t, loop, AnalRefCo = True)
                #print(freqDomYAxisRef)
                #print(freqDomYAxis2AnalRef)
                #print(freqDomYAxisRef, freqDomYAxis2AnalRef, "Y STUFF")
        #plotter(dataRange, yAxisData1 =freqDomYAxisRef, yAxisData2 =freqDomYAxis2AnalRef)  
        
    elif loop == False:
        freqDomYAxisRef =[]
        freqDomYAxis2AnalRef= []
        tic =tim.perf_counter()
        matDef.matSetup(V,P,C_P,C_V, newFreq_in = Low)
        toc = tim.perf_counter()
        print(toc-tic, "time for matsetup to run")
        tic =tim.perf_counter()
        V, P, C_V, C_P= Controller(V, P, C_V, C_P)
        toc = tim.perf_counter()
        print(toc-tic, "time for controller to run")
        xAxis = np.zeros(P.Nz+1)
        for i in range(0, P.Nz):
            xAxis[i] = i
        t =np.arange(0,len(V.x1ColBe))
        #tPad = np.zeros(len) 
        tic =tim.perf_counter()
        freqDomYAxisRef = results(V, P, C_V, C_P, t, loop, True)
        freqDomYAxis2AnalRef = results(V, P, C_V, C_P, t, loop, AnalRefCo = True)
        toc = tim.perf_counter()
        print(toc-tic, "time for results to run")
        #plotter(xAxis, V.Ex_History[20], yAxisLim =2)  
        print(freqDomYAxisRef, "measured")
        print(freqDomYAxis2AnalRef, "analytical")
        VideoMaker(P, V)
    #winsound.Beep(freq, duration)  
    engine = pyttsx3.init()
    engine.say('beep')
    engine.runAndWait()
    return V, P, C_V, C_P


tic = tim.perf_counter()
V, P, C_V, C_P = LoopedSim(V,P,C_V, C_P, loop = False, Low =0.78e9)
toc = tim.perf_counter()
print(toc-tic, "Time for Looped sim to run non-loop")


#t =np.arange(0,len(V.x1ColBe))


UpHyMat = V.UpHyMat
UpExMat = V.UpExMat
Ex = V.Ex
Hy = V.Hy
Ex_History = V.Ex_History
Hy_History = V.Hy_History
Psi_Ex_History=V.Psi_Ex_History
Psi_Hy_History=V.Psi_Hy_History
Exs = V.Exs
Hys = V.Hys
x1ColBe = V.x1ColBe
x1ColAf = V.x1ColAf
epsilon = V.epsilon
mu = V.mu
polarisationCurr= V.polarisationCurr

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
eLoss = P.eLoss
eSelfCo = P.eSelfCo
eHcompsCo = P.eHcompsCo
mLoss = P.mLoss
hSelfCo = P.hSelfCo
hEcompsCo = P.hEcompsCo
pmlWidth = P.pmlWidth
######


###CPML PARAMS

kappaMax = C_P.kappaMax
sigmaEMax= C_P.sigmaEMax
sigmaHMax= C_P.sigmaHMax
sigmaOpt= C_P.sigmaOpt
alphaMax = C_P.alphaMax
r_scale = C_P.r_scale
r_a_scale = C_P.r_a_scale

##CPML Vars

kappa_Ex = C_V.kappa_Ex
kappa_Hy = C_V.kappa_Hy
psi_Ex = C_V.psi_Ex
psi_Hy=  C_V.psi_Hy
alpha_Ex =  C_V.alpha_Ex#
alpha_Hy = C_V.alpha_Hy
sigma_Ex =  C_V.sigma_Ex
sigma_Hy = C_V.sigma_Hy
beX = C_V.beX
bmY = C_V.bmY
ceX = C_V.ceX
cmY= C_V.cmY
Ca = C_V.Ca
Cb= C_V.Cb
Cc= C_V.Cc
C1= C_V.C1
C2= C_V.C2
C3= C_V.C3 

eLoss_CPML = C_V.eLoss_CPML
mLoss_CPML = C_V.mLoss_CPML
den_Hydz = C_V.den_Hydz
den_Exdz =C_V.den_Exdz


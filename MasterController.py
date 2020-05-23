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
import Environment_Setup as envDef
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
from numba import njit as num
from numba import jitclass as jclass
from numba import int32, float32, int64, float64, ndarray
import time as tim
 

duration = 1000  # milliseconds
freq = 440  # Hz

specV = [('Nz', int32),              
    ('timeSteps', float32),
    ('UpHyMat', ndarray),
    ('UpExMat', ndarray),
    ('UpExHcompsCo', ndarray), 
    ('UpExSelf', ndarray), 
    ('UpHySelf', ndarray),
    ('UpHyEcompsCo', ndarray),
    ('Ex', ndarray),
    ('Hy', ndarray), 
    ('Dx', ndarray),
    ('Ex_History',float32[:] ),
    ('Hy_History', float32[:]),
    ('Psi_Ex_History', float32[:]),
    ('Psi_Hy_History', float32[:]),
    ('Exs', float32[:]),
    ('Hys', float32[:]),
    ('x1ColBe', float32[:]),
    ('x1ColAf', float32[:]),
    ('epsilon', float32[:] ),
    ('mu', float32[:]),
    ('inputVec', float32[:])
    ('outputPlots', float32[:]), 
    ('polarisationCurr', ndarray)
    ('plasmaFreqE', float32),
    ('gammaE', float32),
    ('omega_0E', float32),
    ('tempVarPol', float32[:]),
    ('tempTempVarPol', float32[:]),
    ('tempVarE', float32[:]),
    ('tempTempVarE', float32[:])]          
@num.jitclass(specV)
class Variables(object):

    def __init__(self, Nz, timeSteps):
        Nz = Nz+1
        self.UpHyMat = np.ones(Nz)
        self.UpExMat = np.ones(Nz)
        self.UpExHcompsCo = np.ones(Nz)
        self.UpExSelf =np.ones(Nz)
        self.UpHySelf = np.ones(Nz)
        self.UpHyEcompsCo = np.ones(Nz)
        self.Ex = np.zeros(timeSteps)
        self.Hy = np.zeros(timeSteps)
        self.Dx = np.zeros(timeSteps)
       # self.By = By
        self.Ex_History = [[]]*timeSteps
        self.Hy_History = [[]]*timeSteps
        self.Psi_Ex_History= [[]]*timeSteps
        self.Psi_Hy_History = [[]]*timeSteps
        self.Exs = np.zeros(timeSteps)
        self.Hys = np.zeros(timeSteps)
        self.x1ColBe = np.zeros(timeSteps)
        self.x1ColAf = np.zeros(timeSteps)
        self.epsilon = np.ones(P.Nz)
        self.mu = np.ones(P.Nz)
        self.inputVec = []
        self.outputPlots = []
        self.polarisationCurr = np.zeros(P.Nz)
        self.plasmaFreqE = 2*np.pi*1.1027e9
        self.gammaE = 1e8
        self.omega_0E = 2*np.pi*0.1591e9
        self.tempVarPol =np.zeros(timeSteps)
        self.tempTempVarPol =np.zeros(timeSteps)
        self.tempVarE =np.zeros(timeSteps)
        self.tempTempVarE =np.zeros(timeSteps)
        
    def __str__(self):
        return 'Contains data that will change during sim'
    
    def __repr__(self):
        return (f'{self.__class__.__name__}', ": Contains field variables that change during sim")
       
        
    # methods to handle user input errors during instantiation.
    
class Params(object):
    
    def __init__(self, Nz, timeSteps, eLoss, mLoss, eSelfCo, eHcompsCo, hSelfCo, hEcompsCo, x1Loc, x2Loc, materialFrontEdge, materialRearEdge, pmlWidth, nzsrc, lamMin, dz, delT, courantNo, period, domainSize, freq_in):
        #self.epsRe = epsRe
        #self.muRe = muRe
        
        self.permit_0 = sci.constants.epsilon_0
        self.permea_0 = sci.constants.mu_0
        self.CharImp =376.730313668
        self.c0 = 299792458.0
        self.freq_in = freq_in
        self.lamMin = lamMin
        self.Nlam = 40
        self.dz = dz
        self.delT = delT
        self.courantNo= courantNo
        self.materialFrontEdge = materialFrontEdge
        self.materialRearEdge = materialRearEdge
        self.Nz = Nz
        self.timeSteps = timeSteps
        self.x1Loc = x1Loc
        self.x2Loc = x2Loc
        self.nzsrc = nzsrc
        self.period = period
        self.eLoss = eLoss
        self.eSelfCo = eSelfCo
        self.eHcompsCo = eHcompsCo
        self.mLoss = mLoss
        self.hSelfCo = hSelfCo
        self.hEcompsCo = hEcompsCo
        self.pmlWidth = pmlWidth
        self.domainSize = domainSize
        
    def __repr__(self):
        return (f'{self.__class__.__name__}'(f'{self.epsRe!r}, {self.muRe!r}'))
    
    def __str__(self):
        return 'Class containing all values that remain constant throughout a sim' 


class CPML_Params(object):
    def __init__(self, dz):
        self.kappaMax =12 # 'Stretching co-ordinate of pml, to minimise numerical dispersion set it as 1' : DOI: 10.22190/FUACR1703229G see conclusion
        self.r_scale = 5.4# Within ideal bounds see Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17 (scaling power is called 'm' )
        self.r_a_scale=1
        self.sigmaEMax=1.4*(0.8*(self.r_scale)/(dz*(sci.constants.mu_0/sci.constants.epsilon_0)**0.5))#1.1*sigmaOpt # Within ideal bounds for value, : Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
        self.sigmaHMax =self.sigmaEMax#1.1*sigmaOpt # See International Journal of Computer Science and Network Security, VOL.18 No.12, December 2018, page 4 right hand side.
        self.sigmaOpt  =self.sigmaEMax
    #Optimal value of pml conductivity at far end of pml: DOI: 10.22190/FUACR1703229G see equation 13
    
    
    alphaMax= 0.05# with bounds of ideal cpml alpha max, complex frequency shift parameter, Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
    
    def __repr__(self):
        return (f'{self.__class__.__name__}')
    
    def __str__(self):
        return 'Class containing all CPML values that remain constant throughout a sim' 
    
    
class CPML_Variables(object):
    def __init__(self, Nz):
        Nz = Nz+1
        self.kappa_Ex = np.zeros(Nz)
        self.kappa_Hy = np.zeros(Nz)
        self.psi_Ex =  np.zeros(Nz)
        self.psi_Hy =  np.zeros(Nz)
        self.alpha_Ex = np.zeros(Nz)
        self.alpha_Hy = np.zeros(Nz)
        self.sigma_Ex =np.zeros(Nz)   # specific spatial value of conductivity 
        self.sigma_Hy = np.zeros(Nz)
        self.beX =np.zeros(Nz)
        self.bmY =np.zeros(Nz)#np.exp(-(sigmaHy/(permea_0*kappa_Hy) + alpha_Hy/permea_0 )*delT)
        self.ceX = np.zeros(Nz)
        self.cmY = np.zeros(Nz)
        self.Ca = np.zeros(Nz)
        self.Cb = np.zeros(Nz)
        self.Cc = np.zeros(Nz)
        self.C1 = np.zeros(Nz)
        self.C2 = np.zeros(Nz)
        self.C3 = np.zeros(Nz)
        
        self.eLoss_CPML =np.zeros(Nz)   # sigma e* delT/2*epsilon
        self.mLoss_CPML = np.zeros(Nz)
        self.den_Hydz = np.zeros(Nz)
        self.den_Exdz = np.zeros(Nz) 
    
   
    def __repr__(self):
        return (f'{self.__class__.__name__}')
    
    def __str__(self):
        return 'Class containing all CPML values that vary throughout a sim' 
@num.njit       
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

# calc nz and timesteps from domain size and freq
    #keep same domain once it has been setup, so matsetup doesn't need to be called during run.

domainSize = 1000
freq_in = 1e9

# using matsetup anyway, feed in from here?
setupReturn = []*19

setupReturn=envDef.envSetup(freq_in, domainSize)

P= Params(*setupReturn, domainSize, freq_in) #be careful with tuple index
V=Variables(P.Nz, P.timeSteps)
C_P = CPML_Params(P.dz)
C_V = CPML_Variables(P.Nz)





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
               # matDef.matSetup(V,P,C_P,C_V, dataRange[loop] ) 
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
        #matDef.matSetup(V,P,C_P,C_V, newFreq_in = Low)
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
    engine.say('beep boop beep')
    engine.runAndWait()
    return V, P, C_V, C_P


tic = tim.perf_counter()
V, P, C_V, C_P = LoopedSim(V,P,C_V, C_P, loop = False, Low =P.freq_in)
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
#epsRe = P.epsRe
#clearmuRe =  P.muRe
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


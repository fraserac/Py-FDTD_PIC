  # -*- coding: utf-8 -*-
"""
Fields controller, this script is the master that controls all the processes,
calling the field updates and eventually particle updaters and field interpolater
as well as any memory saving methods like sparse matrices etc.
BC functions called as well
If GUI is incorporated it will act as a direct interface with this script.
"""

import numpy as np
import scipy as sci
from scipy import sparse 
import matplotlib.pylab as plt
import genericStability as gStab
 
import BaseFDTD11
import Environment_Setup as envDef
from Validation_Physics import VideoMaker 
 
import TransformHandler as transH
import TransformHandler as tHand
import winsound
import pyttsx3
from sklearn.linear_model import Ridge

from numba import njit as nj
from numba.experimental import jitclass as jclass
from numba import int32, float32, int64, float64, boolean
import time as tim
#from memory_profiler import profile
import BulkTest as bt
import matrixConstruct as matCon
from varname import nameof
import sys
duration = 1000  # milliseconds
freq = 300  # Hz


"""
Below, specify typings of jclass members, as numba 
doesn't use dynamic typing like base python.

"""

specV = [('Nz', int32),              
    ('timeSteps', float32),
    ('UpHyMat', float64[:]),
    ('UpExMat', float64[:]),
    ('UpExHcompsCo',float64[:]), 
    ('UpExSelf', float64[:]), 
    ('UpHySelf', float64[:]),
    ('UpHyEcompsCo', float64[:]),
    ('Ex',float64[:]),
    ('Hy', float64[:]), 
    ('Dx', float64[:]),
    ('Jx',float64[:]),
    ('Ex_History',float64[:,:]),
    ('Hy_History', float64[:,:]), 
    ('Jx_History', float64[:,:]),
    ('polCurr_History', float64[:,:]),
    ('Dx_History',float64[:,:]),
    ('Psi_e_History', float64[:,:]),
    ('x1ColBe', float64[:]),
    ('x1ColAf', float64[:]),
    ('x1Jx',  float64[:]),
    ('x1Hy',  float64[:]),
    ('x1ExOld',  float64[:]),
    ('x1JxOld',  float64[:]),
    ('x1HyOld',  float64[:]),
    ('epsilon', float64[:] ),
    ('mu', float64[:]), 
    ('polarisationCurr', float64[:]),
    ('plasmaFreqE', float64),
    ('gammaE', float32),
    ('omega_0E', float32),
    ('tempVarPol', float64[:]),
    ('tempTempVarPol', float64[:]),
    ('tempVarE', float64[:]),
    ('tempTempVarE', float64[:]), 
    ('tempVarHy', float64[:]),
    ('tempTempVarHy', float64[:]),
    ('tempVarJx', float64[:]),
    ('tempTempVarJx', float64[:])]
         
@jclass(specV)
class Variables(object):

    def __init__(self, Nz, timeSteps):
        Nz = Nz+1
        self.UpHyMat = np.ones(Nz)
        self.UpExMat = np.ones(Nz)
        self.UpExHcompsCo = np.ones(Nz)
        self.UpExSelf =np.ones(Nz)
        self.UpHySelf = np.ones(Nz)
        self.UpHyEcompsCo = np.ones(Nz)
        self.Ex = np.zeros(Nz)
        self.Hy = np.zeros(Nz)
        self.Dx = np.zeros(Nz)
        self.Jx = np.zeros(Nz)
        self.Ex_History= np.zeros((Nz)*timeSteps).reshape(timeSteps, (Nz))
        self.Hy_History = np.zeros((Nz)*timeSteps).reshape(timeSteps, (Nz))
        self.Jx_History = np.zeros((Nz)*timeSteps).reshape(timeSteps, (Nz))
        self.polCurr_History = np.zeros((Nz)*timeSteps).reshape(timeSteps, (Nz))
        self.Dx_History = np.zeros((Nz)*timeSteps).reshape(timeSteps, (Nz))
        self.Psi_e_History =np.zeros((Nz)*timeSteps).reshape(timeSteps, (Nz))
        self.x1ColBe = np.zeros(timeSteps)
        self.x1ColAf = np.zeros(timeSteps)
        self.x1Jx = np.zeros(timeSteps)
        self.x1Hy = np.zeros(timeSteps)
        self.x1ExOld = np.zeros(timeSteps)
        self.x1JxOld = np.zeros(timeSteps)
        self.x1HyOld = np.zeros(timeSteps)
        self.epsilon = np.ones(Nz)
        self.mu = np.ones(Nz)
        self.polarisationCurr = np.zeros(Nz)
        self.plasmaFreqE = 10e9
        self.gammaE = 1e8
        self.omega_0E= 5e9
        self.tempVarPol =np.zeros(Nz)
        self.tempTempVarPol =np.zeros(Nz)
        self.tempVarE =np.zeros(Nz)
        self.tempTempVarE =np.zeros(Nz)
        self.tempVarHy =np.zeros(Nz)
        self.tempTempVarHy =np.zeros(Nz)
        self.tempVarJx =np.zeros(Nz)
        self.tempTempVarJx =np.zeros(Nz)
        
        
    def __str__(self):
        return 'Contains data that will change during sim'
    
    def __repr__(self):
        return (f'{self.__class__.__name__}', ": Contains field variables that change during sim")
       
        
    # methods to handle user input errors during instantiation.
    
specP=[('Nz', int32),              
    ('timeSteps', float32), 
    ('eLoss', int32),
    ('mLoss', int32),
    ('eSelfCo', float32),
    ('eHcompsCo', float32),
    ('hSelfCo', float32),
    ('hEcompsCo', float32),
    ('x1Loc', int32),
    ('x2Loc', int32),
    ('materialFrontEdge', int32 ),
    ('materialRearEdge', int32),
    ('pmlWidth', int32), 
    ('nzsrc', int32),
    ('lamMin', int32),
    ('dz', float64),
    ('delT', float64),
    ('courantNo', float32),
    ('period', float32), 
    ('domainSize', int32),
    ('freq_in', float64), 
    ('permit_0', float64),
    ('permea_0', float64),
    ('CharImp', float64),
    ('c0', float64),
    ('freq_in', float64),
    ('lamMin', float64),
    ('Nlam', float64),
    ('dz', float64),
    ('delT', float64),
    ('courantNo', float32),
    ('materialFrontEdge', int32),
    ('materialRearEdge', int32),
    ('Nz', int32),
    ('timeSteps',int32),
    ('x1Loc',int32),
    ('x2Loc', int32),
    ('nzsrc', int32),
    ('period', float64),
    ('eLoss', float64),
    ('eSelfCo', float64),
    ('eHcompsCo', float64),
    ('mLoss', float64),
    ('hSelfCo', float64),
    ('hEcompsCo', float64),
    ('pmlWidth', int32),
    ('hEcompsCo', float64),
    ('domainSize', int32), 
    ('MORmode', boolean), 
    ('delayMOR', int32)]


@jclass(specP)
class Params(object):
    def __init__(self,  Nz, timeSteps, eLoss, mLoss, eSelfCo, eHcompsCo, hSelfCo, hEcompsCo, x1Loc, x2Loc, materialFrontEdge, materialRearEdge, pmlWidth, nzsrc, lamMin, dz, delT, courantNo, period, Nlam, MORmode, domainSize, freq_in, delayMOR):
        
        self.permit_0 = sci.constants.epsilon_0
        self.permea_0 = sci.constants.mu_0
        self.CharImp =376.730313668
        self.c0 = 299792458.0
        self.freq_in = freq_in
        self.lamMin = lamMin
        self.Nlam = Nlam
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
        self.MORmode = MORmode
        self.delayMOR = delayMOR
        
    def __repr__(self):
        return (f'{self.__class__.__name__}'(f'{self.epsRe!r}, {self.muRe!r}'))
    
    def __str__(self):
        return 'Class containing all values that remain constant throughout a sim' 

specCP = [('kappaMax',float32),
          ('r_scale',float32),
          ('r_a_scale',float32),
          ('sigmaEMax',float64),
          ('sigmaHMax', float64),
          ('sigmaOpt',float64),
          ('alphaMax',float32),]


@jclass(specCP)
class CPML_Params(object):
    def __init__(self, dz):
        self.kappaMax =10# 'Stretching co-ordinate of pml, to minimise numerical dispersion set it as 1' : DOI: 10.22190/FUACR1703229G see conclusion
        self.r_scale = 4.0 #Within ideal bounds see Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17 (scaling power is called 'm' )
        self.r_a_scale=1.0
        self.sigmaEMax=10*(0.8*(1)/(dz*(sci.constants.mu_0/sci.constants.epsilon_0)**0.5))#1.1*sigmaOpt # Within ideal bounds for value, : Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
        self.sigmaHMax =10*(0.8*(1)/(dz*(sci.constants.mu_0/sci.constants.epsilon_0)**0.5))#1.1*sigmaOpt # See International Journal of Computer Science and Network Security, VOL.18 No.12, December 2018, page 4 right hand side.
        self.sigmaOpt  =10*(0.8*(1)/(dz*(sci.constants.mu_0/sci.constants.epsilon_0)**0.5))
    #Optimal value of pml conductivity at far end of pml: DOI: 10.22190/FUACR1703229G see equation 13
        self.alphaMax=0.05# with bounds of ideal cpml alpha max, complex frequency shift parameter, Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
    
    def __repr__(self):
        return (f'{self.__class__.__name__}')
    
    def __str__(self):
        return 'Class containing all CPML values that remain constant throughout a sim' 
    
    
specCV = [('Nz', int32),
          ('kappa_Ex', float64[:]),
          ('kappa_Hy', float64[:]),
          ('psi_Ex', float64[:]),
          ('psi_Ex_Probe', float64[:]),
          ('psi_Hy', float64[:]),
          ('psi_Hy_Probe', float64[:]),
          ('alpha_Ex', float64[:]),
          ('alpha_Hy', float64[:]),
          ('sigma_Ex', float64[:]),
          ('sigma_Hy', float64[:]),
          ('beX', float64[:]),
          ('bmY', float64[:]),
          ('ceX', float64[:]),
          ('cmY', float64[:]),
          ('Ca', float64[:]),
          ('Cb', float64[:]),
          ('Cc', float64[:]),
          ('C1', float64[:]),
          ('C2', float64[:]),
          ('C3', float64[:]),
          ('eLoss_CPML', float64[:]),
          ('mLoss_CPML', float64[:]),
          ('den_Hydz', float64[:]),
          ('den_Exdz', float64[:]),
          ('tempTempVarPsiEx', float64[:]),
          ('tempVarPsiEx', float64[:]),
          ('tempTempVarPsiHy', float64[:]),
          ('tempVarPsiHy', float64[:]),
          ('psi_Ex_Old', float64[:]),
          ('psi_Hy_Old', float64[:])]
@jclass(specCV)
class CPML_Variables(object):
    def __init__(self, Nz, timeSteps):
        Nz = Nz+1
        self.kappa_Ex = np.zeros(Nz)
        self.kappa_Hy = np.zeros(Nz)
        self.psi_Ex =  np.zeros(Nz)
        self.psi_Ex_Probe = np.zeros(timeSteps)
        self.psi_Hy =  np.zeros(Nz)
        self.psi_Hy_Probe = np.zeros(timeSteps)
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
        self.tempTempVarPsiEx = np.zeros(Nz)
        self.tempVarPsiEx = np.zeros(Nz)
        self.tempTempVarPsiHy = np.zeros(Nz)
        self.tempVarPsiHy = np.zeros(Nz)
        self.psi_Ex_Old= np.zeros(timeSteps)
        self.psi_Hy_Old = np.zeros(timeSteps)
    
    def __repr__(self):
        return (f'{self.__class__.__name__}')
    
    def __str__(self):
        return 'Class containing all CPML values that vary throughout a sim' 


#CONTROLLER: calls functions for simulation, not hooked up to MOR set up. 
#Initialises all vars (NOT NECESSARY if instantiating class each time) 
#BASEFDTD update equations for loop for time stepping, move to numba function integrator
@nj
def integrator(V, P, C_V, C_P,Exs, Hys, i, probeReadStart):
    
    for counts in range(0,P.timeSteps):
            
        C_V.psi_Ex, V.Ex  = BaseFDTD11.CPML_Psi_e_Update(V,P, C_V, C_P)
        V.Ex = BaseFDTD11.ADE_ExUpdate(V,P, C_V, C_P)
        if i == 1:
             V.tempTempVarPol, V.tempVarPol, V.tempVarE, V.tempTempVarE, V.tempTempVarHy, V.tempVarHy, V.tempTempVarJx, V.tempVarJx, C_V.tempTempVarPsiEx, C_V.tempVarPsiEx, C_V.tempTempVarPsiHy, C_V.tempVarPsiHy = BaseFDTD11.ADE_TempPolCurr(V,P, C_V, C_P)
             V.polarisationCurr = BaseFDTD11.ADE_PolarisationCurrent_Ex(V, P, C_V, C_P)
             V.Jx = BaseFDTD11.ADE_JxUpdate(V,P, C_V, C_P)  
        
        
        
        V.Hy[P.nzsrc-1] = BaseFDTD11.HyTfSfCorr(V,P, counts, Exs)
        V.Ex[P.nzsrc] = BaseFDTD11.ExTfSfCorr(V,P, counts, Hys)
        
        V.Hy = BaseFDTD11.ADE_HyUpdate(V,P, C_V, C_P)
        C_V.psi_Hy, V.Hy  = BaseFDTD11.CPML_Psi_m_Update(V,P, C_V, C_P)
        
       
        #C_V.psi_Ex, V.Ex  = BaseFDTD11.CPML_Psi_e_Update(V,P, C_V, C_P)
           
        V.Ex_History[counts] = V.Ex
        #breakpoint()
        #print(np.max(V.Ex),"Ex max")
        #Psi_Ex_History[counts] = C_V.psi_Ex
        #V.Psi_e_History[counts] = C_V.psi_Ex
        #V.Hy_History[counts] = V.Hy
        #V.Jx_History[counts] = V.Jx
        #V.Dx_History[counts] = V.Dx
        #V.polCurr_History[counts] = V.polarisationCurr
               
        if i ==0:
             if counts <= P.timeSteps-1:
                 if counts >= probeReadStart:
               # print("x1colbe")
                    V.x1ColBe[counts] = V.Ex_History[counts][P.x1Loc] 
            
        elif i ==1:
             if counts <= P.timeSteps-1:
                  if counts >= probeReadStart:
                     V.x1ColAf[counts] = V.Ex_History[counts][P.x2Loc] 
                     V.x1Hy[counts] = V.Hy[P.x2Loc]
                     V.x1Jx[counts] = V.Jx[P.materialFrontEdge +10]
                     C_V.psi_Ex_Probe[counts] = C_V.psi_Ex[P.x2Loc]
                     C_V.psi_Hy_Probe[counts] = C_V.psi_Hy[P.x2Loc]
                     
                     V.x1ExOld[counts] = V.tempTempVarE[P.x2Loc]
                     V.x1HyOld[counts] = V.tempTempVarHy[P.x2Loc]
                     V.x1JxOld[counts] = V.tempTempVarJx[P.x2Loc]
                     C_V.psi_Ex_Old[counts] = C_V.tempTempVarPsiEx[P.x2Loc]
                     C_V.psi_Hy_Old[counts] = C_V.tempTempVarPsiEx[P.x2Loc]
 
    
    return V.Ex, V.Hy,  C_V.psi_Ex,C_V.psi_Hy, V.x1ColBe,  V.x1ColAf


def Controller(V, P, C_V, C_P,Exs, Hys):
    probeReadStart = int(P.timeSteps*0.05)
    V.x1ColBe = np.zeros(P.timeSteps)
    V.x1ColAf = np.zeros(P.timeSteps)
    Exs = np.asarray(Exs)
    Hys = np.asarray(Hys)
    for i in range(0, 2):
        V.tempVarPol, V.tempTempVarE, V.tempVarE, V.tempTempVarPol, V.polarisationCurr, V.Ex, V.Dx, V.Hy = BaseFDTD11.FieldInit(V,P)
        
        V.UpHyMat, V.UpExMat = BaseFDTD11.EmptySpaceCalc(V,P)   
        
        C_V = BaseFDTD11.CPML_FieldInit(V,P, C_V, C_P)
        analReflectCo, dispPhaseVel, realV = BaseFDTD11.AnalyticalReflectionE(V,P)
        lamCont, lamDisc, diff, V.plasmaFreqE, fix = gStab.spatialStab(P.timeSteps,P.Nz,P.dz, P.freq_in, P.delT, V.plasmaFreqE, V.omega_0E, V.gammaE)
        C_V.sigma_Ex, C_V.sigma_Hy, C_V.alpha_Ex,  C_V.alpha_Hy, C_V.kappa_Ex, C_V.kappa_Hy= BaseFDTD11.CPML_ScalingCalc(V, P, C_V, C_P)
        C_V.beX, C_V.ceX = BaseFDTD11.CPML_Ex_RC_Define(V, P, C_V, C_P)
        C_V.bmY, C_V.cmY = BaseFDTD11.CPML_HY_RC_Define(V, P, C_V, C_P)
        C_V.eLoss_CPML, C_V.Ca, C_V.Cb, C_V.Cc = BaseFDTD11.CPML_Ex_Update_Coef(V,P, C_V, C_P)
        C_V.mLoss_CPML, C_V.C1, C_V.C2, C_V.C3 = BaseFDTD11.CPML_Hy_Update_Coef(V,P, C_V, C_P)
        C_V.den_Exdz, C_V.den_Hydz = BaseFDTD11.denominators(V, P, C_V, C_P)
        #gStab.vonNeumannAnalysis(V,P,C_V,C_P)
        
        
        
        V.Ex, V.Hy,  C_V.psi_Ex,C_V.psi_Hy, V.x1ColBe,  V.x1ColAf = integrator(V, P, C_V, C_P,Exs, Hys, i, probeReadStart)
               
        
    return V, P, C_V, C_P, Exs, Hys

 
#######################################
###############   VARIABLES AND CALLS FOR PROGRAM INITIATION
#########

MORmode = False
domainSize=2000
freq_in =1e9
delayMOR =10
noOfEnvOuts = 20
setupReturn = []*noOfEnvOuts
setupReturn =envDef.envSetup(freq_in, domainSize, 1500, 1750)   # this function returns a list with all evaluated model parameters
P= Params(*setupReturn, MORmode, domainSize, freq_in, delayMOR) #be careful with tuple, check ordering of parameters 
V=Variables(P.Nz, P.timeSteps)
C_P = CPML_Params(P.dz)
C_V = CPML_Variables(P.Nz, P.timeSteps)





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
        sig_fft, sample_freq= transH.genFourierTrans(V,P, C_V, C_P, V.x1ColBe)
        sig_fft2, sample_freq= transH.genFourierTrans(V,P, C_V, C_P, V.x1ColAf)
        #plt.plot(sample_freq,sig_fft) 
        
       # plt.plot(sample_freq,abs(sig_fft2))
        #breakpoint()
        reflectCo = transH.ReflectionCalc(P, V, sample_freq, sig_fft, sig_fft2)
        print(abs(reflectCo), "FDTD reflection")
        
        return reflectCo
    if AnalRefCo==True:
        analReflectCo, dispPhaseVel, realV = BaseFDTD11.AnalyticalReflectionE(V,P)
        return analReflectCo, dispPhaseVel, realV

def plotter(xAxisData, yAxisData1, yAxisData2, yAxisLim = 1, xAxisLabel = " ", yAxisLabel = " ", legend = " ", title= " "):
    fig, ax = plt.subplots()
    ax.clear()
    ax.plot(xAxisData, yAxisData1)
    ax.plot(xAxisData, yAxisData2)
    ax.set_ylim(-yAxisLim, yAxisLim)    
    
  
InputSweepSwitch = {"Input frequency sweep": results,
                "test function" : results } 
  

#@jit
def LoopedSim(V,P,C_V, C_P, MORmode, stringparamSweep = "Input frequency sweep", loop =False,  Low = 1e9, Interval = 1e8, RefCoBool = True):
    #print("loop = ", loop)
    if MORmode == False:
        print("Forward difference, yee grid iterative method enabled...")
        if loop == True:
            points = 5
            dataRange = np.arange(Low, points*Interval+Low, Interval) 
            #print(len(dataRange))
            freqDomYAxisRef =np.zeros(points)
            timeDomYAxis = np.zeros(points)       
            freqDomYAxis2AnalRef= np.zeros(points) 
            Exs, Hys = BaseFDTD11.SmoothTurnOn(V,P)
            if stringparamSweep == "Input frequency sweep":
                for loop in range(points): 
                   # matDef.matSetup(V,P,C_P,C_V, dataRange[loop] ) 
                    V, P, C_V, C_P= Controller(V, P, C_V, C_P, Exs, Hys)
                    
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
           
            Exs, Hys = BaseFDTD11.SmoothTurnOn(V,P)
            
            toc = tim.perf_counter()
            print(toc-tic, "time for matsetup to run")
            tic =tim.perf_counter()
            V, P, C_V, C_P, Exs, Hys= Controller(V, P, C_V, C_P, Exs, Hys)
            print("Max vals: Ex, Jx, Hy: ", np.max(V.x1ColBe),np.max(V.x1Jx),np.max(V.x1Hy))
            tocc = tim.perf_counter()
            fdDataEx = tHand.genFourierTrans(V,P,C_V, C_P, V.x1ColAf)
            ticc = tim.perf_counter()
           # print('fDData fft takes this many seconds:', ticc-tocc)
            fdDataHy = tHand.genFourierTrans(V,P,C_V, C_P, V.x1Hy)
            fdDataJx = tHand.genFourierTrans(V,P,C_V, C_P, V.x1Jx)
            fdDataPsiEx = tHand.genFourierTrans(V,P,C_V, C_P, C_V.psi_Ex_Probe)
            fdDataPsiHy = tHand.genFourierTrans(V,P,C_V, C_P, C_V.psi_Hy_Probe)
            
            fdDataExOld = tHand.genFourierTrans(V,P,C_V, C_P, V.x1ExOld)
            fdDataHyOld = tHand.genFourierTrans(V,P,C_V, C_P, V.x1HyOld)
            fdDataJxOld = tHand.genFourierTrans(V,P,C_V, C_P, V.x1JxOld)
            fdDataPsiExOld = tHand.genFourierTrans(V,P,C_V, C_P, C_V.psi_Ex_Old)
            fdDataPsiHyOld = tHand.genFourierTrans(V,P,C_V, C_P, C_V.psi_Hy_Old)
           
    
            
            toc = tim.perf_counter()
            
            t =np.arange(0,len(V.x1ColBe))*P.delT
            spatial = np.arange(P.Nz+1)*P.dz
            """
            histDict = V.Ex_History
            h2 = V.Hy_History
            h3 = V.Jx_History
            h4 = V.Dx_History
            h5 = V.polCurr_History
            
            bt.genericTest(histDict, inputTimeVal = t, inputSpaceVal = spatial, nameofVar = "Ex_Hist")
            bt.genericTest(h2, inputTimeVal = t, inputSpaceVal = spatial, nameofVar ="Hy_Hist") 
            bt.genericTest(h3, inputTimeVal = t, inputSpaceVal = spatial,nameofVar = "Jx_Hist")
            bt.genericTest(h4, inputTimeVal = t, inputSpaceVal = spatial, nameofVar ="Dx_Hist") 
            bt.genericTest(h5, inputTimeVal = t, inputSpaceVal = spatial,nameofVar = "polCurr_Hist")
            """
            
            print(toc-tic, "time for controller to run")
            xAxis = np.zeros(P.Nz+1)
            for i in range(0, P.Nz):
                xAxis[i] = i
            
            #tPad = np.zeros(len) 
            tic =tim.perf_counter()
            freqDomYAxisRef= results(V, P, C_V, C_P, t, loop, RefCo= True)
            freqDomYAxis2AnalRef, dispVel, realV = results(V, P, C_V, C_P, t, loop, AnalRefCo = True)
            
            toc = tim.perf_counter()
            print(toc-tic, "time for results to run")
            #plotter(xAxis, V.Ex_History[20], yAxisLim =2)  
            print(1-freqDomYAxisRef, "measured ref")
            print(dispVel, realV,"numerical phase velocity, vs real velocity" )
            print(abs(freqDomYAxis2AnalRef), "analytical ref")
            VideoMaker(P, V)
        winsound.Beep(freq, duration)  
        engine = pyttsx3.init()
        engine.say('beep')
        engine.runAndWait() 
        
        
        
    elif MORmode == True:
        print("Model order reduction mode enabled...")
        tic = tim.perf_counter()

#figure out how to use sparse matrices in numba, indptr, indices and data members could help
        """
        A, B, source = matCon.initMat(V, P, C_V, C_P)
        
        De, Dh, K, Kt = matCon.blockBuilder(V, P, C_V, C_P)
        
        
        
        Xn = matCon.BasisVector(V, P, C_V, C_P)
        UnP1A, UnP1B, B = matCon.BAndSourceVector(V, P, C_V, C_P)
        
        B, V.Ex, V.Ex_History, V.Hy, UnP1, Xn = matCon.TimeIter(A, B, UnP1A, UnP1B, Xn, V, P, C_V, C_P)
        """
        
        A, B, source = matCon.initMat(V, P, C_V, C_P)
        De, Dh, K, Kt = matCon.blockBuilder(V, P, C_V, C_P)
        R, F = matCon.RandFBuild(P, De, Dh, K, Kt)
       # A, blocks =matCon.ABuild(A, P, De, Dh, K, Kt)
        #print("(R+F)^-1*(R-F), stability")
       
        
       # toInv = R.todense()+F.todense()
        #inverse = np.linalg.inv(toInv)
        #inverse = sparse.csc_matrix(inverse)
        
        #matCon.vonNeumannAnalysisMOR(V,P,C_V,C_P, inverse@(R-F))
       
        
        Xn = matCon.BasisVector(V, P, C_V, C_P)
        UnP1A, UnP1B = matCon.SourceVector(V, P, C_V, C_P)
        V.Ex, V.Ex_History, V.Hy, UnP1 = matCon.solnDenecker(R, F, A, UnP1A, UnP1B, Xn, V, P, C_V, C_P, K,  Kt, De, Dh)
        toc = tim.perf_counter()
        print("Time taken with sparse matrix: ", toc-tic)
        
        VideoMaker(P, V)
        winsound.Beep(freq, duration)  
        engine = pyttsx3.init()
        engine.say('FRASER, MODEL ORDER REDUCTION METHOD FINISHED')
        engine.runAndWait()
    if P.MORmode == False:
        return V, P, C_V, C_P, Exs, Hys
    elif P.MORmode == True:
        return R, F, UnP1, V.Ex, V.Hy

plt.close('all')
tic = tim.perf_counter()
if(P.MORmode == False):
    V, P, C_V, C_P, Exs, Hys = LoopedSim(V,P,C_V, C_P, P.MORmode, loop = False, Low =P.freq_in)
elif(P.MORmode == True):
    R, F, UnP1, V.Ex, V.Hy = LoopedSim(V,P,C_V, C_P, P.MORmode, loop = False, Low =P.freq_in)
toc = tim.perf_counter()
print(toc-tic, "Time for Looped sim to run non-loop")
#print(np.max(Ex_History), "max history")

#t =np.arange(0,len(V.x1ColBe))







### EXPOSING VARS TO INSPECTOR
UpHyMat = V.UpHyMat
UpExMat = V.UpExMat
Ex = V.Ex
Hy = V.Hy
Ex_History = V.Ex_History
Hy_History = V.Hy_History
#Psi_e_History=V.Psi_e_History
#Psi_Hy_History=V.Psi_Hy_History
#Exs = V.Exs
#Hys = V.Hys
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


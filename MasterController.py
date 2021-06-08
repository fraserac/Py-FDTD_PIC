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
from tqdm import tqdm 

#from TransformHandler import genFourierTrans as gft
#import TransformHandler as tHand
#import winsound
import pyttsx3

from numba import jit
from numba import njit as nj 
from numba.experimental import jitclass as jclass
from numba import int32, float32, int64, float64, boolean, complex128
from numba.typed import Dict
from numba.types import DictType, List, string, optional
import time as tim
import timeit
#from memory_profiler import profile
import BulkTest as bt

#from varname import nameof
import sys
from jinja2 import Template as tmp, FileSystemLoader as fsl, Environment as env
import os
from datetime import datetime, date
path = os.getcwd() 
newDir = "\\Report"
my_path = path + newDir
###################################################
duration = 1000  # milliseconds
freq = 300  # Hz
sys.path[0] = os.getcwd()
import BaseFDTD11
import Environment_Setup as envDef
from Validation_Physics import VideoMaker 
 
import TransformHandler as transH
import Solver_Engine as SE
#import JuliaHandler as JH
"""
Below, specify typings of jclass members, as numba 
doesn't use dynamic typing like base python.

"""
specR=[('dict1', DictType(string, string))]
@jclass(specR)
class Reporter(object):
    def __init__(self):
        self.dict1 = {"": ""}
        
    def printer(self, item ="", name ="", show = False):
        
        
        self.dict1[name] = item 
        
        # write this to file in organised manner, module for this?
        if show == True:
            print(item)
        
        #take in some string from a print to console and save to output file with
        #
        #option to print to console 
        
    def __str__(self):
        pass
    def write_report(self):
        pass
       
        ## check file exists, if not create, if so replace. 
        #dump env stream using template to txt file called "Report.txt"
        

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
    ('gammaE', float64),
    ('omega_0E', float64),
    ('tempVarPol', float64[:]),
    ('tempTempVarPol', float64[:]),
    ('tempVarE', float64[:]),
    ('tempTempVarE', float64[:]), 
    ('tempVarHy', float64[:]),
    ('tempTempVarHy', float64[:]),
    ('tempVarJx', float64[:]),
    ('tempTempVarJx', float64[:]), 
    ('tempVarDx', float64[:]),
    ('tempTempVarDx', float64[:]), 
    ('test', int32),
    ('tempTest', float64[:]),
    ('tempTempTest', int32), 
    ('attenAmnt', int32),
    ('x1Atten', float64[:,:]),
    ('Gx3', float64[:]),
    ('alpha3', float32),
    ('Qx3', float64[:]),
    ('Pbar3', float64[:]), 
    ('nonLin3gammaE', float64),
    ('nonLin3Omega_0E', float64),
    ('chi1Stat', float64),
    ('chi3Stat', float64),
    ('JxKerr', float64[:]),
    ('JxRaman', float64[:]),
    ('Acubic', float64[:]),
    ('cubPoly', complex128[:,:]),
    ('roots', complex128[:])]

         
@jclass(specV)
class Variables(object):

    def __init__(self, Nz, timeSteps, interval, attenAmnt):
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
        self.Ex_History= np.zeros((Nz)*int(timeSteps/interval)).reshape(int(timeSteps/interval), (Nz))
        #### fix below shapes 
        self.Hy_History = np.zeros((Nz)*int(timeSteps/interval)).reshape(int(timeSteps/interval), (Nz))
        self.Jx_History = np.zeros((Nz)*int(timeSteps/interval)).reshape(int(timeSteps/interval), (Nz))
        self.polCurr_History = np.zeros((Nz)*int(timeSteps/interval)).reshape(int(timeSteps/interval), (Nz))
        self.Dx_History = np.zeros((Nz)*int(timeSteps/interval)).reshape(int(timeSteps/interval), (Nz))
        self.Psi_e_History =np.zeros((Nz)*int(timeSteps/interval)).reshape(int(timeSteps/interval), (Nz))
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
        self.plasmaFreqE =np.sqrt(((1.5)*(2*np.pi*20e9)**2))
        self.gammaE = 2*np.pi*20e9*0.1
        self.omega_0E= 2*np.pi*20e9
        self.tempVarPol =np.zeros(Nz)
        self.tempTempVarPol =np.zeros(Nz)
        self.tempVarE =np.zeros(Nz)
        self.tempTempVarE =np.zeros(Nz)
        self.tempVarHy =np.zeros(Nz)
        self.tempTempVarHy =np.zeros(Nz)
        self.tempVarJx =np.zeros(Nz)
        self.tempTempVarJx =np.zeros(Nz)
        self.tempVarDx = np.zeros(Nz)
        self.tempTempVarDx = np.zeros(Nz)
        self.test =5
        self.tempTest = np.zeros(Nz)
        self.tempTempTest= 0#
        self.attenAmnt = attenAmnt
        self.x1Atten = np.zeros(attenAmnt*timeSteps).reshape(attenAmnt, timeSteps)
        self.Gx3 = np.zeros(Nz)
        self.alpha3 = 0.7
        self.Qx3 = np.zeros(Nz)
        self.Pbar3 = np.zeros(Nz)
        self.nonLin3gammaE = 0
        self.nonLin3Omega_0E = 6e9
        self.chi1Stat = 0.69617
        self.chi3Stat = 7e-1
        self.JxKerr = np.zeros(Nz)
        self.JxRaman = np.zeros(Nz)
        self.Acubic =np.zeros(Nz)
        self.cubPoly =np.zeros(4*Nz, dtype=complex128).reshape(Nz, 4)
        self.roots =np.zeros(4, dtype=complex128)
        
        

    def __str__(self):
        return 'Contains data that will change during sim'

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
    ('delayMOR', int32), 
    ('CPMLXp', boolean),
    ('CPMLXm', boolean),
    ('TFSF', boolean),
    ('SineCont', boolean),
    ('Gaussian', boolean),
    ('Ricker', boolean),
    ('Amplitude', float32),
    ('Periods', float32),
    ('LorentzMed', boolean), 
    ('nonLinMed', boolean),
    ('FreeSpace', boolean), 
    ('epsRe', float32), 
    ('muRe', float32), 
    ('vidMake', boolean ),
    ('vidInterval', int32), 
    ('atten', boolean),
    ('julia', boolean)]


@jclass(specP)
class Params(object):
    def __init__(self,  Nz, timeSteps, eLoss, mLoss, eSelfCo, eHcompsCo, hSelfCo, hEcompsCo, x1Loc, x2Loc, materialFrontEdge, materialRearEdge, pmlWidth, nzsrc, lamMin, dz, delT, courantNo, period, Nlam, MORmode, domainSize, freq_in, delayMOR, LorentzMed = False, nonLinMed = False, SineCont = False, Gaussian = False, TFSF = False):
        
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
        self.CPMLXp = True
        self.CPMLXm = True
        self.TFSF = TFSF
        self.SineCont = SineCont
        self.Gaussian = Gaussian 
        self.Ricker = False
        self.Amplitude = 1.0
        self.Periods = 1.0
        self.LorentzMed = LorentzMed
        self.nonLinMed = nonLinMed
        self.FreeSpace = True
        self.epsRe = 1.0
        self.muRe = 1.
        self.vidMake = True
        self.vidInterval = 50
        self.atten = False
        self.julia = False

    def __str__(self):
        return 'Class containing all values that remain constant throughout a sim' 

specCP = [('kappaMax',float32),
          ('r_scale',float32),
          ('r_a_scale',float32),
          ('sigmaEMax',float64),
          ('sigmaHMax', float64),
          ('sigmaOpt',float64),
          ('alphaMax',float32),]

## kmax def most important!
@jclass(specCP)   # with PLRC-CPML alpha seems to have larger stable range
class CPML_Params(object):  # good params: KappaMax , sigmaEMax ,alphamax , nlam , 
    def __init__(self, dz):
        self.kappaMax =2 # 'Stretching co-ordinate of pml, to minimise numerical dispersion set it as 1' : DOI: 10.22190/FUACR1703229G see conclusion
        self.r_scale =4 #Within ideal bounds see Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17 (scaling power is called 'm' )
        self.r_a_scale=1
        self.sigmaEMax= 0.75*(0.8*(1)/(dz*(sci.constants.mu_0/sci.constants.epsilon_0)**0.5))#1.1*sigmaOpt # Within ideal bounds for value, : Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
        self.sigmaHMax =self.sigmaEMax#0.75*(0.8*(1)/(dz*(sci.constants.mu_0/sci.constants.epsilon_0)**0.5))#1.1*sigmaOpt # See International Journal of Computer Science and Network Security, VOL.18 No.12, December 2018, page 4 right hand side.
        self.sigmaOpt  =self.sigmaEMax#0.75*(0.8*(1)/(dz*(sci.constants.mu_0/sci.constants.epsilon_0)**0.5))
    #Optimal value of pml conductivity at far end of pml: DOI: 10.22190/FUACR1703229G see equation 13
        self.alphaMax=0.1# with bounds of ideal cpml alpha max, complex frequency shift parameter, Journal of ELECTRICAL ENGINEERING, VOL 68 (2017), NO1, 47–53, see paragraph under eqn. 17
    
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


    


def Controller(V, P, C_V, C_P):
    probeReadFinishBe = int(P.timeSteps*0.7)
    probeReadStartAf = int(P.timeSteps*0.05)  # More rigorous on/off criteria
    V.x1ColBe = np.zeros(P.timeSteps)
    V.x1ColAf = np.zeros(P.timeSteps)


    if P.LorentzMed:
         V.Ex, V.Hy, Exs, Hys,  C_V.psi_Ex,C_V.psi_Hy, V.x1ColBe, V.x1ColAf = SE.IntegratorLinLor1D(V, P, C_V, C_P, probeReadFinishBe, probeReadStartAf)
    elif P.FreeSpace:
        V.Ex, V.Hy, Exs, Hys, C_V.psi_Ex,C_V.psi_Hy, V.x1ColBe, V.x1ColAf = SE.IntegratorFreeSpace1D(V, P, C_V, C_P, probeReadFinishBe, probeReadStartAf)
    elif P.nonLinMed:
        V.Ex, V.Hy, Exs, Hys,  C_V.psi_Ex,C_V.psi_Hy, V.x1ColBe,  V.x1ColAf = SE.IntegratorNL1D(V,P,C_V,C_P,probeReadFinishBe, probeReadStartAf)



    
   # breakpoint()
   # print("run ", i, " took this many seconds: ", tockee-tickee)
        
    return V, P, C_V, C_P, Exs, Hys


def results(V, P, C_V, C_P, time_Vec,  RefCo = False, FFT = False, AnalRefCo = False, attenRead = False):
    
    valM = np.zeros(len(V.x1Atten))
    attenCo = np.zeros(len(valM))
    if attenRead ==True:
        sig_pow, sample_freq, val= transH.RefTester(V,P, V.x1ColBe, 1)   
        for i in range(len(V.x1Atten)):
            valM[i] = transH.RefTester(V,P, V.x1Atten[i], 1, mul = True)

            attenCo[i] = valM[i]/val
        #breakpoint()
        return attenCo
    
    elif RefCo == True:
    
        #transm, sig_fft1, sig_fft2, sample_freq, timePadded = 
        sig_pow, sample_freq, val= transH.RefTester(V,P, V.x1ColBe, 1)
        sig2_pow, sample_freq, val2= transH.RefTester(V,P, V.x1ColAf, 1)
       
        #breakpoint()
        reflectCo = val2/val
        #plt.plot(sample_freq,sig_fft) 
        
       # plt.plot(sample_freq,abs(sig_fft2))
        #breakpoint()
        #reflectCo = transH.ReflectionCalc(P, V, sample_freq, sig_pow, sig2_pow)
       
        
        return reflectCo
    
    elif AnalRefCo==True:
        analReflectCo= BaseFDTD11.AnalyticalReflectionE(V,P)
        return analReflectCo
    return "results ran to end"
    

def plotter(xAxisData,  yAxisData1, yAxisData2, xAxisLim = 2, myAxisLim = 0, yAxisLim = 1, xAxisLabel = " ", yAxisLabel = " ", legend = " ", title= " "):
   
    # Add block data option for more than 2 data sets
    fig, ax = plt.subplots()
    ax.clear()
    ax.plot(xAxisData, yAxisData1)
    ax.plot(xAxisData, yAxisData2)
    ax.set_title(title) 
    ax.legend(legend)
    ax.set_ylim(myAxisLim, yAxisLim)
    ax.xlabel = xAxisLabel
    ax.ylabel = yAxisLabel
    #ax.set_xlim(-xAxisLim, xAxisLim)
    #titles and neatly formatting 
    #breakpoint()
    
  
InputSweepSwitch = {"Input frequency sweep": results,
                "test function" : results } 
  

#@jit
def LoopedSim(Rep, V,P,C_V, C_P, MORmode, domainSize, lowLimTim, highLimTim, stringparamSweep = "Input frequency sweep", loop =False,  Low = 3e9, Interval = 1e8, RefCoBool = True):
    #print("loop = ", loop)
   
    if loop == True:
        points = 20
        dataRange = np.arange(Low, points*Interval+Low, Interval) 
        #print(len(dataRange))
        freqDomYAxisRef =np.zeros(points)
        timeDomYAxis = np.zeros(points)       
        freqDomYAxis2AnalRef=  np.zeros(points)  
       # Exs, Hys = BaseFDTD11.SmoothTurnOn(V,P)
       
        if stringparamSweep == "Input frequency sweep":
            for loop in range(points): 
                
                noOfEnvOuts =20
                setupReturn = []*noOfEnvOuts
                setupReturn =envDef.envSetup(P.freq_in, domainSize,lowLimTim,highLimTim, VExists = True, V = V, P= P)   # this function returns a list with all evaluated model parameters
                P= Params(*setupReturn, P.MORmode, domainSize, P.freq_in, 20, LorentzMed = P.LorentzMed, SineCont = P.SineCont, Gaussian = P.Gaussian, TFSF = P.TFSF) #be careful with tuple, check ordering of parameters 
                V=Variables(P.Nz, P.timeSteps, P.vidInterval, 1)  # Make last two args defaults in class constr
                C_P = CPML_Params(P.dz)
                C_V = CPML_Variables(P.Nz, P.timeSteps)
                V, P, C_V, C_P, Exs, Hys= Controller(V, P, C_V, C_P)
                print("Max vals: Ex, Jx, Hy: ", np.max(V.x1ColBe),np.max(V.x1Jx),np.max(V.x1Hy))
                
                t =np.arange(0,len(V.x1ColBe))*P.delT
                #breakpoint()
                freqDomYAxisRef[loop] = results(V, P, C_V, C_P, t, RefCo=True) # One refco y point
                freqDomYAxis2AnalRef[loop] =results(V, P, C_V, C_P, t, AnalRefCo = True)
                P.freq_in = P.freq_in+Interval
                print("LOOP OF SIM, PARAMETER: ", loop)
                print(freqDomYAxisRef[loop], freqDomYAxis2AnalRef[loop], " measured vs analytical ref")
                print("freq_in : ", P.freq_in)
                plt.close('all')
                
                #VideoMaker(P, V)
                #sys.exit()
                #print(freqDomYAxis2AnalRef)
                #print(freqDomYAxisRef, freqDomYAxis2AnalRef, "Y STUFF")
        plotter(dataRange, yAxisLim = 1, yAxisData1 =freqDomYAxisRef, yAxisData2 =freqDomYAxis2AnalRef, legend = ["Measured","Analytical"], title = "Analytical vs Measured reflection")  
        
    elif loop == False:
            points =1
            freqDomYAxisRef =np.zeros(points)
            freqDomYAxis2AnalRef= np.zeros(points) 
            freqDomAttenAmp = np.zeros(points*V.attenAmnt).reshape(points, V.attenAmnt)
            tic =tim.perf_counter()
            #source manager comes in here? Source manager can call smoothturnon when necessary
            
            
            toc = tim.perf_counter()
            Rep.printer(str(toc-tic), "matSetupTime")
            tic =tim.perf_counter()
            V, P, C_V, C_P, Exs, Hys= Controller(V, P, C_V, C_P)
            print("Max vals: Ex, Jx, Hy: ", np.max(V.x1ColBe),np.max(V.x1Jx),np.max(V.x1Hy))
            #breakpoint()
            t =np.arange(0,len(V.x1ColBe))*P.delT
            if P.FreeSpace:
                print("Free space so no probe reviewed")
            elif P.LorentzMed:
                freqDomYAxisRef[0] = results(V, P, C_V, C_P, t,  RefCo=True) # One refco y point
                freqDomYAxis2AnalRef[0] =results(V, P, C_V, C_P, t, AnalRefCo = True)
            #freqDomAttenAmp[0] =results(V, P, C_V, C_P, t, attenRead = True)   # if working duplicate in loop = True
                print(freqDomYAxisRef[0], freqDomYAxis2AnalRef[0], " measured vs analytical ref")#
            #plot atten read
           #spatial = np.arange(P.Nz+1)*P.dz
            
            
            VideoMaker(P, V)
           # winsound.Beep(freq, duration)
            #ngine = pyttsx3.init()
            #engine.say('beep')
            #engine.runAndWait()


            #plt.close('all')
    return V, P, C_V, C_P, Exs, Hys
    


##########################   START OF CODE WHEN MASTERCONTROLLER IS RUN



#######################################
###############   VARIABLES AND CALLS FOR PROGRAM INITIATION
#########


#@jit(parallel =True)
def __Main__():
    plt.close('all')
    startedAt = date.today()
    startedTime = datetime.now()  # move to report 
    print("Code started at: ", startedTime)
    loopMode =False
    MORmode = False
    domainSize=0.4   #metres
    lowLimTim = 2000
    highLimTim = 3000 # More rigorous via speed of wave and dist 
    freq_in = 6e9
    delayMOR =20
    LorMed = False
    nonLinMed = True
    noOfEnvOuts = 20
    setupReturn = []*noOfEnvOuts
    setupReturn =envDef.envSetup(freq_in, domainSize,lowLimTim,highLimTim, nonLinMed =nonLinMed, LorMed = LorMed)   # this function returns a list with all evaluated model parameters
    P= Params(*setupReturn, MORmode, domainSize, freq_in, delayMOR) #be careful with tuple, check ordering of parameters 
    P.vidInterval = 50
    attenAmount = 10
    V=Variables(P.Nz, P.timeSteps, P.vidInterval, attenAmount)
    C_P = CPML_Params(P.dz)
    C_V = CPML_Variables(P.Nz, P.timeSteps)
    
    Rep = Reporter() 
    P.atten = True 
    P.epsRe =1# 0.917**2
    P.CPMLXp =True    ### CHANGE TO Z FOR PROP DIRECTION
    P.CPMLXm =True
    P.TFSF = True
    P.Gaussian =False
    P.SineCont = True   # Set up smoothturn on for multiple repeats
    P.Periods = 5
    P.LorentzMed=LorMed # MAKE SURE EPSRE IS 1
    P.nonLinMed = nonLinMed
    ticK = tim.perf_counter()
    P.julia = False

    if P.nonLinMed == True:
        P.FreeSpace = False
        P.LorentzMed = False   # Redundant code because first batch handles all 
    if P.LorentzMed == True:
        P.FreeSpace = False
        P.nonLinMed = False
    
    
  # Use previous results of sweep to speed up next run? 
    V, P, C_V, C_P, Exs, Hys = LoopedSim(Rep,V,P,C_V, C_P, P.MORmode, domainSize, lowLimTim, highLimTim, loop = loopMode, Low =P.freq_in, Interval = 5e8)
    tocK = tim.perf_counter()
    Rep.printer(str(tocK-ticK), "LoopMainTime")
    templateEnv = env(loader = fsl('.'))
    #dict1 = Rep.dict1
    template = templateEnv.get_template("Template_Report.txt")
    output = template.render(dict1= Rep.dict1)
    #Rep.write_report(output)
    with open(my_path +"\\" + "Report_File.txt", "w+") as reader:
        reader.write(output)
    
    #print(tocK-ticK, "Time for Looped sim to run non-loop")
    return V, P, C_P, C_V, Rep

V, P, C_P, C_V, Rep = __Main__()
    

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


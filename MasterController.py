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



class Variables(object):
    def __init__(self, UpHyMat, UpExMat, Ex, Hy, Ex_History, Hy_History,Psi_Ex_History, Psi_Hy_History, Hys, Exs, x1ColBe, x1ColAf, epsilon, mu, UpExHcompsCo, UpExSelf, UpHyEcompsCo, UpHySelf, Dx, My):
        self.UpHyMat = UpHyMat
        self.UpExMat = UpExMat
        self.UpExHcompsCo = UpExHcompsCo
        self.UpExSelf = UpExSelf
        self.UpHySelf = UpHySelf
        self.UpHyEcompsCo = UpHyEcompsCo 
        self.Ex = Ex
        self.Hy = Hy
        self.Dx = Dx
       # self.By = By
        self.Ex_History = Ex_History
        self.Hy_History = Hy_History
        self.Psi_Ex_History= Psi_Ex_History
        self.Psi_Hy_History = Psi_Hy_History
        self.Exs = Exs
        self.Hys = Hys
        self.x1ColBe = x1ColBe
        self.x1ColAf = x1ColAf
        self.epsilon = epsilon
        self.mu = mu
        self.inputVec = []
        self.outputPlots = []
        self.polarisationCurr = np.zeros(P.Nz+1)
        self.plasmaFreqE = 2*np.pi*1.1027e9
        self.gammaE = 1e8
        self.omega_0E = 2*np.pi*0.1591e9
        self.tempVarPol =[]
        self.tempTempVarPol =[]
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
    
    def __init__(self, epsRe, muRe, freq_in, lMin, nlm, dz, delT, courantNo, domainSize,  matRear, matFront, gridNo, timeSteps, x1Loc, nzsrc, period, eLoss, eSelfCo, eHcompsCo, mLoss, hEcompsCo, hSelfCo, pmlWidth, x2Loc ):
        self.epsRe = epsRe
        self.muRe = muRe
        self.freq_in = freq_in
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
    
    
    def __init__(self, kappaMax, sigmaEMax, sigmaHMax, sigmaOpt, alphaMax, r_scale, r_a_scale ):
        self.kappaMax=kappaMax
        self.sigmaEMax=sigmaEMax
        self.sigmaHMax=sigmaHMax
        self.sigmaOpt=sigmaOpt
        self.alphaMax=alphaMax
        self.r_scale=r_scale
        self.r_a_scale=r_a_scale
    def __repr__(self):
        return (f'{self.__class__.__name__}')
    
    def __str__(self):
        return 'Class containing all CPML values that remain constant throughout a sim' 
    
    
class CPML_Variables(object):
    
    
    def __init__(self, kappa_Ex, kappa_Hy, psi_Ex, psi_Hy, alpha_Ex, alpha_Hy, sigma_Ex, sigma_Hy, beX, bmY, ceX, cmY, Ca, Cb, Cc, C1, C2, C3, eLoss_CPML, mLoss_CPML, den_Hydz, den_Exdz ):
        self.kappa_Ex=kappa_Ex
        self.kappa_Hy=kappa_Hy
        self.psi_Ex=psi_Ex
        self.psi_Hy=psi_Hy
        self.alpha_Ex= alpha_Ex
        self.alpha_Hy=alpha_Hy
        self.sigma_Ex=sigma_Ex
        self.sigma_Hy=sigma_Hy
        self.beX=beX
        self.bmY=bmY
        self.ceX=ceX
        self.cmY=cmY
        self.Ca=Ca
        self.Cb= Cb
        self.Cc=Cc
        self.C1=C1
        self.C2=C2
        self.C3=C3
        self.eLoss_CPML=eLoss_CPML
        self.mLoss_CPML = mLoss_CPML
        self.den_Hydz = den_Hydz
        self.den_Exdz = den_Exdz
    def __repr__(self):
        return (f'{self.__class__.__name__}')
    
    def __str__(self):
        return 'Class containing all CPML values that vary throughout a sim' 
"""
#Next, loop the FDTD over the time domain range (in integer steps, specific time would be delT*timeStep include this on plots later?)

"""

#FUNCTION THAT LOADS IN MATERIAL DEF, CAN BE PASSED IN AS A FIRST CLASS FUNCTION, RETURNS ALL
#PARAMETERS.
def Controller(V, P, C_V, C_P):  #Needs dot syntax
    V.tempTempVarPol, V.polarisationCurr, V.Ex, V.Dx, V.Ex_History, V.Hy, V.Hy_History, V.Psi_Ex_History, V.Psi_Hy_History,V.Exs, V.Hys = BaseFDTD.FieldInit(V,P)
    V.Exs, V.Hys = BaseFDTD.SmoothTurnOn(V,P)
    
    V.UpHyMat, V.UpExMat = BaseFDTD.EmptySpaceCalc(V,P)   #RENAME EMPTY SPACE CALC
    C_V = BaseFDTD.CPML_FieldInit(V,P, C_V, C_P)
    V.x1ColBe = [[]]*(P.timeSteps)
    V.x1ColAf = [[]]*(P.timeSteps)
    for counts in range(0,P.timeSteps):   ### for media one transmission
       #V.Hy[P.Nz-1] = HyBC(V,P)
       #print(counts ,'counts 1')
       
       """
       C_V.sigma_Ex, C_V.sigma_Hy, C_V.alpha_Ex,  C_V.alpha_Hy, C_V.kappa_Ex, C_V.kappa_Hy= BaseFDTD.CPML_ScalingCalc(V, P, C_V, C_P)
       C_V.beX, C_V.ceX = BaseFDTD.CPML_Ex_RC_Define(V, P, C_V, C_P)
       C_V.bmY, C_V.cmY = BaseFDTD.CPML_HY_RC_Define(V, P, C_V, C_P)
       C_V.eLoss_CPML, C_V.Ca, C_V.Cb, C_V.Cc = BaseFDTD.CPML_Ex_Update_Coef(V,P, C_V, C_P)
       C_V.mLoss_CPML, C_V.C1, C_V.C2, C_V.C3 = BaseFDTD.CPML_Hy_Update_Coef(V,P, C_V, C_P)
       C_V.den_Exdz, C_V.den_Hydz = BaseFDTD.denominators(V, P, C_V, C_P)
       """
       #V.polarisationCurr, V.tempVarPol = BaseFDTD.ADE_PolarisationCurrent_Ex(V, P, C_V, C_P)
       #V.Hy = BaseFDTD.CPML_HyUpdate(V,P, C_V, C_P)
       V.Hy = BaseFDTD.ADE_HyUpdate(V, P, C_V, C_P)
      # V.Ex = BaseFDTD.ADE_ExCreate(V, P, C_V, C_P)
       
       V.Hy[P.nzsrc-1] = BaseFDTD.HyTfSfCorr(V,P, counts)
       
      # C_V.psi_Hy, V.Hy  = BaseFDTD.CPML_Psi_m_Update(V,P, C_V, C_P)
       
       #V.Dx = BaseFDTD.ADE_DxUpdate(V, P, C_V, C_P)
       
       V.Ex[P.nzsrc] = BaseFDTD.ExTfSfCorr(V,P, counts)
       
       
       V.Ex =BaseFDTD.ADE_ExUpdate(V, P, C_V, C_P)
       #V.Ex = BaseFDTD.CPML_ExUpdate(V,P, C_V, C_P)
       
       
       
       #C_V.psi_Ex, V.Ex  = BaseFDTD.CPML_Psi_e_Update(V,P, C_V, C_P)
       
       
     
      
      
       
       V.Ex_History[counts] = np.insert(V.Ex_History[counts], 0, V.Ex)
       #V.Psi_Ex_History[counts] = np.insert(V.Psi_Ex_History[counts], 0, C_V.psi_Ex)
       #V.Psi_Hy_History[counts] = np.insert(V.Psi_Hy_History[counts], 0, C_V.psi_Hy)
       
       
     
       
      
      
       if counts <= P.timeSteps-1:
           V.x1ColBe[counts] = V.Ex_History[counts][P.x1Loc] ##  X1 SHOULD BE ONE POINT! SPECIFY WITH E HISTORY ADDITIONAL INDEX.
       
       V.Hy_History[counts] = np.insert(V.Hy_History[counts], 0, V.Hy)
       #breakpoint()
    
     
    
    
    V.UpHyMat, V.UpExMat = BaseFDTD.EmptySpaceCalc(V,P)  
    V.tempTempVarPol, V.polarisationCurr, V.Ex,  V.Dx, V.Ex_History, V.Hy, V.Hy_History, V.Psi_Ex_History, V.Psi_Hy_History, V.Exs, V.Hys = BaseFDTD.FieldInit(V,P)
    V.Exs, V.Hys = BaseFDTD.SmoothTurnOn(V,P)
    #V.UpHyMat, V.UpExMat = BaseFDTD.UpdateCoef(V,P)
    #C_V = BaseFDTD.CPML_FieldInit(V,P, C_V, C_P)
    for count in range(P.timeSteps):   ### for media one transmission
    
       #print(count ,'count 2')
       """
       C_V.sigma_Ex, C_V.sigma_Hy, C_V.alpha_Ex,  C_V.alpha_Hy, C_V.kappa_Ex, C_V.kappa_Hy= BaseFDTD.CPML_ScalingCalc(V, P, C_V, C_P)
       C_V.beX, C_V.ceX = BaseFDTD.CPML_Ex_RC_Define(V, P, C_V, C_P)
       C_V.bmY, C_V.cmY = BaseFDTD.CPML_HY_RC_Define(V, P, C_V, C_P)
       C_V.eLoss_CPML, C_V.Ca, C_V.Cb, C_V.Cc = BaseFDTD.CPML_Ex_Update_Coef(V,P, C_V, C_P)
       C_V.mLoss_CPML, C_V.C1, C_V.C2, C_V.C3 = BaseFDTD.CPML_Hy_Update_Coef(V,P, C_V, C_P)
       C_V.den_Exdz, C_V.den_Hydz = BaseFDTD.denominators(V, P, C_V, C_P)
       """
       
       
       
       V.tempTempVarPol, V.tempVarPol = BaseFDTD.ADE_TempPolCurr(V,P)
       V.polarisationCurr = BaseFDTD.ADE_PolarisationCurrent_Ex(V, P, C_V, C_P)
       #V.Hy = BaseFDTD.CPML_HyUpdate(V,P, C_V, C_P)
       V.Hy[P.nzsrc-1] = BaseFDTD.HyTfSfCorr(V,P, count) 
       V.Ex =BaseFDTD.ADE_ExUpdate(V, P, C_V, C_P)
       V.Ex[P.nzsrc] = BaseFDTD.ExTfSfCorr(V,P, count)
       
       V.Dx, V.Ex = BaseFDTD.ADE_DxUpdate(V, P, C_V, C_P)
       V.Hy = BaseFDTD.ADE_HyUpdate(V, P, C_V, C_P)
       
       #V.Ex = BaseFDTD.CPML_ExUpdate(V,P, C_V, C_P)
       
       
       
       
       
       #V.Ex = BaseFDTD.ADE_ExCreate(V, P, C_V, C_P)
       
      
       
       
      # V.Dx = BaseFDTD.ADE_DxUpdate(V, P, C_V, C_P)
       #V.Ex = BaseFDTD.ADE_ExCreate(V, P, C_V, C_P)
       
       
       
       #C_V.psi_Ex, V.Ex  = BaseFDTD.CPML_Psi_e_Update(V,P, C_V, C_P)
       
       #V.Hy[P.Nz-1]= BaseFDTD.CPML_PMC(V,P,C_V, C_P)
       
       V.Ex_History[count] = np.insert(V.Ex_History[count], 0, V.Ex)
      # V.Psi_Ex_History[count] = np.insert(V.Psi_Ex_History[count], 0, C_V.psi_Ex)
       #V.Psi_Hy_History[count] = np.insert(V.Psi_Hy_History[count], 0, C_V.psi_Hy)
       
       
      #MORE ELEGANT TESTING SOLUTION
           
       
      
       if count <= (P.timeSteps-1):
            V.x1ColAf[count] = V.Ex_History[count][P.x2Loc] ##  X1 SHOULD BE ONE POINT! SPECIFY WITH E HISTORY ADDITIONAL INDEX.
      
       V.Hy_History[count] = np.insert(V.Hy_History[count], 0, V.Hy)
     
    #FFT x1ColBe and x1ColAf? 
   
    
# should have constant val of transmission over all freq range of source, will need harmonic source?   

    return V, P, C_V, C_P

P = Params(matDef.epsRe, matDef.muRe, matDef.freq_in, matDef.lamMin, matDef.Nlam, matDef.dz, matDef.delT, matDef.courantNo, matDef.domainSize, matDef.MaterialRearEdge, matDef.MaterialFrontEdge, matDef.Nz, matDef.timeSteps, matDef.x1Loc, matDef.nzsrc, matDef.period, matDef.eLoss, matDef.eSelfCo, matDef.eHcompsCo, matDef.mLoss, matDef.hEcompsCo, matDef.hSelfCo, matDef.pmlWidth, matDef.x2Loc )    
V = Variables(matDef.UpHyMat, matDef.UpExMat, matDef.Ex, matDef.Hy, matDef.Ex_History, matDef.Hy_History, matDef.Psi_Ex_History, matDef.Psi_Hy_History, matDef.Hys, matDef.Exs, matDef.x1ColBe, matDef.x1ColAf, matDef.epsilon, matDef.mu, matDef.UpExHcompsCo, matDef.UpExSelf, matDef.UpHyEcompsCo, matDef.UpHySelf, matDef.Dx, matDef.My)
C_P =  CPML_Params(matDef.kappaMax, matDef.sigmaEMax, matDef.sigmaHMax, matDef.sigmaOpt, matDef.alphaMax, matDef.r_scale, matDef.r_a_scale)
C_V = CPML_Variables(matDef.kappa_Ex, matDef.kappa_Hy, matDef.psi_Ex, matDef.psi_Hy, matDef.alpha_Ex, matDef.alpha_Hy, matDef.sigma_Ex, matDef.sigma_Hy,matDef.beX, matDef.bmY, matDef.ceX, matDef.cmY, matDef.Ca, matDef.Cb, matDef.Cc, matDef.C1, matDef.C2, matDef.C3, matDef.eLoss_CPML, matDef.mLoss_CPML, matDef.den_Hydz, matDef.den_Exdz )


#V, P, C_V, C_P=LoopSim(V,P, C_V, C_P, 0, 1e10 )

#loopNo might be unnecessary
def results(V, P, C_V, C_P, time_Vec, loopNo = 0, RefCo = False, FFT = False):
    if RefCo == True:
        transm, sig_fft1, sig_fft2, sample_freq = FourierTrans(P, V, V.x1ColBe, V.x1ColAf, time_Vec, P.delT)
        reflectCo = ReflectionCalc(P, V, sample_freq, sig_fft1, sig_fft2)
    return reflectCo# call fourier trans func then call reflectionCalc func and plot result against input   
   # IF FFT 
    #SHOW FFT 
    pass

def plotter(xAxisData, yAxisData, yAxisLim = 1, xAxisLabel = " ", yAxisLabel = " ", legend = " ", title= " "):
    fig, ax = plt.subplots()
    ax.clear()
    ax.plot(xAxisData, yAxisData)
    ax.set_ylim(-yAxisLim, yAxisLim)    
    pass
  
InputSweepSwitch = {"Input frequency sweep": results,
                "test function" : results }   # when calling results, feed something into it

def LoopedSim(V,P,C_V, C_P, stringparamSweep = "Input frequency sweep", loop =False,  Low = 1e9, Interval = 1e8, RefCoBool = True):
    print("loop = ", loop)
    if loop == True:
        points = 10
        dataRange =[]  ##m maybe remove
        dataRange = np.arange(Low, points*Interval+Low, Interval)
       # differentParams for selector, will set blank if 
        freqDomYAxisRef =np.zeros(points)
        timeDomYAxis = np.zeros(points)       
       
        if stringparamSweep == "Input frequency sweep":
            for loop in range(points): 
                #LOOP OF SIM, NEED TO SET UP BEFORE LOOP STARTS
                matDef.matSetup(V,P, dataRange[loop]) #, should return something? frequency dependence of environment, maybe tuples?
                V, P, C_V, C_P= Controller(V, P, C_V, C_P)
                t =np.arange(0,len(V.x1ColBe))
                
                freqDomYAxisRef[loop] = results(V, P, C_V, C_P, t, loop, True) # One refco y point
                print(freqDomYAxisRef)
        #plotter( relevant info) 
        plotter(dataRange, freqDomYAxisRef)  
        
        # FUNC ABOVE NEEDS TO CALC 
    elif loop == False:
        matDef.matSetup(V,P, newFreq_in= 7.5e8)
        V, P, C_V, C_P= Controller(V, P, C_V, C_P)
        xAxis = np.zeros(P.Nz+1)
        for i in range(0, P.Nz):
            xAxis[i] = i
        plotter(xAxis, V.Ex_History[250], yAxisLim =2)  
        VideoMaker(P, V)
        
    return V, P, C_V, C_P




V, P, C_V, C_P = LoopedSim(V,P,C_V, C_P, loop = False)

#V, P, C_V, C_P= Controller(V, P, C_V, C_P)

#Now we prepare to make the video including I/O stuff like setting up a new directory in the current working directory and 

#deleting the old directory from previous run and overwriting.

#VideoMaker(P, V)
#reflectionCalc(P,V)

t =np.arange(0,len(V.x1ColBe))
#transWithExp, sig1Freq, sig2Freq, sample_freq = FourierTrans(P, V, V.x1ColBe, V.x1ColAf,  t, P.delT)
####variable exposes

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


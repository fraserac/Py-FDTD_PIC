# -*- coding: utf-8 -*-
"""
@author: Fraser
This code is called by mastercontroller and will set up simulation domain based 
off of constructor arguments. It also has defaults. 




BE CAREFUL WITH MATERIAL WIDTH, CONTROLLED BY DOMAIN SIZE 
"""
import numpy as np
import scipy.constants as sci
import sys
import math as ma
#from MasterController import *




def envSetup(newFreq_in, domainSize, minim=400, maxim=600):# domain size and freq 
   # P.epsRe =1
    #P.epsIm = 0
    #P.muRe = 1
    #P.muIm = 0
    freq_in = newFreq_in
    c0 = 299792458.0
    lamMin = (c0/freq_in)
    
    #print("LamMin ", P.lamMin)
    Nlam =40#np.floor(20*np.sqrt(P.epsRe*P.muRe))
    dz =lamMin/Nlam  
    #P.courantNo = 1   # LOOK INTO 2D VERSION
    delT =  dz/(c0)#0.95/(P.c0*np.sqrt(1/(P.dz**2)))
   # decimalPlaces = 11
   # multiplier = 10 ** decimalPlaces
    #P.delT = ma.floor(P.delT* multiplier) / multiplier
    #CharImp =np.sqrt(sci.mu_0)/np.sqrt(sci.epsilon_0)
    period = 1/freq_in
    courantNo = (c0*delT)/dz
    if courantNo > 1.01 or courantNo <0:
        print(courantNo, "courantNo is unstable")
        sys.exit()
    pmlWidth =150 #+int((30*P.freq_in)/1e9)
    #print('pmlWidth = ' , P.pmlWidth)
    if (pmlWidth >= 500):
        print('pmlWidth too big', pmlWidth)
        sys.exit()
    # +int(5*P.freq_in)
    print(domainSize, 'domainSize')
    dimen =1
    nonInt =  False

    Nz = domainSize +2*dimen*pmlWidth   #Grid size

    for N in range(minim,maxim):
        
        check = (freq_in*N)/(1/delT)
        #print(check)
        if int(check)-check ==0:
            timeSteps = N 
            print("middle")
            break
        elif N == maxim-1:
           print('Could not find timestep that allowed freq_in to fall on an integer frequency bin index with range provided.') 
           #sys.exit()
           nonInt = True
           
       
    if(nonInt == True):       
        checkNear = (freq_in*minim)/(1/delT)
        N =0
        for N in range(minim, maxim):
            dummyCheck = (freq_in*N)/(1/delT)
            if (int(dummyCheck)-dummyCheck  < int(checkNear)-checkNear): # is closer to freq bin 
                #print(checkNear, dummyCheck, "checkNear, dummyCheck")
                checkNear = dummyCheck
                
                #print("dummy")
               
        
    timeSteps = N
            
    
    
    print('timesteps: ', timeSteps)
    
    if(timeSteps >= 10000):
        print('timeSteps too large')
        sys.exit()
    t=np.arange(0, timeSteps, 1)*(delT)  # FOR VERIFICATION PLOTTING, EVALUATE IN CLASS
    
    nzsrcFromPml = 200
    if nzsrcFromPml >= domainSize/2:
        print(nzsrcFromPml, 'src is too far into domain')
        sys.exit()
        
    nzsrc = nzsrcFromPml + pmlWidth# Position of source 
    x2LocBehindSrc = 10
    if(nzsrc -x2LocBehindSrc <= pmlWidth):
        print('The probe for fft is in the PML region')
        sys.exit()   
    
    MaterialDistFromPml = 300
    materialFrontEdge = MaterialDistFromPml + pmlWidth   # Discrete tile where material begins (array index)
    materialRearEdge =  Nz-pmlWidth-1
    MaterialWidth = materialRearEdge-materialFrontEdge
   
    if MaterialWidth < 10:
        print(MaterialWidth, "width is too small or negative")
        sys.exit()
    
    x1Loc = nzsrc+int((materialFrontEdge-nzsrc)/2)
    x2Loc = nzsrc - (nzsrcFromPml+x2LocBehindSrc)  +50
    
    eLoss =0   # sigma e* delT/2*epsilon
    mLoss = 0
    eSelfCo = (1-eLoss)/(1+eLoss)#
    eHcompsCo = 1/(1+eLoss)
    hSelfCo = (1-mLoss)/(1+mLoss)
    hEcompsCo = 1/(1+mLoss)
    
    return Nz, timeSteps, eLoss, mLoss, eSelfCo, eHcompsCo, hSelfCo, hEcompsCo, x1Loc, x2Loc, materialFrontEdge, materialRearEdge, pmlWidth, nzsrc, lamMin, dz, delT, courantNo, period 
    """
    V.x1ColBe=[[]] 
    V.x1ColAf=[[]]
    V.UpHySelf= np.ones(P.Nz)
    V.UpHyEcompsCo = np.ones(P.Nz)
    V.UpExSelf = np.ones(P.Nz)
    V.UpExHcompsCo =np.ones(P.Nz)
    V.UpExMat =np.zeros(P.Nz)
    V.UpHyMat = np.zeros(P.Nz)
    V.Ex =np.zeros(P.timeSteps)
    V.Hy=np.zeros(P.timeSteps)
    V.Ex_History= [[]]
    V.Hy_History= [[]]
    V.Psi_Ex_History= [[]]
    V.Psi_Hy_History= [[]]
    V.Hys = []
    V.Exs = []
    
    V.epsilon = np.ones(P.Nz)
    V.mu = np.ones(P.Nz)
    
    V.Dx = [[]]
    V.My = [[]]
    C_P.sigmaEMax= 1.1*(0.8*(C_P.r_scale+1)/(P.dz*(P.permea_0/P.permit_0)**0.5))
    C_V.kappaEx = np.zeros(P.Nz)
    C_V.kappaHy = np.zeros(P.Nz)
    C_V.psi_Ex =  np.zeros(P.Nz)
    C_V.psi_Hy =  np.zeros(P.Nz)
    C_V.alpha_Ex = np.zeros(P.Nz)
    C_V.alpha_Hy = np.zeros(P.Nz)
    C_V.sigma_Ex =np.zeros(P.Nz)   # specific spatial value of conductivity 
    C_V.sigma_Hy = np.zeros(P.Nz)
    C_V.beX =np.zeros(P.Nz)
    C_V.bmY =np.zeros(P.Nz)#np.exp(-(sigmaHy/(permea_0*kappa_Hy) + alpha_Hy/permea_0 )*delT)
    C_V.ceX = np.zeros(P.Nz)
    C_V.cmY = np.zeros(P.Nz)
    C_V.Ca = np.zeros(P.Nz)
    C_V.Cb = np.zeros(P.Nz)
    C_V.Cc = np.zeros(P.Nz)
    C_V.C1 = np.zeros(P.Nz)
    C_V.C2 = np.zeros(P.Nz)
    C_V.C3 = np.zeros(P.Nz)
    
    C_V.eLoss_CPML =[]   # sigma e* delT/2*epsilon
    C_V.mLoss_CPML = []
    C_V.den_Hydz = []
    C_V.den_Exdz = [] 

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


        self.kappaMax=kappaMax
        self.sigmaEMax=sigmaEMax
        self.sigmaHMax=sigmaHMax
        self.sigmaOpt=sigmaOpt
        self.alphaMax=alphaMax
        self.r_scale=r_scale
        self.r_a_scale=r_a_scale

"""
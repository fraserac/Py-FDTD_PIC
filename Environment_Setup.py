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



def envSetup(newFreq_in, domainSize, minim=400, maxim=600, VExists =False, V =[], P=[], nonLinMed = False, LorMed = False):# domain size and freq 
    pi = np.pi
    
    #CHOSEN GAMMA HAS 1 IN FRONT NOT 2!!
    if (VExists): 
        epsNum = (V.plasmaFreqE*V.plasmaFreqE)
        epsDom = (V.omega_0E*V.omega_0E-(2*pi*P.freq_in*2*pi*P.freq_in) + 1j*V.gammaE*2*pi*P.freq_in)
        eps0 = P.permit_0   
        epsilon = 1 + epsNum/epsDom
     
    #P.epsIm = 0
    #P.muRe = 1
    #P.muIm = 0
    freq_in = newFreq_in
    #print(freq_in, "freq")
    c0 = 299792458.0
    lamMin = (c0/freq_in)
    
    #print("LamMin ", P.lamMin)
    if VExists: 
        Nlam =int(60*(np.real(epsilon))**1.05)
        print("in VExists env setup, max eps = ", np.max(np.real(epsilon)))
        if nonLinMed:
            Nlam = int(200*(np.real(epsilon))**1.05)
    elif VExists ==False:
        Nlam = 400
        if nonLinMed: 
            Nlam = 350
    dz =lamMin/Nlam
    #P.courantNo = 1   # LOOK INTO 2D VERSION
    delT = (dz/c0)*0.95#/(P.c0*np.sqrt(1/(P.dz**2)))  # COUPLED TO DZ THROUGH COURANT CONSTRAINT! 
    if nonLinMed:
        delT *=1
    print("delt ->", delT)
   # decimalPlaces =11
   # multiplier = 10 **decimalPlaces
    #P.delT = ma.floor(P.delT* multiplier) / multiplier
    #CharImp =np.sqrt(sci.mu_0)/np.sqrt(sci.epsilon_0)
    period = 1/freq_in
    courantNo = (c0*delT)/dz
    if courantNo > 3 or courantNo <0:
        print(courantNo, "courantNo is unstable")
        sys.exit()
    pmlWidth =6*int(lamMin/dz) #+int((30*P.freq_in)/1e9)
    #print('pmlWidth = ' , P.pmlWidth)
    if (pmlWidth >= 12000):
        print('pmlWidth too big', pmlWidth)
        sys.exit()
    # +int(5*P.freq_in)
    
    dimen =1
    nonInt =  False
    No = 0
    
    Nz = int((domainSize)/dz) +2*dimen*pmlWidth  #CONST DISTANCE. 
    
        #Nz = (domainSize) +2*dimen*pmlWidth   #Grid size
    print(Nz, 'Nz')
    if minim== maxim:
        No = minim
        
        
       
    if minim ==  maxim:
        minim-=1#
    for N in range(minim,maxim):
        
        check = (freq_in*N)/(1/delT)
       # print(check, "check")
        if int(check)-check ==0:
            timeSteps = N 
            print("middle")
            break
        elif N == maxim-1:
           print('Could not find timestep that allowed freq_in to fall on an integer frequency bin index with range provided.') 
           #sys.exit()
           nonInt = True
           
      # Bodge fix of timesteps increase to match delT shrinkage with dz
           
          # print("hereeee")
       
    if(nonInt == True):       
        checkNear = (freq_in*minim)/(1/delT)
        N =0
        for N in range(minim, maxim):
            dummyCheck = (freq_in*N)/(1/delT)
            if (int(dummyCheck)-dummyCheck  < int(checkNear)-checkNear): # is closer to freq bin 
                #print(checkNear, dummyCheck, "checkNear, dummyCheck")
                checkNear = dummyCheck
                
                #print("dummy")
            
        
    if N >=minim:
        timeSteps = N 
    else:
        timeSteps = minim
        
    timeSteps+=int(timeSteps*(Nlam/200))  
    
    
    print('timesteps: ', timeSteps)
    
    if(timeSteps >= 2**15):
        print('timeSteps too large')
        sys.exit()
    t=np.arange(0, timeSteps, 1)*(delT)  # FOR VERIFICATION PLOTTING, EVALUATE IN CLASS
    
    nzsrcFromPml =int(0.05/dz)
    if nzsrcFromPml >= Nz*0.65:
        print(nzsrcFromPml, 'src is too far into domain')
        sys.exit()
        
    nzsrc = nzsrcFromPml + pmlWidth# Position of source 
  #  print(nzsrc)
    
    x2LocBehindSrc = 10
    if(nzsrc -x2LocBehindSrc <= pmlWidth):
        print('The probe for fft is in the PML region')
        sys.exit()   
    
    MaterialDistFromPml = int(0.1/dz)
   # print(MaterialDistFromPml)
    materialFrontEdge = MaterialDistFromPml + pmlWidth   # Discrete tile where material begins (array index)
    materialRearEdge =  Nz-1
    MaterialWidth = materialRearEdge-materialFrontEdge
   
    if MaterialWidth < 10:
        print(MaterialWidth, "width is too small or negative")
        sys.exit()
    if MaterialDistFromPml >= domainSize/dz:
        print("Material starts in CPML region")
        sys.exit()
    if materialFrontEdge <= nzsrc:
        print("Source is inside material")
        sys.exit()
    x1Loc = materialFrontEdge -20
    x2Loc = nzsrc-100
    
    eLoss =0   # sigma e* delT/2*epsilon
    mLoss = 0
    eSelfCo = (1-eLoss)/(1+eLoss)#
    eHcompsCo = 1/(1+eLoss)
    hSelfCo = (1-mLoss)/(1+mLoss)
    hEcompsCo = 1/(1+mLoss)
    
    return Nz, timeSteps, eLoss, mLoss, eSelfCo, eHcompsCo, hSelfCo, hEcompsCo, x1Loc, x2Loc, materialFrontEdge, materialRearEdge, pmlWidth, nzsrc, lamMin, dz, delT, courantNo, period, Nlam 
   
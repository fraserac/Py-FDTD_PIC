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



def envSetup(newFreq_in, domainSize, minim=400, maxim=600):# domain size and freq 
   # P.epsRe =1
    #P.epsIm = 0
    #P.muRe = 1
    #P.muIm = 0
    freq_in = newFreq_in
    #print(freq_in, "freq")
    c0 = 299792458.0
    lamMin = (c0/freq_in)
    
    #print("LamMin ", P.lamMin)
    Nlam =40
    dz =lamMin/Nlam
    #P.courantNo = 1   # LOOK INTO 2D VERSION
    delT = (dz/c0)#0.95/(P.c0*np.sqrt(1/(P.dz**2)))
   # decimalPlaces =11
   # multiplier = 10 **decimalPlaces
    #P.delT = ma.floor(P.delT* multiplier) / multiplier
    #CharImp =np.sqrt(sci.mu_0)/np.sqrt(sci.epsilon_0)
    period = 1/freq_in
    courantNo = (c0*delT)/dz
    if courantNo > 3 or courantNo <0:
        print(courantNo, "courantNo is unstable")
        sys.exit()
    pmlWidth = 3*int(lamMin/dz) #+int((30*P.freq_in)/1e9)
    #print('pmlWidth = ' , P.pmlWidth)
    if (pmlWidth >= 12000):
        print('pmlWidth too big', pmlWidth)
        sys.exit()
    # +int(5*P.freq_in)
    
    dimen =1
    nonInt =  False
    No = 0
    Nz = (domainSize) +2*dimen*pmlWidth   #Grid size
    print(Nz, 'Nz')
    if minim== maxim:
        No = minim
        
        
        
        #
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
            
    
    
    print('timesteps: ', timeSteps)
    
    if(timeSteps >= 10000):
        print('timeSteps too large')
        sys.exit()
    t=np.arange(0, timeSteps, 1)*(delT)  # FOR VERIFICATION PLOTTING, EVALUATE IN CLASS
    
    nzsrcFromPml = 6*int(lamMin/dz)
    if nzsrcFromPml >= domainSize*0.65:
        print(nzsrcFromPml, 'src is too far into domain')
        sys.exit()
        
    nzsrc = nzsrcFromPml + pmlWidth# Position of source 
  #  print(nzsrc)
    
    x2LocBehindSrc = 10
    if(nzsrc -x2LocBehindSrc <= pmlWidth):
        print('The probe for fft is in the PML region')
        sys.exit()   
    
    MaterialDistFromPml = 10*int(lamMin/dz)
   # print(MaterialDistFromPml)
    materialFrontEdge = MaterialDistFromPml + pmlWidth   # Discrete tile where material begins (array index)
    materialRearEdge =  Nz-1
    MaterialWidth = materialRearEdge-materialFrontEdge
   
    if MaterialWidth < 10:
        print(MaterialWidth, "width is too small or negative")
        sys.exit()
    if MaterialDistFromPml >= domainSize:
        print("Material starts in CPML region")
        sys.exit()
    if materialFrontEdge <= nzsrc:
        print("Source is inside material")
        sys.exit()
    x1Loc = materialFrontEdge-10
    x2Loc = nzsrc -10
    
    eLoss =0   # sigma e* delT/2*epsilon
    mLoss = 0
    eSelfCo = (1-eLoss)/(1+eLoss)#
    eHcompsCo = 1/(1+eLoss)
    hSelfCo = (1-mLoss)/(1+mLoss)
    hEcompsCo = 1/(1+mLoss)
    
    return Nz, timeSteps, eLoss, mLoss, eSelfCo, eHcompsCo, hSelfCo, hEcompsCo, x1Loc, x2Loc, materialFrontEdge, materialRearEdge, pmlWidth, nzsrc, lamMin, dz, delT, courantNo, period, Nlam 
   
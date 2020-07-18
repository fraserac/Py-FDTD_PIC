# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:08:14 2020

stability and error measurements 
"""
import numpy as np
import scipy.constants as sci
import sys
import pyttsx3 as pytalk

def spatialStab(fullTime, fullSpace, spaceStep, frequency, timeStep, plasmaFreq, resonantFreq, gamma,lim_Of_Stability = np.pi/2):
    fT = fullTime
    fS = fullSpace
    sp = spaceStep
    los = (np.pi)/2
    frq = frequency
    rad = 2*np.pi*frq
    twoPi = rad/frq
    c0 = sci.speed_of_light
    tim = timeStep
    print("timesteps: ", tim, rad*tim)
    if rad*tim > np.pi/2 and frq <= 5e9:
        print("unstable timestep", rad*tim, tim)
        sys.exit()
   
    sw = np.sinc(np.pi*frq*tim)
    pf = plasmaFreq
    oldPf = pf
    rf = resonantFreq
    es = (pf**2)/(rf**2) -1
    gam = gamma
    sqN = (rad**2*sw**2-es*rf**2*np.cos(rad*tim)+1.0j*gam*rad*sw)
    sqD = (rad**2*sw**2-rf**2*np.cos(rad*tim)+1.0j*gam*rad*sw)
    arg = (rad/c0)*(sp/2)*sw*np.sqrt(sqN/sqD)
    ans = (2/sp)*np.arcsin(arg)
    kNum = abs(ans)
    epsNum =  (((pf)**2))
    epsDom = (rf**2-(rad**2) - 1j*gam*rad)
    epsilon = 1 + (epsNum/epsDom)
    refr = np.sqrt(abs(np.real(epsilon)))
    print(epsilon, "eps stability")
    print("refr", refr)
    matAdjNum =  c0*tim*np.sin((kNum*refr*sp)/2)
    matAdjDen = refr*sp*np.sin((kNum*c0*tim)/2)
    fix = matAdjNum/matAdjDen
    print(fix, "fix")
    pf = np.sqrt(abs(fix))*pf
    vpNum = rad/kNum
    lamCont = c0/frq
    lamDisc = (twoPi)/kNum
    diff = abs(lamCont-lamDisc)
    print("Adjusted plasmaF vs original plasmaF", abs((oldPf-pf)/oldPf))
    #print(lamCont, lamDisc, diff, kNum, vpNum, c0, "lamCont, Disc, diff, kNum, diff in Vp, c0")
    if kNum*sp > los: # HIGH FREQ LIMIT
        print("unstable, wave is not resolved:", kNum*sp, kNum, abs(frq-(c0/(twoPi/kNum))))
        sys.exit()
    print("domain should be: ", (lamDisc*5)/sp)
    if kNum*sp*fS < (5*lamDisc):
        print("unstable because domain too small,", (lamDisc*5)/sp)
        sys.exit()
    return lamCont, lamDisc, diff, pf, fix

def vonNeumannAnalysis(V,P,C_V,C_P):
    betaT =np.zeros(P.Nz)
    B =np.zeros(P.Nz)
    stabilityVals = np.zeros(P.Nz*3).reshape(P.Nz, 3)
    for nz in range(P.Nz-1):
        #print(nz,"nz is:")
        #need nz too to choose a specific spatial point 
        
        #if nz < P.materialFrontEdge:
         #   plasmaFreqE = 1
          #  gammaE = 1
        #elif nz >= P.materialFrontEdge:
         #    if nz < P.materialRearEdge:
        plasmaFreqE = V.plasmaFreqE
        gammaE = V.gammaE
       
        
        betaE = (0.5*plasmaFreqE**2*P.permit_0*P.delT)/(1+0.5*gammaE*P.delT)
        kapE = (1-0.5*gammaE*P.delT)/(1+0.5*gammaE*P.delT)
        bib = ((2*P.permit_0-betaE*P.delT))
        bob =((2*P.permit_0+betaE*P.delT))
        betaT[nz] = (P.delT/bob)*(1/P.courantNo)*C_V.den_Exdz[nz]
        
        pollBit = (-0.5*(1+kapE))
        
        simpleRoot = False
        B[nz]= ((2*P.delT)/(2*P.permit_0+betaE*P.delT))*(1/P.courantNo)*C_V.den_Exdz[nz]
       
        coEfMat = [[1, -(1/P.courantNo)*(C_V.den_Hydz[10]), 0], [bib/bob, B[nz], pollBit*B[nz]], [0, betaT[nz]*C_V.den_Hydz[nz], kapE]]
        counter = 0
      #  print("in gStab")
        if abs(np.linalg.det(coEfMat)) == 0:
            print(stabilityVals[nz], "stability not met (Scarborough)")
            sys.exit()
    
        elif abs(np.linalg.det(coEfMat)) != 0:    
                for ii in range(len(np.linalg.eigvals(coEfMat))):
                    stabilityVals[nz][ii] = abs(np.linalg.eigvals(coEfMat)[ii])
                    
                for jj in range(len(stabilityVals[nz])):
                    if np.isclose(stabilityVals[nz][jj], 1, rtol = 1e-8):
                        counter +=1
                        
                if counter > 1:
                    #print("simple root present.")
                    simpleRoot = True
                if counter ==1:
                    print("single 1 on boundary, unstable", stabilityVals[nz])
                    sys.exit()
                    
                if np.max(stabilityVals[nz]) > 1:
                    if simpleRoot == False:     
                        print("Stability eigenvalues greater than one, von Neumann instability.", stabilityVals[nz])
                        breakpoint()
                       # engine = pytalk.init()
                        #engine.say("Stability eigenvalues greater than one, von Neumann instability.")
                        #engine.runAndWait()
                        #sys.exit()
                    
                #elif counter ==1:
                    # breakpoint()
                     #print("A single root exists on unit radius boundary, unstable.", stabilityVals[nz])
                 #   sys.exit()
               
                
        
                
                 
             
       
        #print(stabilityVals, "and nz:", nz)
       
            #BE CAREFUL WITH SIMPLE ROOTS ON BOUNDARY
        #scarborough criterion, for each row of coEfMat, check to see if i,i index is larger than sum of row - i,i term. 
        # necessary but not sufficient condition
        
            
        #for row in range(len(coEfMat)):
        
            #for items in range(len(coEfMat[row])):
             #   coEfMat[row][items] = abs(coEfMat[row][items])
              
           #if abs(coEfMat[row][row]) > np.sum((coEfMat[row]))-abs(coEfMat[row][row]):
               # print("Scarborough criterion met: ", abs(coEfMat[row][row]), np.sum((coEfMat[row]))-abs(coEfMat[row][row]) )
        #breakpoint()       
            #if abs(coEfMat[row][row]) <= np.sum((coEfMat[row]))-abs(coEfMat[row][row]):
               # print("Scarborough criterion not met, system MAY not converge: ", abs(coEfMat[row][row]), np.sum((coEfMat[row]))-abs(coEfMat[row][row]))
                #sys.exit()
            
            
            
            
             
            
            
            
            
            
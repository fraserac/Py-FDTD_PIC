# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 14:08:14 2020

stability and error measurements 
"""
import numpy as np
import scipy.constants as sci
import sys

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
    refr = np.sqrt(np.real(epsilon))
    print(epsilon, "eps stability")
    print("refr", refr)
    matAdjNum =  c0*tim*np.sin((kNum*refr*sp)/2)
    matAdjDen = refr*sp*np.sin((kNum*c0*tim)/2)
    fix = matAdjNum/matAdjDen
    print(fix, "fix")
    pf = np.sqrt(fix)*pf
    vpNum = rad/kNum
    lamCont = c0/frq
    lamDisc = (twoPi)/kNum
    diff = abs(lamCont-lamDisc)
    print("Adjusted plasmaF vs original plasmaF", abs((oldPf-pf)/oldPf))
    #print(lamCont, lamDisc, diff, kNum, vpNum, c0, "lamCont, Disc, diff, kNum, diff in Vp, c0")
    if kNum*sp > los: # HIGH FREQ LIMIT
        print("unstable, wave is not resolved:", kNum*sp, kNum, abs(frq-(c0/(twoPi/kNum))))
        sys.exit()
    print("domain should be: ", (lamDisc*10)/sp)
    if kNum*sp*fS < (6*lamDisc):
        print("unstable because domain too small,", (lamDisc*10)/sp)
        sys.exit()
    return lamCont, lamDisc, diff, pf, fix
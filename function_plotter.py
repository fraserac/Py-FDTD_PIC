# -*- coding: utf-8 -*-
"""
Created on Sun Dec 27 19:57:53 2020
FUNCTION PLOT
@author: Owner
"""
import numpy as np
import matplotlib.pyplot as plt
#import MasterController as Mast

#instantiate classes for quick run option 

def funcPlot(V,P):
    
    plt.close('all')
    plt.xlabel("Frequency")
    plt.ylabel("Epsilon")
    plt.title("Permittivity")
    
    gammaE = V.gammaE
    omega_0E= V.omega_0E
    
    freq_in = np.arange(1e7, 5e10, 1e8)
    
    eps8= 1.5
    epsS =3
    plasmaFreqE = V.plasmaFreqE
    pi = np.pi
    epsNum = plasmaFreqE**2
    epsDom = (omega_0E**2-(2*pi*freq_in*2*pi*freq_in) -2j*gammaE*2*pi*freq_in)
    
    epsDom2 =(omega_0E**2-(4*pi*freq_in*4*pi*freq_in) -2j*gammaE*4*pi*freq_in)
    
    epsilon2 = 1+ epsNum/(epsDom**2*epsDom2)
    epsilon =  1+epsNum/epsDom
    
    plt.plot(freq_in, np.real(epsilon))
    plt.plot(freq_in, np.imag(epsilon))
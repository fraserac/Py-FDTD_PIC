# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 14:21:36 2019

@author: Fraser

Transform operations:
    
This script will transform between domains to perform operations, such as fourier transforms and Z transforms and so on. 

This will enable us to take in the freq dom constitutive data from COMSOL/Matlab etc and transform the time domain fields produced in 
baseFDTD, multiply the two quantities and then transform back to time domain to evaluate the effects of the metamaterial
on the field. 

THIS CODE WILL EVENTUALLY TIE IN WITH AN AUXILIARY DIFFERENTIAL EQUATION FDTD FOR THE NONLINEAR MAGNETISATION.

NONLINEAR BOUNDARY CONDITIONS? 

FOR NOW, PURPOSE OF THIS SCRIPT IS TO SIMPLY TAKE A FREQUENCY DOMAIN COMPLEX PERMITTIVITY, A TIME DOMAIN SINUSOIDAL
E FIELD, TRANSFORM E FIELD TO FREQ DOM, PRODUCT EPS AND E, then transform product back to new E field. Plot new E field in time domain 
at various times?


What will this do? How will this tie back into sim?
"""

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt
from decimal import *

def FourierTrans(itemForTransf1, itemForTransf2, x1Loc, time_vec, delT):
    sig_fft1 = 2*fftpack.fft(itemForTransf1)/len(itemForTransf1)
    sig_fft2 = 2*fftpack.fft(itemForTransf2)/len(itemForTransf2)
   # sig_fft2 = fftpack.fftshift(sig_fft2)
   # sig_fft2 = fftpack.ifft(sig_fft2)
    transm = sig_fft2/sig_fft1
   # transm = fftpack.fftshift(transm)
    #plt.plot(time_vec, transm, label='Original signal ')  # make this work by feeding in stuff
    sample_freq = fftpack.fftfreq(len(itemForTransf1), d= delT)
   # sample_freq = fftpack.fftshift(sample_freq)
    #transmPow = np.abs(transm)
   # plt.plot(sample_freq, transmPow, label='Power spectrum freq. ')
    return transm, sig_fft1, sig_fft2, sample_freq
    
    # feed in x1ColBe and x1ColAf separately. 
    


def ReflectionCalc():
    pass

def TransmissionCalc():
    pass








########################################
    
""" 
np.random.seed(123)   #pseudo-random number generator


period = 1/1e9
time_step = 0.01e-9   # Nyquist limit stuff, there will be error in amplitude/freq resolution to account for
maxTime = time_step*(2**9)
time_vec = np.arange(0, maxTime, time_step)
sig = np.sin(2 * np.pi / period * time_vec)

  # pseudo-random signal generatio

plt.figure(figsize=(6, 5))
plt.plot(time_vec, sig, label='Original signal')

# The FFT of the signal
sig_fft = fftpack.fft(sig)

# And the power spectrum (sig_fft is of complex type)
power = 2*(np.abs(sig_fft)/len(sig))

# The corresponding frequencies
sample_freq = fftpack.fftfreq(sig.size, d=time_step)# Freq resolution set up 

# Plot the FFT power
plt.figure(figsize=(6, 5))
plt.plot(sample_freq, power)
plt.xlabel('Frequency [Hz]')
plt.ylabel('power')

# Find the peak frequency: we can focus on only the positive frequencies

pos_mask = np.where(sample_freq > 0) # linq style command whereby a condition is checked against over iterable data
#outputs array of integer element no's where condition is met (ignore negative conjugate)
freqs = sample_freq[pos_mask]
peak_freq = freqs[power[pos_mask].argmax()]  # argmax Return value of maximum elements along an axis.
## understand syntax above
peak_freq_Ind = np.argmax(power)
peak_freqMin = int(peak_freq_Ind-peak_freq_Ind*0.7)
peak_freqMax = int(peak_freq_Ind+peak_freq_Ind*1.2)
# Check that it does indeed correspond to the frequency that we generate
# the signal with
similar = np.allclose(peak_freq, 1./period)
print(similar)
# all close more like are these two values close within tolerance?

# An inner plot to show the peak frequency
axes = plt.axes([0.55, 0.3, 0.3, 0.5]) # THIS IS NEAT TRICK, WOULD LOOK GOOD FOR PRESENTATIONS
plt.title('Peak frequency')
plt.plot(freqs[peak_freqMin:peak_freqMax], power[peak_freqMin:peak_freqMax])
plt.setp(axes, yticks=[])

"""
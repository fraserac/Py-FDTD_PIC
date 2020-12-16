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
from scipy import fftpack, fft
from matplotlib import pyplot as plt
import sys
from numba import njit as nj


def RefTester(V,P, y, m):
    #breakpoint()
    fs = 1/P.delT
    
    Y = fftpack.fft(y)
    Ypow = (2*np.abs(Y))/P.timeSteps
   
    freqs = fftpack.fftfreq(len(y), d = P.delT)
   # breakpoint()
    fig, ax = plt.subplots()
    finder = np.where(freqs==P.freq_in) # outputs tuple
    if  len(finder[0]) != 0:
        val = Ypow[finder[0][0]]
    else:
        bob = np.isclose(freqs, P.freq_in, rtol = 1e-3)
        bobInd = np.where(bob == True)
        valRange = Ypow[bobInd[0][0]-50: bobInd[0][0]+50]
        val = np.argmax(valRange)
    
    
        
    ax.stem(freqs, m*2*np.abs(Y)/len(Y), use_line_collection = True)
    ax.set_xlabel('Frequency in Hertz [Hz]')
    ax.set_ylabel('Frequency Domain (Spectrum) Magnitude')
    ax.set_xlim(0,fs / 2)
    ax.set_ylim(-2, 2)
    
    
    bob = (np.abs(Y[int(len(Y)/2)+10:len(Y)-500])*m*2)/len(Y)
    fig1, ax1 = plt.subplots()
    ax1.plot(bob)
    
    return bob, freqs[int(len(freqs)/2)+10:len(freqs)-500], val
    

def FourierTrans(P,V, itemForTransf1, time_vec, delT):
  
    powers=0.0
    
    iii=0
    jjj=0
    cleanSig1 =[]#np.zeros(int(P.period/P.delT)+1)
    cleanSig2= []#np.zeros(int(P.period/P.delT)+1)
    phase1 = False
    finished1 = False
    finished2 = False
        
    #
     
    prevWave = 0.0
    prevPrevWave=0.0
    zeroCross = 0
    ### not going into to third if statement
    plt.plot(time_vec, itemForTransf1, label='Original signal')
    for yy in range (len(itemForTransf1)):
        if zeroCross < 2:  
           #print(abs(itemForTransf1[yy]),"itemFortrans")
           
           if(abs(itemForTransf1[yy]) >=0.05 ):
                
                prevPrevWave = prevWave
                prevWave = itemForTransf1[yy]
                
               
                cleanSig1= np.append(cleanSig1, itemForTransf1[yy])
                iii+=1
                    #print(cleanSig1, "cleanSig1")
        if  np.sign(prevPrevWave) != np.sign(itemForTransf1[yy]) and np.sign(prevPrevWave) != 0.0 and  np.sign(itemForTransf1[yy]) !=0.0:
            #print("prevPrevWave, itemFor", prevPrevWave, itemForTransf1[yy])
            cleanSig1[iii-1] = 0.0
            zeroCross +=1
                  
        if zeroCross ==2:
            cleanSig1 = np.append(cleanSig1, -cleanSig1) 
           # print("here in zerocross ==2")
            break#
 
     
    for xx in range(6,14):
       if len(time_vec) <= 2**xx:
            powers = xx
            #print("Power ", power)
            break
    if powers == 0.0:
        print("power is still 0")
        sys.exit()
      
    
    maxTime = P.timeSteps
    tExtend=2**powers-maxTime
    maxTime = (tExtend +maxTime)*P.delT
    #print("maxtime", maxTime)
    time_step = P.delT
    #time_vec = np.arange(0, maxTime, time_step)
    signal = np.zeros(int(maxTime/P.delT))
   # tExtend2=2**powers-len(itemForTransf2)
    #timePadded =np.append(time_vec,np.zeros(tExtend))
    #timePadded2 = np.append(time_vec, np.zeros(tExtend2))
    #print(len(cleanSig1), "clean sig length")
    for tt in range(0,len(cleanSig1)):
        signal[tt] = cleanSig1[tt]
       # print("signal")
        
        
        
    period = 1/P.freq_in
       # Nyquist limit stuff, there will be error in amplitude/freq resolution to account for
    #padded time later
    
    #print(len(time_vec), "timeVec")
    #print(time_vec[4])
    sig = signal
   # print(sig, "sig")
    
      # pseudo-random signal generatio
    
    plt.figure(figsize=(6, 5))
    
    
    # The FFT of the signal
    sig_fft = fftpack.fft(sig)
    
    # And the power spectrum (sig_fft is of complex type)
    power = 2*(np.abs(sig_fft)/len(sig))
    
   
    #cleanSig1 = np.append(cleanSig1, timePadded)
    #cleanSig2 = np.append(itemForTransf2, timePadded2)
    #plt.plot(time_vec, itemForTransf1, label='Original signal')
    # The corresponding frequencies
   
    
    sample_freq = fftpack.fftfreq(len(sig), d=time_step)# Freq resolution set up 
    
    # Plot the FFT power
    plt.figure(figsize=(6, 5))
    plt.plot(sample_freq, power)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('power')
    
    # Find the peak frequency: we can focus on only the positive frequencies
    
    pos_mask = np.where(sample_freq > 0) # linq style command whereby a condition is checked against over iterable data
    #outputs array of integer element no's where condition is met (ignore negative conjugate)
    freqs = sample_freq[pos_mask]
    peak_freq =  freqs[power[pos_mask].argmax()]  # argmax Return value of maximum elements along an axis.
    ## understand syntax above
    peak_freq_Ind = np.argmax(power)
    peak_freqMin = int(peak_freq_Ind-peak_freq_Ind*0.7)
    peak_freqMax = int(peak_freq_Ind+peak_freq_Ind*1.2)
    # Check that it does indeed correspond to the frequency that we generate
    # the signal with
    similar = np.allclose(peak_freq, 1./period)
    print("similar = ", similar)
    # all close more like are these two values close within tolerance?
    
    # An inner plot to show the peak frequency
    axes = plt.axes([0.55, 0.3, 0.3, 0.5]) # THIS IS NEAT TRICK, WOULD LOOK GOOD FOR PRESENTATIONS
    plt.title('Peak frequency')
    plt.plot(freqs[peak_freqMin:peak_freqMax], power[peak_freqMin:peak_freqMax])
    plt.setp(axes, yticks=[])
    

    return sig_fft,  sample_freq# transm, sig_fft1, sig_fft2, freq_list, timePadded
    
    # feed in x1ColBe and x1ColAf separately. 
    


def ReflectionCalc(P, V, sample_freq, sig_fft, sig_fft2):
    
   #print("timePadding ", timePadded)    
    ## need to remove high end noise and conjugate side
   # breakpoint()
    #sig_fft = 2*abs(sig_fft)/len(sig_fft)
    #sig_fft2 = 2*abs(sig_fft2)/len(sig_fft2)
    sig_fft_Cut = sig_fft#[int(len(sig_fft)/2):-int(len(sig_fft)/30)]
    sig_fft_2_Cut = sig_fft2#[int(len(sig_fft2)/2):-int(len(sig_fft2)/30)] 
    sample_freq = sample_freq#[int(len(sample_freq)/2):-int(len(sample_freq)/30 )]                         
    #breakpoint()
    freqIndex =np.argmax(sig_fft_Cut)
    
    freqIndex2= np.argmax(sig_fft_2_Cut)
    #breakpoint()
    fig3, ax3 = plt.subplots()
    ax3.plot(sample_freq, abs(sig_fft_Cut))
    ax3.set_title("sig_fft abs")
    
    fig4, ax4 = plt.subplots()
    ax4.plot(sample_freq, abs(sig_fft_2_Cut))
    ax4.set_title("sig_fft2 abs")
    
    
    #print("FREQ INDICES ",freqIndex, freqIndex2)
    if int(freqIndex)-freqIndex !=0:
        freqIndex = int(freqIndex)
    reflectionCo = abs(sig_fft_Cut[freqIndex]/(sig_fft_2_Cut[freqIndex2]))
    if np.isnan(reflectionCo):
        print("NAN REFLECTION, timeSteps might not be big enough.", reflectionCo)
        #sys.exit()
   # breakpoint()
    #print(sig_fft2[freqIndex])
   
    
    #print("SIG!!!!", abs(sig_fft2[int(freqIndex)]), abs(sig_fft1[int(freqIndex)]))
    #print(reflectionCo)
    return reflectionCo

def TransmissionCalc():
    pass


def genFourierTrans(V,P, C_V, C_P, Tddata):
    """
    Time vector, data to transform, frequency x axis
    
    """
    Td = Tddata
    FDdata = np.zeros(P.timeSteps)
    FDVec = np.zeros(P.timeSteps)   
    FDdata = fftpack.fft(Td)
    #FDdata = FDdata[int(len(FDdata/2)): -int(len(FDdata/50))]
    FDXaxis = fftpack.fftfreq(len(FDdata), d= P.delT) #sample_freq
    FDdataPow = 2*abs(FDdata)/(len(FDdata))
    plt.plot(FDXaxis, FDdataPow)
    
    
  
    return FDdata, FDXaxis, FDdataPow


#
########################################
    

def fourierTest(sig, P, V):
    period = 1/P.freq_in
    plt.figure(figsize=(6, 5))
    time_vec = np.arange(P.timeSteps)*P.delT
    #breakpoint()
    plt.plot(time_vec, sig, label='Original signal')
    time_step = P.delT
    # The FFT of the signal
    sig_fft = fftpack.fft(sig)
    
    # And the power spectrum (sig_fft is of complex type)
    power = 2*(np.abs(sig_fft)/len(sig_fft))
    
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
    peak_freq =  freqs[power[pos_mask].argmax()]  # argmax Return value of maximum elements along an axis.
    ## understand syntax above
    peak_freq_Ind = np.argmax(power)
    peak_freqMin = int(peak_freq_Ind-peak_freq_Ind*0.7)
    peak_freqMax = int(peak_freq_Ind+peak_freq_Ind*1.2)
    # Check that it does indeed correspond to the frequency that we generate
    # the signal with
    similar = np.allclose(peak_freq, 1./period, atol = 1e-3)
    print(similar)
    # all close more like are these two values close within tolerance?
    
    # An inner plot to show the peak frequency
    axes = plt.axes([0.55, 0.3, 0.3, 0.5]) # THIS IS NEAT TRICK, WOULD LOOK GOOD FOR PRESENTATIONS
    plt.title('Peak frequency')
    plt.plot(freqs[peak_freqMin:peak_freqMax], power[peak_freqMin:peak_freqMax])
    plt.setp(axes, yticks=[])
    return power,sample_freq
  
"""
    
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
peak_freq =  freqs[power[pos_mask].argmax()]  # argmax Return value of maximum elements along an axis.
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

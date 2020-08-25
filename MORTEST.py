# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:25:45 2020

@author: Fraser
Model order reduction simple tests

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as sgf
from scipy.linalg import lu 
from scipy import sparse
from scipy import signal as sig
from numba import njit as nj
from scipy.sparse.linalg import splu, inv
from numba import int32, float32, int64, float64, boolean
import numba
from numba.experimental import jitclass as jc
from scipy import fftpack,fft
from statsmodels.nonparametric.smoothers_lowess import lowess

#refactor for numba
#issues np.matrix, vt typing, norm of matrix, hermitian 
#replace np matrix with array 
specEA= [('H', float64[:,:]), 
         ('V', float64[:,:]), 
         ('vt',float64[:]),#
         ('k', int32)]

@jc(specEA)
class EA(object):
    def __init__(self, k, n):
        self.H  = np.zeros((k+1,k))
        self.V = np.zeros((n,n))
        self.vt = np.zeros(n)
        self.k = k


@nj
def ExampleArnoldi(A, v0, k, EAVars ): # Issue with [0,0] 
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A
    Author: Vasile Gradinaru, 14.12.2007 (Rennes)
    """
    #print 'ARNOLDI METHOD'
   # inputtype = A.dtype.type
    
    EAVars.V = v0
    
   # breakpoint()
    for m in range(k):
        EAVars.vt = A@EAVars.V[ :, m]
        #print("progress through Arnoldi: ", m, "/", k)
        for j in range(m+1):
            EAVars.H[ j, m] = (EAVars.V[ :, j].transpose().conj() @ EAVars.vt )
            
            
            EAVars.vt -= EAVars.H[ j, m] * EAVars.V[:, j]
        EAVars.H[ m+1, m] = np.linalg.norm(EAVars.vt);
        if m is not k-1:
            EAVars.V[:,j] =   EAVars.vt / EAVars.H[ m+1, m] 
            
    return EAVars.V,  EAVars.H[:-1,:]
    


def smooth(x,window_len=11,window='hanning'):
    
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError ("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y

def luInvert(a):
    print("LUinvert")
    a = sparse.csc_matrix(a)
    M = splu(a)
    M = M.solve(In)
    M = sparse.csc_matrix(M)
    """
    Per, L, U = lu(a)
    U = U + 1e-6
    Linv = np.linalg.inv(L)
    Uinv = np.linalg.inv(U)
    Linv = sparse.csc_matrix(Linv)
    Uinv = sparse.csc_matrix(Uinv)
    M = Uinv@Linv
    M = M.todense()
    """
    
    return M


# FFT OF REDUCED ORDER MODEL TO FIND HARMONICS, CLEAN NOISE, IFFT, UNREDUCE
#CHANGE SMAT TO MATCH FREQ
    
#(s0In - A)^-1 = NEWSMAT
# s0 = i*2*np.pi*freq 
# In = I(nxn)
# A = smat

freq = 4e9
freq2 = 8e9

#s02 = (1j)*2*np.pi*freq2
noOfVals = 3500
In = np.identity(noOfVals)

S = np.linspace(-freq*np.pi, freq*np.pi, noOfVals)
S2 =np.linspace(-freq2*np.pi, freq2*np.pi, noOfVals)

Y = np.sin(S*2*np.pi*1e-10) + 0.5*np.sin(S2*np.pi*1e-10)
#plt.plot(Y)

nzsrc = int(100)
Nz = noOfVals
k =1000
Smat = np.diag(Y)

B = np.zeros((Nz, Nz))
B[nzsrc, nzsrc] = 1
B[int(Nz/2)+1+nzsrc-1, nzsrc-1 +int(Nz/2)+1] =1

freqNo = 10
M = In
M = sparse.csc_matrix(M)
Mtemp =[]
M2 = In
M2 = sparse.csc_matrix(M2)
for i in range(4,9,4):
    s0 = (1j)*2*np.pi*i
    M = M@luInvert(s0*In-Smat)
     
M = M.todense()  
M = np.asarray(M, dtype = np.float64) 
    

for jj in range(4,9,4):
    s0 = (1j)*2*np.pi*jj
    M2= M2@(luInvert((s0*In)-Smat).conj().T)

M2 = M2.todense()
M2 = np.asarray(M2, dtype = np.float64)

#SmatInv = np.linalg.inv(Smat)
Xn = np.ones(len(Y)).reshape(len(S),1)
v0 = np.random.randint(5, size=(len(Y),k))
v0 = v0.astype(float)
v2 = np.random.randint(5, size=(len(Y),k))
v2 = v0.astype(float)
    
    
EAVars = EA(k, len(v0))

#U = np.ones(len(Y))
#SmatInvB = SmatInv@B
#plt.plot(Smat@Xn)
H = np.zeros((k+1,k))
#v0[:,0]= (v0[:,0] / np.linalg.norm(v0[:,0]))
v0, R = np.linalg.qr(v0)
v2, R2 = np.linalg.qr(v2)
v0 = M@v0
v2 = M2@v2

#v2[:,0]= (v2[:,0] / np.linalg.norm(v2[:,0]))
print("entering arnoldi")
V, H = ExampleArnoldi(M, v0, k, EAVars)
V = np.block([np.real(V), np.imag(V)])


W, R = ExampleArnoldi(M2,v2, k, EAVars)

W = np.block([np.real(W),np.imag(W)])

#W =V
window_len = noOfVals

check = W.T@V
Smat_red = W.T@Smat@V
Xn_red = W.T@Xn

#smooth with savgol and then stretch with e^itheta
xaxis = np.arange(k).ravel()
prepre= np.array(Smat_red@Xn_red).ravel()

XnP1_red = Smat_red@Xn_red
#XnP1_red_smoo = sgf(XnP1_red.ravel(), len(XnP1_red), 7 )
XnP1 = W@XnP1_red
XnP1 = np.asarray(XnP1)

if noOfVals%2 == 0:
    wind = int(noOfVals/2+1)
elif noOfVals%2 ==1:
    wind = int(noOfVals/2)
    
XnP1Act =Smat@Xn
def experimPhaseShifter(a, phi = np.pi*0.1):
    b = np.sin(np.arcsin(a) +(phi))
    #b = np.poly1d(np.polyfit(np.arange(len(b1)), b1.ravel(), 7))(np.arange(len(b1)))
    return b
#XnP1Smoo = sgf(XnP1.ravel(), noOfVals,7)
#mult = np.max(XnP1)/np.max(XnP1Smoo)

XnP1Clean =smooth(np.array([XnP1]).ravel(), 150, window = 'flat')
XnP1Clean2 = smooth(np.array([XnP1Clean]).ravel(), 150, window = 'hanning')

XnPol = np.poly1d(np.polyfit(np.arange(len(XnP1Clean)), XnP1Clean.ravel(), 7))(np.arange(len(XnP1Clean))) #
ratio = len(XnP1)/k
#XnP1Exp = experimPhaseShifter(XnP1Clean) 
#XnP1Exp -= np.median(XnP1Exp)
#XnP1Exp2 = experimPhaseShifter(XnP1Clean, phi = -30) 
#XnP1Exp2 -= np.median(XnP1Exp2)
b,a =sig.butter(2,0.5)
zi = sig.lfilter_zi(b,a)
XnP1_0 = sig.filtfilt(b,a, XnP1, padlen = XnP1.shape[-1]-2)
XnP1_0 =np.poly1d(np.polyfit(np.arange(len(XnP1_0)), XnP1_0.ravel(), 9))(np.arange(len(XnP1_0)))
#plt.plot(XnP1Exp*2*ratio, 'k--')
#plt.plot(XnP1Exp2*2*ratio, 'r--')
#plt.plot(XnP1Clean*ratio*2)
#plt.plot(XnP1Clean2*ratio*2)
plt.plot(XnP1_0*ratio)
plt.plot(XnP1Act, 'go')
#plt.plot(XnPol*ratio)

plt.legend()
print("Finished")

trans = fftpack.fft(XnP1Clean)
transAct =fftpack.fft(XnP1Act)
FDXaxis = fftpack.fftfreq(len(trans), d= 0.0062847564963046665)
FDXaxisAct = fftpack.fftfreq(len(transAct), d = 0.0062847564963046665)

print(ratio, "improvement factor")


"""
Spectrum capture of polyfit/MOR seems good.
To do: Improve speed of arnoldi iterations
H2, Hinf norm checks
Incorporate into main code
Re-derive MNA form for linear dispersive
eig/singular pert for stab
PRIMA/SPRIM for passivity
all the errors
refactor codes for numba including overload numba when needed
refactor for oop good coding. 
expand to nonlin
expand to 2d
incorp particle
incorp specific comsol/matlab pipeline
verifications 
optimise
experiment
write thesis

"""





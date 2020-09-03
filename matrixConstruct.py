# -*- coding: utf-8 -*-
"""
Creating logic for matrix construction for MOR methods
Using vectorise and ogrid 
"""


import numpy as np
import time 
from numba import njit as nj
from scipy import linalg as li
from scipy import sparse, mat
from scipy import signal as sig
from scipy.sparse.linalg import splu, inv
import BaseFDTD11 as BF
import sys
from sklearn.linear_model import Ridge
#from MORTEST import luInvert, ExampleArnoldi
from numba import int32, float32, int64, float64, boolean
import numba
from numba.experimental import jitclass as jc

import matplotlib.pyplot as plt
from scipy.optimize import least_squares as lstsq
from statsmodels.nonparametric.smoothers_lowess import lowess
from matplotlib.animation import FuncAnimation
import os
import shutil
import cv2
import natsort

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
def ExampleArnoldi(A, v0, k, EAVars): # Issue with [0,0] 
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

def initMat(V,P, C_V, C_P):
    blocks =2
    dRange =P.Nz#check
    A = np.zeros(blocks**2*(dRange)*dRange).reshape(blocks*dRange, blocks*dRange)
    A = sparse.csc_matrix(A)
    B = np.zeros((blocks*dRange))
    
    source = np.zeros((blocks*dRange))
    #B = sparse.csc_matrix(B)
   
    #source = sparse.csc_matrix(source)
    return A, B, source

def coEfFinder(V, P, C_V, C_P):
      betaE = (0.5*V.plasmaFreqE**2*P.permit_0*P.delT)/(1+0.5*V.gammaE*P.delT)
      kapE = (1-0.5*V.gammaE*P.delT)/(1+0.5*V.gammaE*P.delT)
      selfCoEfHy = 1
      selfCoEfEx = ((2*P.permit_0-betaE*P.delT)/(2*P.permit_0+betaE*P.delT))
      curlinExCoEf=((2*P.delT)/(2*P.permit_0+betaE*P.delT))*(1/P.courantNo)*C_V.den_Exdz
      curlinHyCoef = (1/P.courantNo)*C_V.den_Hydz
      
      return selfCoEfHy, selfCoEfEx, curlinExCoEf, curlinHyCoef



def ABuild(A, P, De, Dh, K, Kt, blocks=2):  # VERY SPARSE UPDATE MATRIX OVER ALL SPACE
    print("Building A matrix...")
    A1 = np.identity(P.Nz+1)
    A1 = sparse.csc_matrix(A1)
    A2 = -1*np.linalg.inv(De)*K*P.delT
    A2 = sparse.csc_matrix(A2)
    A3 = np.linalg.inv(Dh)*Kt*np.identity(P.Nz+1)*P.delT   # with conductivities there will not be as many I
    A3 = sparse.csc_matrix(A3)
    A4 = np.identity(P.Nz+1)*np.linalg.inv(Dh)*Kt*np.linalg.inv(De)*K*P.delT**2
    A4 = sparse.csc_matrix(A4)   
    A = sparse.bmat([[A1, A2], [A3, A4]])

    return A, blocks

def RandFBuild(P, De, Dh, K, Kt):   # MOR method outlined by Bart Denecker
    
    R1 = (1/(P.delT*P.c0))*De
    R1 = sparse.csc_matrix(R1)
    R2 = -0.5*K
    R2 = sparse.csc_matrix(R2)
    R3 = -0.5*Kt
    R3 = sparse.csc_matrix(R3)
    R4 = (1/(P.delT*P.c0))*Dh
    R4 = sparse.csc_matrix(R4)
    
    F1 = np.zeros((P.Nz+1, P.Nz+1))
    F1 =sparse.csc_matrix(F1)
    F2 = 0.5*K
    F2 = sparse.csc_matrix(F2)
    F3 = R3
    F4 = F1
    
    R = sparse.bmat([[R1, R2],[R3, R4]])
    F = sparse.bmat([[F1, F2],[F3, F4]])
    
    
    return R, F


    

def BasisVector(V, P, C_V, C_P):
    Xn = np.block([V.Ex, V.Hy])
    Xn = Xn.reshape(len(Xn), 1)
    return Xn

def SourceVector(V,P, C_V, C_P):
    
    UnP1A = np.ones(P.Nz+1)
    UnP1B = np.ones(P.Nz+1)
    
    
    return UnP1A, UnP1B

def BGenerator(P):
    B = np.zeros((P.Nz*2+2, P.Nz*2+2))
    B[P.nzsrc, P.nzsrc] = 1
    B[P.Nz+1+P.nzsrc-1, P.nzsrc-1 +P.Nz+1] =1
    return B
"""
This function builds the blocks for the MOR method update matrix
Firstly, build bare bones basic from ground up. E,H + source.


QUADRANTS: A1 A2
           A3 A4
          
An Explicit and Unconditionally Stable FDTD
Method for Electromagnetic Analysis
Md. Gaffar and Dan Jiao
also Xihao reduction paper


Build De, Dh

"""
def blockBuilder(V, P, C_V, C_P):
    K =  np.zeros((P.Nz+1, P.Nz+1))
    DiagE = np.ones(P.Nz+1)
    DiagH = np.ones(P.Nz+1)
    De = np.diag(DiagE)
    Dh = np.diag(DiagH)
    for j in range(1, P.Nz):
                K[j, j] = -1/P.dz
                K[j, j-1] = 1/P.dz
    Kt = K.T
    return De, Dh, K, Kt


    
def TimeIter(A, B, UnP1A, UnP1B, Xn, V, P, C_V, C_P):
    Exs, Hys = BF.SmoothTurnOn(V,P)
    print("MOR Time stepping...")
   
    for jj in range(P.timeSteps):
        print("Timestepping, step number: ", jj)
        UnP1A, UnP1B, B = BAndSourceVector(V, P, C_V, C_P)
        UnP1A[P.nzsrc] = Hys[jj]/P.CharImp
        UnP1B[P.nzsrc-1] = -Exs[jj]
        UnP1 = np.block([UnP1A, UnP1B])
        #UnP1 = sparse.csc_matrix(UnP1)
        B = sparse.csc_matrix(B)
        #Xn = sparse.csc_matrix(Xn)
        XnP1 = np.zeros(2*P.Nz+2)
        #XnP1 = sparse.csc_matrix(XnP1)
        if jj == 0:
            XnP1 = A@Xn.T
            XnP1 += B@UnP1.T
        elif jj >0:
            XnP1 = A@XnP1.T 
            XnP1 += B@UnP1.T
        
       # XnP1 = XnP1.todense()    
        for ii in range(len(V.Ex)):
            V.Ex[ii] = XnP1[ii]
            V.Hy[ii] = XnP1[ii+len(V.Ex)]
        V.Ex_History[jj] = V.Ex
        
        Xn = XnP1
        
    return B, V.Ex, V.Ex_History, V.Hy, UnP1, Xn

def luInvert(a):
    In=np.identity(len(a))
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


def smooth( x,window_len=5,window='hanning'):
    
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
    print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    print(y.shape)
    return y[:len(x)]


def solnDenecker(R, F, A, UnP1A, UnP1B, Xn, V, P, C_V, C_P):
    Exs, Hys = BF.SmoothTurnOn(V,P)
   # breakpoint()
    print("MOR Time stepping...")
    if R.shape != (2*P.Nz+2, 2*P.Nz+2):
        print("R is wrong shape")
        sys.exit()
    if F.shape != (2*P.Nz+2, 2*P.Nz+2):
        print("F is wrong shape")
        sys.exit()#
    
        
    XnP1 = np.zeros(2*P.Nz+2)
   
    XnP1 = XnP1.reshape(len(XnP1), 1)
    k = 600
    ratio = len(XnP1)/k
    
    #XnP1 =sparse.csc_matrix(XnP1)
    #Xn = sparse.csc_matrix(Xn)
    #R = sparse.csc_matrix(R)
    #F = sparse.csc_matrix(F)
     # verify too big or small? A priori if better
    #Q, H, k, Xn_red, XnP1_red, Q_sp = MORViaSPRIM(Xn, XnP1, 2*P.Nz+2,R, k, F, P, V, C_V, C_P)
    summer = (R+F)
    winter = (R-F)
  
    
    summer = sparse.csc_matrix(summer)
    if(sparse.issparse(winter)):
        winter = winter.todense()
    winter = np.asmatrix(winter)
    
    #winter = sparse.csc_matrix(winter)
    #s_0 = P.freq_in
   # stencil = (s_0*summer -winter)
    #stencil = stencil.todense()
    
    #Per, stenL, stenU = li.lu(stencil)
    #invStenL = np.linalg.inv(stenL)
    #invStenU = np.linalg.inv(stenU)
    #invSten = invStenU@invStenL
  #  Per, stenL, stenU = li.lu(winter)
    #invStenL = np.linalg.inv(stenL)
    #invStenU = np.linalg.inv(stenU)
    #invSten = invStenU@invStenL
    #invSten = sparse.csc_matrix(invSten)
   # M = invSten@summer #multiply summer by inverse of LU decomp
    B = BGenerator(P)
    B = sparse.csc_matrix(B)
   # M2 = invSten@B
   # Q = invSten@B
    #V_a = np.random.randint(4, size =(len(winter), k) )
    #V_a = V_a.astype(float)
    #Q = Q.astype(float)
    #M = M.astype(float)
    
   
    #M = (S0(summer)-winter)^-1@summer, R = (s0((summer)-winter)^-1@B)
    #LU DECOMP OF M 
    #inv = sparse.csc_matrix(inv)
    
    In = np.identity(len(winter))
    M = In
    M = sparse.csc_matrix(M)
    M2 = In
    M2 = sparse.csc_matrix(M2)
    for i in range(int(P.freq_in),int(P.freq_in*2+1),int(P.freq_in/4)):
        s0 = (1j)*2*np.pi*i
        M = M@luInvert(s0*In-winter)
         
    M = M.todense()  
    M = np.asarray(M, dtype = np.float64) 
        
    
    for jj in range(int(P.freq_in),int(P.freq_in*2+1),int(P.freq_in/4)):
        s0 = (1j)*2*np.pi*jj
        M2= M2@(luInvert((s0*In).conj().T-winter.conj().T))
    
    M2 = M2.todense()
    M2 = np.asarray(M2, dtype = np.float64)
    
    v0 = np.random.randint(5, size=(len(winter),k))
    v0 = v0.astype(float)
    v2 = np.random.randint(5, size=(len(winter),k))
    v2 = v0.astype(float)
    v0, R = np.linalg.qr(v0)
    v2, R2 = np.linalg.qr(v2)
    v0 = M@v0
    v2 = M2@v2
    EAVars = EA(k, len(v0))
    #A = sparse.csc_matrix(A)
    #Q = np.random.randint(5, size=(2*P.Nz+2,k+1))
    #Q= Q.astype(float)
    
    #if(sparse.issparse(M)):
     #   M = M.todense() 
    print("Starting Arnoldi process................")
    tic = time.perf_counter()
    V_sp, H = ExampleArnoldi(M, v0, k, EAVars)
    #breakpoint()
    #W_sp, H =ExampleArnoldi(M2@summer, v2, k, EAVars)
    toc = time.perf_counter()
    print("FINISHED ARNOLDI in: ",toc-tic)
    #V_sp = np.block([np.real(V_sp), np.imag(V_sp)])
    #breakpoint()
    #V_sp, R = np.linalg.qr(V_sp)#

    print("Performing Orthonormal tests")
    check = np.allclose(V_sp.T @ V_sp, np.eye(V_sp.shape[1]), rtol =1e-1)
    print(" ortho check: ", check)
     #unit tests, check hessenberg/orthonorm
    #print("Projector check:" )
    
    #breakpoint()
    
    #check = np.allclose(V_sp.T@V_sp, V_sp)
    #print(check)
   # breakpoint()
    
    
    #Xn_red = np.dot(Q_spt, Xn)
   # Q_sp = sparse.csc_matrix(Q_sp)
    #Q_spt = sparse.csc_matrix(Q_spt)
   # breakpoint()
    
    #Q, k, V_sp =MORViaSPRIM(M, V_sp, XnP1, R, k, F, P, V, C_V, C_P)
    V_spt = V_sp.T
   # breakpoint()
    XnP1_red =  V_spt@XnP1
    
   # M = sparse.csc_matrix(M)
    #A_red = M#Q_spt@M@Q_sp
    winter_red = V_spt@winter@V_sp
    summer_red = V_spt@summer@V_sp
   
    #A_red = sparse.csc_matrix(A_red)
    #if (sparse.issparse(A_red)):
      #  A_red = A_red.todense()
    print("Eigenvalue decomp...")
    #breakpoint()
    #winter_red= np.real(eigenvalueDecomp(winter_red)) 
    #summer_red  =np.real(eigenvalueDecomp(summer_red))
   # breakpoint()
    winter_red = sparse.csc_matrix(winter_red)
    summer_red = sparse.csc_matrix(summer_red)
    #B_a_red = Q_spt@inv@Q_sp
    interval = int(P.timeSteps/10)
    for jj in range(0,P.timeSteps):
        if jj %interval == 0:
            print("Timestepping, step number: ", jj, "/ ", P.timeSteps)
        
       # y = y.reshape(len(y),1)
                #XnP1 = sparse.linalg.spsolve((R+F), y)
        
        #XnP1 = sparse.csc_matrix(XnP1)
        UnP1A, UnP1B = SourceVector(V, P, C_V, C_P)
        
        if B.shape != (2*P.Nz+2, 2*P.Nz+2):
            print("B is wrong shape")
            sys.exit()
       
        UnP1A = ((Hys[jj]*UnP1A)/P.courantNo)*P.CharImp
        UnP1B = -((Exs[jj]*UnP1B)/P.courantNo)
        UnP1 = np.block([UnP1A, UnP1B])
        
        UnP1 = UnP1.reshape(len(UnP1), 1)
        UnP1_red = UnP1#Q_spt@UnP1
        UnP1_red = UnP1_red.reshape(len(UnP1_red), 1)
       # UnP1_red = sparse.csr_matrix(UnP1_red)
        #B = sparse.csr_matrix(B) #move out of bandsourvect
       
        B_red = V_spt@B
        #B_full_red = np.dot(B_a_red, B_red)
        #B_full_red = sparse.csc_matrix(B_full_red)
        
        
        y = winter_red@XnP1_red +B_red@UnP1_red
        #XnP1 = A@XnP1 + B@UnP1
        #breakpoint()
        y = sparse.csc_matrix(y)
        
        XnP1_red = sparse.linalg.spsolve(summer_red, y)
        XnP1_red = XnP1_red.reshape(len(XnP1_red),1)
        #solve in two steps again.
        #y = y.todense()
        
        #XnP1_red = XnP1_red.reshape(len(XnP1_red), 1)
        #XnP1 = sparse.csc_matrix(XnP1)
        
        #print(XnP1[np.nonzero(XnP1)], "NONZERO")
        #breakpoint()
         
        #XnP1 = XnP1.reshape(len(XnP1), 1)
      #  XnP1 = Q_sp@XnP1_red
        #XnP1 = XnP1.todense()
        
        XnP1 = V_sp@XnP1_red*ratio
       # breakpoint()
        checkNan = np.isnan(np.sum(XnP1))
        if checkNan == True:
            print("NAN FOUND, GET HER OFF THE MILK CARTON")
            breakpoint()
        #tol = 0.3
       #if np.max(XnP1) >= tol:
            # sos =sig.butter(2,0.5, output ='sos')
             #XnP1_0 = sig.sosfiltfilt(sos, XnP1, padlen = XnP1.shape[-1]-2, padtype = 'constant')
            # XnP1_0 =np.poly1d(np.polyfit(np.arange(len(XnP1)), XnP1.ravel(), 11))(np.arange(int(1.1*len(XnP1))))
            # XnP1_0 = XnP1_0[:len(XnP1_0)-int(len(XnP1_0) - len(XnP1))] #np.poly1d(np.polyfit(np.arange(len(V.Ex)), XnP1[:len(V.Ex)].ravel(), 13))(np.arange(len(V.Ex))).reshape(len(V.Ex),1)#
            #XnP1[:30] = 0.0
           # XnP1[-30:]=0.0
           #  XnP1 = XnP1_0*ratio # ONLY FOR VID PURPOSES
       # breakpoint()
        for ii in range(len(V.Ex)-1):
           # breakpoint()
            V.Ex[ii] = XnP1[ii]
         
            V.Hy[ii] = XnP1[ii+len(V.Ex)]
        V.Ex_History[jj] = V.Ex
        
        #XnP1 = sparse.csc_matrix(XnP1)
            
        #Xn = XnP1
        
    return V.Ex, V.Ex_History, V.Hy, UnP1



#PRIMA THEN SPRIM.





def vonNeumannAnalysisMOR(V,P,C_V,C_P, coEfMat):
    #breakpoint()
    if sparse.issparse(coEfMat) == False:
        print("Non-sparse matrix in vonNeumann")
        sys.exit()
    
    print("VonNeumann Analysis for MOR...")
    noOfVals = 3
    stabilityVals = np.zeros(noOfVals)
    simpleRoot = False
    """
    Eigenvalue pertubation, there are no eigenvalues!
    """
    
    counter = 0
   
   # if abs(np.linalg.det(coEfMat)) != 0:    
    tic = time.perf_counter()
    for ii in range(noOfVals):
        stabilityVals[ii]=(abs(sparse.linalg.eigs(coEfMat, which = 'LM', k = 3, tol = 1e-2, maxiter = 100000, return_eigenvectors = False)[ii]))
    toc = time.perf_counter()
    print("time for eigenvals: ", toc-tic)
    for jj in range(noOfVals):
        if np.isclose(stabilityVals[jj], 1, rtol = 1e-4):
            counter +=1
            
    if counter > 1:
        print("simple root present...", stabilityVals)
        simpleRoot = True
    if counter ==1:
        print("single 1 on boundary, unstable", stabilityVals)
        sys.exit()
        
    if np.max(stabilityVals) > 1:
        if simpleRoot == False:     
            print("Stability eigenvalues greater than one, von Neumann instability.", stabilityVals)
            sys.exit()
      
    

# call this in soln denecker before solving equations
def MORViaSPRIM(Q, V_sp, XnP1, R, k, F, P, V, C_V, C_P):  #from Li Xihao and SPRIM FREUND and intro to MOR
    """
    Takes in matrices E and A from LTI state vectors from matrix
    eqns.
    
    Arnoldi process and Krylov space 
    
    include semi-def and definite tests of R and F etc
    """
   # summer = (R+F)
    #winter = (R-F)
    #summer = summer.todense()
   # A = np.linalg.inv(summer)@winter
    #A = sparse.csc_matrix(A)
   # Q = np.ones((num,k+1))
     # Normalisation (sets L2 norm = 1)
    
   
    #Q, H, k = ArnoldiProcess(A, Q, k)
    # H = eigenvalueDecomp(H)
   
   #CONSIDER EXPLICIT RESTART
    blocks = 2
   #SPRIM -> partition V into blocks size of E. H etc, set as diagonal block matrix 
    
    Q1 = V_sp[:int(len(R/2)), :k]
    Q2 = V_sp[int(len(R/2)):, :k]
    #breakpoint()
    Q_sp = np.block( [  [Q1, np.zeros(Q1.shape)], [np.zeros(Q2.shape), Q2]      ])
    
    
   # Xn_red = Q_sp.T@Xn
    #XnP1_red = Q_sp.T@XnP1
    
   #Matrix multiply Vsprim by Xn
   
   #Eigenvalue perturbation -> eigenvalue decomposition 
   # Vec.T diagEigVal Vec 
   # eigs>1 = (gamma * eigVal[i])/(norm(eigVal[i]))
   #where gamma = 0.999
   #Vec.T diagEigs>1 Vec =  H_stab
   # use to create reduced system (Xn+1 = H_stab@Xn + V.T@B@UnP1)
   # result is obtained by V@Xn+1 roughlEq Xn+1 feed into V.Ex, V.Hy
   # solve system for Xn+1 follow eqn through 
    
    return Q, k, Q_sp

def ArnoldiProcess(A, Q, k):
        
    """
    Arnoldi algorithm (Krylov approximation of a matrix)
        input: 
            A: matrix to approximate
            v0: initial vector (should be in matrix form) 
            k: number of Krylov steps 
        output: 
            V: matrix (large, N*k) containing the orthogonal vectors
            H: matrix (small, k*k) containing the Krylov approximation of A
    H = V.T @ A @ V? 
    """
    breakpoint()
    inputType = A.dtype.type
     # Q =  [q1 q2 q3... qn] where q1.n are column vectors
    H = np.asmatrix( np.zeros((k+1,k), dtype=inputType) ) # AmQn = Qn+1Hn
    A = sparse.csc_matrix(A)
    
    nrm = np.linalg.norm(Q[:,0])
    Q[:,0] = Q[:,0]/nrm
    breakpoint()
    if np.isclose(np.linalg.norm(Q[:,0]),1):
       
        for m in range(k-1):
            q = Q[:,m]
            qp1 = A@q # take the column vector and multiply by A
            for j in range(m+1):
                
                H[ j, m] = (Q[:,j].conj().T* qp1 )[0]
                
                qp1 -= H[ j, m] * Q[:,j]
            H[ m+1, m] = np.linalg.norm(qp1)  
            
            tol0 = 1e-12
            if m is not k-1:
                if H[m+1, m] > tol0:
                    
                    Q[:, m+1] = Q[:,m]/(np.linalg.norm(Q[:,m])) #normalise
                    
                else: 
                    return Q, H[:-1,:], k
    else:
        print("Q first col not normalised: ", np.linalg.norm(Q[:,0]))
        sys.exit()
    return Q, H[:-1,:], k

def eigenvalueDecomp(H):
    HeigVals = li.eigh(H, eigvals_only=True)
    #breakpoint()
    HeigVecs = np.asmatrix(np.linalg.eig(H)[1]) #columns are eigenvectors associated with each matching eigenval
   # HeigVecs = sparse.csc_matrix(HeigVecs)
    
    
   # Per,  HeigVecsL, HeigVecsU = li.lu(HeigVecs)
    print("inverting for eigenvalue decomposition")
   # HeigVecsInvL = np.linalg.inv(HeigVecsL)
    #HeigVecsInvU = np.linalg.inv(HeigVecsU)
    #HeigVecsInv = HeigVecsInvU@HeigVecsInvL
    HeigVecsInv=luInvert(HeigVecs)
    print("matrices inverted")
    shrinkFactor = 0.999
    
    
    #Check sparsity, size etc 
    for i in range(len(HeigVals)):
        if abs(HeigVals[i]) > 1:
            HeigVals[i] = (shrinkFactor*HeigVals[i])/abs(HeigVals[i])
    HeigValsDiag = np.diag(HeigVals)

    H1 = HeigVecs@HeigValsDiag
    H = H1@HeigVecsInv
    
    return H
    
def GlobalArnoldi(A, V_a, k):
    
    H = np.asmatrix( np.ones((k+1,k)) ) # AmQn = Qn+1Hn
    A = sparse.csc_matrix(A) # may cause issue
    #fnrm = np.linalg.norm(Q[:,0])
   # breakpoint()
#    Q[:,0]= Q[:,0]/fnrm
    
   #
    V, R = np.linalg.qr(V_a) #R from kr(M,R) is fed in as Q, output as V then Q_sp
   # breakpoint()
   # V = sparse.csc_matrix(V)
    
    for j in range(k):
        
        W = A@V
        print("progress through Arnoldi: ", j, "/", k)
        for i in range(j+1):
            
            H[i,j] = np.trace(V.T.conj()@W)
           
            W -= H[i,j]*V.conj()
            
        #end for i
        tol0 = 1e-12
       # H[j+1, j] =  np.linalg.norm(W)
        if abs(H[j+1, j]) > tol0:
         V, H = np.linalg.qr(W)
         #V = sparse.csc_matrix
        else:
            print("invariant subspace found ", j)
            return V, H[:-1,:], j
    return V, H[:-1,:], j
##
    #############
    ###############
    ################################
    ##############################################
    #################################
    #################
    #########
    #####
    
#make post testerclearer time RANge, INTERVal, start on timestep. ratio fom/rom
def postTests(V, P, ran, interv, ratio, prev = 'no', start =0, windType = 'hanning', stype = 1, chain = False):
    fig, ax = plt.subplots()
    ax.axvspan(P.nzsrc, P.nzsrc+15, alpha = 0.5)
    ax.grid()
    if prev == 'no':
        test = np.zeros((1,len(V.Ex)))
        
    else:
        tester  = np.zeros((1,len(V.Ex)))
        # SECONDARY COUNTER INDEX FOR TEST AS TEST DOESN'T HAVE ALL THE INDICES?
    j = -1   
    for i in range(start, ran, interv):
        j +=1
        if prev == 'no': 
            if stype == 1:
               #breakpoint()
               print('shape of smooth', ((smooth(V.Ex_History[i].ravel(), 150, window = windType)*ratio)).shape)
               test= np.vstack([test, smooth(V.Ex_History[i].ravel(), 150, window = windType)])
              
               ax.plot(test[j])
            if stype == 2: 
                pf= np.polyfit(np.arange(len(V.Ex_History[i])), V.Ex_History[i].ravel(), 7)*ratio
                test =np.vstack([test, np.poly1d(pf)(np.arange(len(V.Ex_History[i])))])
                #breakpoint()
                print("j counter: ",j)
                ax.plot(test[j])
        elif prev != 'no':
            if stype == 1:
                tester = np.vstack([test, smooth(test[j].ravel(), 150, window = windType)*ratio])
                ax.plot(tester[j])
            if stype == 2: #REFACTOR TESTER
               # tester=np.vstack([np.poly1d(np.polyfit(np.arange(len(test[j])))), test[j].ravel(), 7))(np.arange(len(test[j])))])
                ax.plot(tester[j])
    if prev == 'no':       
        quickAnim(test, ran, interv, start)
        return test
    else:
        quickAnim(tester, ran, interv, start)
        return tester
           #CONTINUE building
           
           #post test ideas@:
           #    Animate graphs to see propagation
           #    Some kind of metric to evaluate accuracy, H2, Hinf norms? 




def quickAnim(data, ran, interv, start):
    fig, ax = plt.subplots()
    
    my_path = os.getcwd() 
    newDir = "Post MOR cleaning GIF"
    
    path = os.path.join(my_path, newDir)
    
    try:     ##### BE VERY CAREFUL! THIS DELETES THE FOLDER AT THE PATH DESTINATION!!!!
        shutil.rmtree(path)
    except OSError as e:
        print (e, "Tried to delete folder Error: no such directory exists")
        
        
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed" %path)
    else:
        print ("Successfully created the directory %s " %path)
        
    
    """
    #Next up we iterate through the time steps of Ex_History (an array of arrays containing all y data from each time step)
    #and create a plot for each time step, including the dielectric material. 
    
    #these plots are converted to png files and saved in the new folder in the working directory
    """
    
    for i in range(len(data)):
        print(str.format('{0:.2f}', (100/len(data))*i,"% complete"))
        ax.clear()
        ax.plot(np.real(data[i]))    
        plt.title("MOR clean check")
        #ax.set_xlim(0, P.Nz)
        ax.set_ylim(-2,2)
        plt.savefig(path + "/" + str(i) + ".png")
        image_folder = path
        video_name = 'MOR Check.mp4'
    
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    
        images = natsort.natsorted(images)  # without this python does a weird alphabetical sort that doesn't work
        
    
        frame = cv2.imread(os.path.join(image_folder, images[0]))
        height, width, layers = frame.shape
        framesPerSec =interv
        video = cv2.VideoWriter(video_name, 0, framesPerSec, (width,height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
            #print(image)
        
    cv2.destroyAllWindows()
    video.release()
    plt.close()    
        
    

    """
    #Next we collect all the images in the new directory and sort them numerically, then use OpenCV to create a 24fps video
    """
    
   

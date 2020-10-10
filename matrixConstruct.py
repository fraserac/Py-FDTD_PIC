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
from numba import int32, float32, int64, float64, boolean, complex128
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
import logging
import inspect
numba_logger = logging.getLogger('numba')
numba_logger.setLevel(logging.DEBUG)



specEA= [('H', float64[:,:]), 
         ('V', float64[:,:]), 
         ('vt',float64[:]),
         ('k', int64)]

@jc(specEA)
class EA(object):
    def __init__(self, k, n):
        self.H = np.zeros((k+1)*k).reshape(k+1,k)
        self.V = np.zeros(n*n).reshape(n,n)
        self.vt = np.zeros(n)
        self.k = k
        
forLocals = {'v0': complex128[:,:], 'vt' : complex128[:], 'H':complex128[:,:]}

@numba.jit(nopython = True, locals=forLocals, debug =True)
def ExampleArnoldi(A, v0, k): # Issue with [0,0] 
   
    #EAVars = EA(k, len(v0))
    #A = np.asarray(A, dtype = np.float64)
    v0 = np.ascontiguousarray(v0)
    
    A = np.ascontiguousarray(A)
    H = np.ones((k+1,k))+(0+0j)
    
    
    
    #print(numba.typeof(A))
   # print(numba.typeof(v0))
    # will not partition if k cannot be split into 10 or within bounds
    #'breakpoint()   
   
    split =  np.floor(k/10)  
    count =1    
    vt = np.ones(len(v0))
    vt = np.ascontiguousarray(vt)
   # breakpoint()
    for m in range(k):
        if m%50 ==0:
            print(" this far through arnoldi: ", m, "/", k)
        vt = A@v0[ :, m]
        vt = np.ascontiguousarray(vt)
        #print("progress through Arnoldi: ", m, "/", k)
        for j in range(m+1):
            H[ j, m] = (v0[ :, j].transpose().conj() @ vt )
            
            
            vt -= H[ j, m] * v0[:, j]
        H[ m+1, m] = np.linalg.norm(vt);
        if m is not k-1:
            v0[:,j] =  vt / H[ m+1, m] 
            
        if m == count*split:
            print("inside extra branch...", m)
            qq, rr = np.linalg.qr(v0[:,:m])
            #v0[:,:m] = qq/np.linalg.norm(qq)
            count+=1
    
            
    return v0,  H[:-1,:]

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
    
    UnP1A = np.zeros(P.Nz+1)
    UnP1B = np.zeros(P.Nz+1)
    
    
    return UnP1A, UnP1B

def BGenerator(P):
    B = np.zeros((P.Nz*2+2, P.Nz*2+2))
    B[P.nzsrc, P.nzsrc] = 1
    B[P.Nz+1+P.nzsrc-1, P.nzsrc-1 +P.Nz+1] =1
    print("shape B: ", B.shape)
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
def singularVDPertPreExpan(P, De, Dh, K):
    #check sparseness make sparse
   
    In = np.identity(len(De))
    print("svd pert, Dh, De cond: ", np.linalg.cond(Dh), np.linalg.cond(De))
    
   # breakpoint()
    Des = li.sqrtm(De)
    Dhs = li.sqrtm(Dh)
    DeI = np.linalg.pinv(Des)   # CHECK THIS, MAY NEED IMPROVEMENT
    DhI = np.linalg.pinv(Dhs)
    
    svdRV, svdS, svdLV = np.linalg.svd(DeI@K@DhI)
    gam = 0.9999
    singBound = 2/P.delT
    for i in range(len(svdS)):
        if svdS[i] > (gam*singBound):
            svdS[i]  = gam*singBound
    svdSI = np.diag(svdS)               
    KnearlyPert = svdRV@svdSI@svdLV
    Kpert = Des@KnearlyPert@Dhs
    return Kpert


def blockBuilder(V, P, C_V, C_P):
    K =  np.zeros((P.Nz+1, P.Nz+1))
    DiagE = np.ones(P.Nz+1)
    DiagH = np.ones(P.Nz+1)
    De = np.diag(DiagE)
    Dh = np.diag(DiagH)
    for j in range(1, P.Nz+1):
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

def luInvert(a, name = "Null"):
    #breakpoint()
 
        
    if sparse.issparse(a)!= True:
        a = sparse.csc_matrix(a)
    length = len(a.todense())
    if len(a.todense()) <800:
        M =np.linalg.inv(a.todense())
        check = np.allclose(a@M, np.identity(length), atol = 1e-4)
        if check == False:
            print("not the inverse!", check)
            print((a@M)[0,0])
            #sys.exit() 
        return M
    
    In=np.identity(length)
    test = In
    condi= np.linalg.cond(a.todense())
    print("LUinvert: ", name)
    print("condition of input: ", condi)
    if condi > 5:
        print("condition is bad! Attempting jacobian preconditioner... ")
        del test
        test = np.linalg.inv(np.diag(np.diag(a.todense())))
        result = np.linalg.cond(test@a.todense())  # try further preconditioners
        print("result: ", result)
        if result<condi:
            a = test@a
            
       
    if sparse.issparse(a)!= True:
       a = sparse.csc_matrix(a)   
    
    M = sparse.linalg.spilu(a)
    
    M = M.solve(In)
    print("SPiLU completed")
   
    #M = sparse.csc_matrix(M)
    if sparse.issparse(a) ==True:
        a = a.todense()
    check = np.allclose(a@M, np.identity(length), atol = 1e-3)
   # breakpoint()
    if check == False:
        print("not the inverse! from SPLU", check)
        print((a@M)[0,0], " First element of product")
        M = np.linalg.pinv(a)
    #breakpoint()
    if np.allclose(M@a, np.identity(len(a))):
        
        print("not the inverse! from pinv", check)
        print((a@M)[0,0], " First element of product")
       #sys.exit() 
        
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
    
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    
    return y[:len(x)]

def checkNan(vals, loc, loopLoc =0):#
    if sparse.issparse(vals):
        vals = vals.todense()
    checkNan = np.isnan(np.sum(vals))
    if checkNan == True:
        print("nan found at line ", loc, "iteration: ", loopLoc)
        sys.exit()
    

def solnDenecker(R, F, A, UnP1A, UnP1B, Xn, V, P, C_V, C_P, Kop, Kopt, De, Dh, pulseInit = True):
    checkLine = "inspect.currentframe().f_back.f_lineno"  # I use this in an eval expression to write the current code line into a
    #console print out when debugging 
    
    
    
    delayMOR = P.delayMOR
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
    k = int(len(Xn)*0.4)
    XnP1_red = np.zeros(k)
   
    XnP1 = XnP1.reshape(len(XnP1), 1)
    XnP1_red = XnP1_red.reshape(len(XnP1_red),1)
    ratio = len(XnP1)/k
    
    
    #Q, H, k, Xn_red, XnP1_red, Q_sp = MORViaSPRIM(Xn, XnP1, 2*P.Nz+2,R, k, F, P, V, C_V, C_P)
    summer = (R+F)
    winter = (R-F)
    summerInv = luInvert(summer)
    ### CONDITION NUMBER EFFECTS ACCURACY OF INVERSION!
    print("conditions numbers of summer, winter: ", np.linalg.cond(summer.todense()), np.linalg.cond(winter.todense()))
   
    #summer = vonNeumannAnalysisMOR(V, P, C_V, C_P, summer, "summer")
    #winter = vonNeumannAnalysisMOR(V, P, C_V, C_P, winter, "winter")
  
    
    #summer = sparse.csc_matrix(summer)
    if(sparse.issparse(winter)):
        winter = winter.todense()
    winter = np.asmatrix(winter)
    
   
    B = BGenerator(P)
    B = sparse.csc_matrix(B)
   
    In = np.identity(len(winter))
    M = In
    M = sparse.csc_matrix(M)
    M2 = In
    M2 = sparse.csc_matrix(M2)
    
    s0 = 2*np.pi*P.freq_in*(1j)
    
    
    M = luInvert(s0*In-summerInv@winter, "M")
    #M2 = luInvert((s0*In -winter).conj().T, "M2")  
    if sparse.issparse(M):             
        M = M.todense()  
    #M = np.asarray(M, dtype = np.complex128)
    #M2 = M2.todense()
    #M2 = np.asarray(M2, dtype = np.complex128)
   
         
    """
    for jj in range(int(P.freq_in),int(P.freq_in)):
        s0 = 2*np.pi*jj
        M2= M2@(luInvert((s0*In).conj().T-winter.conj().T))
    
    M2 = M2.todense()
    M2 = np.asarray(M2, dtype = np.float64)
    """
    v0 = np.random.randint(5, size=(len(winter),k))#*(1+1j)
    v0 = v0.astype(np.complex128) 
    v2 = np.random.randint(5, size=(len(winter),k))#*(1+1j)
   # v2 = v2.astype(np.complex128)
    v0, RR = np.linalg.qr(v0)   # if matix is complex returns unitary, normalise it...
    #v0 = v0/np.linalg.norm(v0)
    print(type(v0))
    #print(type(v2))
    #v2, R2 = np.linalg.qr(v2)
    #v2 = v2/np.linalg.norm(v2)
   # v0 = M@v0   # ?
    #v2 = M2@v2
   # EAVars = EA(k, len(v0))
    #breakpoint()
    print("Starting Arnoldi process................")
    tic = time.perf_counter()
    V_sp, RR = ExampleArnoldi(M, v0, k)
    #breakpoint()
    #V_sp, RR = np.linalg.qr(V_sp)
    orth = V_sp.T.conj()@V_sp
    check = np.allclose(orth, np.identity(len(orth)), atol = 1e-2)
    print(check, " = orthornormal check.")
   
    if check == False:
        
        print("first ten diagonal terms of orth:", np.diag(orth)[:10])
        print(np.linalg.norm(V_sp), " norm of V_sp")
    
    
    #V_sp, R = np.linalg.qr(V_sp)
    #V_sp = V_sp/np.linalg.norm(V_sp)
   
    print(np.linalg.matrix_rank(V_sp, tol =1e-4),"/", V_sp.shape[1], "MATRIX RANK ")
   
    V_sp = np.block([np.real(V_sp[:int(len(V_sp)/2)]), np.imag(V_sp[int(len(V_sp)/2):])])
    toc = time.perf_counter()
    print("FINISHED ARNOLDI in: ",toc-tic)
    V_sp_UL = V_sp[:int(len(V_sp)/2),:k]
    V_sp_UR = np.zeros((int(len(V_sp)/2), k))
    V_sp_LL = np.zeros((int(len(V_sp)/2), k))
    V_sp_LR = V_sp[int(len(V_sp)/2):,:k]
    V_sp = np.block([[V_sp_UL, V_sp_UR],[V_sp_LL, V_sp_LR]])
    
    if V_sp_UL.shape !=V_sp_UR.shape or V_sp_LL.shape != V_sp_LR.shape or V_sp_LL.shape != V_sp_UR .shape:
        print("shapes of blocks for SPRIM partition not the same: ", V_sp_UL.shape, V_sp_UR.shape, V_sp_LL.shape, V_sp_LR.shape)
        sys.exit()
    #
    """
    bob = V_sp.T@summer@V_sp
    bob2 = V_sp@bob@V_sp.T
    avgDiag = np.average(np.diag(bob2))
    for i in range(bob2.shape[0]):
        for j in range(bob2.shape[1]):
            if abs(bob2[i,j]) <= abs(avgDiag)/5:
                bob2[i,j] = 0
                
    factor = np.average(np.diag(summer.todense()) /np.average(np.diag(bob2)))  
    bob2 = bob2*factor
    bob3 = np.real(bob2)
  #  breakpoint()
    
    
    V_sp = np.hstack((np.real(V_sp), np.imag(V_sp)))
   # modify = 1/V_sp[0,0]
    print("V_sp: ", V_sp[:4,:4])
   # V_sp*= modify
    #W_sp = np.hstack((np.real(W_sp), np.imag(W_sp)))
    
    V_sp_UL = V_sp[:int(len(V_sp)/2),:k]
    V_sp_UR = np.zeros((int(len(V_sp)/2), k))
    V_sp_LL = np.zeros((int(len(V_sp)/2), k))
    V_sp_LR = V_sp[int(len(V_sp)/2):,:k]
    V_sp = np.block([[V_sp_UL, V_sp_UR],[V_sp_LL, V_sp_LR]])
    
    if V_sp_UL.shape !=V_sp_UR.shape or V_sp_LL.shape != V_sp_LR.shape or V_sp_LL.shape != V_sp_UR .shape:
        print("shapes of blocks for SPRIM partition not the same: ", V_sp_UL.shape, V_sp_UR.shape, V_sp_LL.shape, V_sp_LR.shape)
        sys.exit()
    if V_sp.shape[0] != P.Nz+2:
        print(V_sp.shape, " V_sp shape", P.Nz, " P.Nz")
   
    #check structure preservation
    
    # take top half of V_sp and W_sp
    print("Dimensions of V_sp should match Xn, Shape V vs Xn: ", V_sp.shape, Xn.shape)
    #breakpoint()
    #SPRIM PARTITION:
    #V_sp = 
    
   # breakpoint()
    #V_sp, R = np.linalg.qr(V_sp)#

   # print("Performing Orthonormal tests")
   # check = np.allclose(V_sp.T @ V_sp, np.eye(V_sp.shape[1]), rtol =1e-1)
   # print(" ortho check: ", check)
    """  
    V_spt = V_sp.T
   # breakpoint()
    
    
    
    
   # M = sparse.csc_matrix(M)
    #A_red = M#Q_spt@M@Q_sp
    
    #winter_red = V_sp.T@winter@V_sp
    #summer_red = V_sp.T@summer@V_sp
    #print("condition numbers before preconditioning: ", np.linalg.cond(summer_red), np.linalg.cond(winter_red))
    Vnorm = np.linalg.norm(V_sp)
    V_sp, rrr = np.linalg.qr(V_sp)
    V_sp *= Vnorm
    R_red = V_sp.T@R@V_sp
    F_red = V_sp.T@F@V_sp
    Kop_red_R = R_red[:int(len(R_red)/2),int(len(R_red)/2):]
    De_red_R = R_red[:int(len(R_red)/2),:int(len(R_red)/2)]
    Dh_red_R = R_red[int(len(R_red)/2):,int(len(R_red)/2):]#V_sp_LR.T@Dh@V_sp_LR
    
    Kop_red_pert = singularVDPertPreExpan(P, De_red_R, Dh_red_R, Kop_red_R)
    R_red[len(Kop_red_R):,:len(Kop_red_R)]= Kop_red_pert.T
    R_red[:len(Kop_red_R),len(Kop_red_R):] = Kop_red_pert
  #  R_red[]
    # structure preservation preserves algebraic operations?
    
    
    
    XnP1_red =  V_sp.T@XnP1
    winter_red = R_red-F_red
    summer_red = R_red + F_red
    summerJac = np.linalg.pinv(np.diag(np.diag(summer_red)))
    print("shape winter: ", winter.shape)
    #sumJac = np.linalg.inv(np.diag(np.diag(summer_red)))
    ## summer_red is badly conditioned?
    #winter_red = V_spt@winter@V_sp
    #summer_red = V_spt@summer@V_sp
    #A_red = sparse.csc_matrix(A_red)
    #if (sparse.issparse(A_red)):
      #  A_red = A_red.todense()
    #print("Eigenvalue decomp...")
    #breakpoint()
    #winter_red= np.real(eigenvalueDecomp(winter_red)) 
    #summer_red  =np.real(eigenvalueDecomp(summer_red))
    
    #AA, BB, CC, DD= eigenvalueDecomp(summer_red)
    # Inv AA @ summer_red should give I?
    #summer_red = AA@summer_red
    #AA, BB, CC, DD= eigenvalueDecomp(winter_red)
    #winter_red = AA@winter_red
    #check shapes of projection matrices
    #print("condition numbers reduced after preconditioning: ", np.linalg.cond(sumJac@summer_red), np.linalg.cond(winter_red))
    #print("V_spt shape: ", V_spt.shape)
   # breakpoint() 
    
    winter_red = sparse.csc_matrix(winter_red)
  #  summer_red_inv = luInvert(summer_red, "summer_red")
   # summer_red_inv_spar = sparse.csc_matrix(summer_red_inv)
    summer_red_sparse = sparse.csc_matrix(summerJac@summer_red)
   # comboBreaker = summer_red_inv_spar@winter_red
   ## H, HeigValsDiag, HeigVecsInv, HeigVecs = eigenvalueDecomp(comboBreaker)
   # summer_inv_Full = luInvert(summer)
    #comboBreakerFull =summer_inv_Full@winter
    #HF, HeigValsDiagF, HeigVecsInvF, HeigVecsF = eigenvalueDecomp(comboBreakerFull)
    
   # A_red = summer_red_inv@winter_red
   # summer_red = sparse.csc_matrix(summer_red)
    #B_a_red = Q_spt@inv@Q_sp
    interval = int(P.timeSteps/10)#
    
    
   # sos = sig.butter(21, 0.01, output='sos')
    
    for jj in range(0,P.timeSteps):
        if jj %interval == 0:
            print("Timestepping, step number: ", jj, "/ ", P.timeSteps)
        
       # y = y.reshape(len(y),1)
                #XnP1 = sparse.linalg.spsolve((R+F), y)
        
        #XnP1 = sparse.csc_matrix(XnP1)
        UnP1A, UnP1B = SourceVector(V, P, C_V, C_P)
        
        #if B.shape != (2*P.Nz+2, 2*P.Nz+2):
        #    print("B is wrong shape")
           # sys.exit()
        
       # UnP1A[:] =  ((Hys[jj])/P.courantNo)*P.CharImp
        UnP1B[:] = ((Exs[jj])/P.courantNo)
        UnP1 = np.block([UnP1A, UnP1B])
        
        UnP1 = UnP1.reshape(len(UnP1), 1)
        UnP1_red = V_sp.T@UnP1#Q_spt@UnP1
        UnP1_red = UnP1_red.reshape(len(UnP1_red), 1)
       # UnP1_red = sparse.csr_matrix(UnP1_red)
        #B = sparse.csr_matrix(B) #move out of bandsourvect
        print(B.shape)
       # breakpoint()
        B_red = V_sp.T@B@V_sp
        B_red = sparse.csc_matrix(B_red)
       
        if (pulseInit == False) or (pulseInit == True and jj >= delayMOR):
        
            #breakpoint()
            #print("shape of B_red: ", B_red.shape)
            
           # breakpoint()
            XnP1_red = XnP1_red.reshape(len(XnP1_red), 1)
            XnP1_red = sparse.csc_matrix(XnP1_red)
            #XnP1_red = summer_red_inv_spar@(winter_red@XnP1_red +B_red@UnP1_red)
            #breakpoint()
            y1 = winter_red@XnP1_red +  B_red@UnP1_red
           #y1Full = V_spt@y1*100
            
            #print(np.average(np.abs(y1)), " y1")
           # print("y shape if first before solve: " ,y.shape)
            #XnP1 = A@XnP1 + B@UnP1
            #breakpoint()
            y1 = sparse.csr_matrix(y1)
           # breakpoint()
           # XnP1_red = summer_red_inv@y1
           
            #breakpoint()
            Spla = sparse.linalg.spilu(summer_red_sparse)
            lam = lambda x: Spla.solve(x)
            if sparse.issparse(summer_red):
                summer_red = summer_red.todense()
            MM= sparse.linalg.LinearOperator(summer_red.shape, lam)
           # print("shapes of y1, summer_red: ", y1.shape, summer_red.shape)
            
            
            XnP1_red, info = sparse.linalg.gmres(summer_red_sparse, summerJac@y1.todense(), x0=XnP1_red.todense(), atol=1e-4, M = MM, maxiter = 250)
            if info != 0:
                print("iterations: ", info)
            #fig, ax = plt.subplots()
           
            XnP1 = (V_sp@XnP1_red)
           
            #breakpoint()
           # XnP1 = sig.sosfiltfilt(sos, smooth(np.asarray([XnP1]).ravel(),1000, window = 'flat'))
            
            
           # print("Avg, min, max: ", np.average(np.abs(XnP1)), np.min(np.abs(XnP1)), np.max(np.abs(XnP1)))   # np.min should return zero especially early on
           # print("SUM OFFFFFFFF: ", np.sum(XnP1))
            checkNan(XnP1, eval(checkLine), jj)
            if (np.max(np.abs(XnP1))*ratio <= 1e-8) and jj >=30: 
                print(np.average(np.abs(XnP1)), " Average val of XnP1", eval(checkLine))
                sys.exit()
            if (np.average(np.abs(XnP1)) >= 1e2):
                print(np.average(np.abs(XnP1)), " Average val of XnP1", eval(checkLine))
                plt.plot(XnP1)
                sys.exit()
            #print("first if")
             # check dimensions v_sp etc
            
            if jj%25 ==0:
                print("Iteration of timeStep in MOR: ", jj, "/ ", P.timeSteps)
            
            
            #XnP1 = (V_sp@XnP1_red)
            #XnP1 = sig.sosfiltfilt(sos, smooth(np.asarray([XnP1]).ravel(),50, window = 'flat'))
            #XnP1 = XnP1*ratio**2
            XnP1 = XnP1.reshape(len(XnP1),1)
            
            XnP1_fin = np.real(XnP1)
           
           # breakpoint()
            
        elif pulseInit == True and jj < delayMOR:
            
            print("FOM pulse init stage: ", jj)
            XnP1 = sparse.csc_matrix(XnP1)
            y = winter@XnP1 +B@UnP1
           # XnP1 = HeigValsDiagF@XnP1 +  HeigVecsInvF@summer_inv_Full@B@UnP1
            #breakpoint()
            if sparse.issparse(y):
                y = y.todense()
            #XnP1 = A@XnP1 + B@UnP1
            #breakpoint()
            #y = sparse.csc_matrix(y)
            #print("y ", np.average(np.abs(y.todense())))
           # print("second elif")
            #XnP1 = sparse.linalg.spsolve(summer, y)
            Spla1 = sparse.linalg.spilu(sparse.csc_matrix(summer))
            lam1 = lambda x: Spla1.solve(x)
            if sparse.issparse(summer):
                summer = summer.todense()
            MM1= sparse.linalg.LinearOperator(summer.shape, lam1)
           # print("shapes of y1, summer_red: ", y1.shape, summer_red.shape)
            
            
            XnP1, info = sparse.linalg.lgmres(summer, y, x0=XnP1.todense(), atol=1e-4, M = MM1, maxiter = 250)
            if info != 0:
                print("iterations: ", info)
            checkNan(XnP1, eval(checkLine), jj)
            #print("Avg, min, max: ", np.average(np.abs(XnP1)), np.min(np.abs(XnP1)), np.max(np.abs(XnP1)))
            XnP1_red =  V_sp.T@XnP1
            #print("Avg, max XnP1_red: ", np.average(np.abs(XnP1_red)), np.max(abs(XnP1_red)))
           # print("SUM OF XnP1 FOM: ", np.sum(XnP1))
            #XnP1 = HeigVecsF@XnP1
           # XnP1 = sig.sosfiltfilt(sos, smooth(np.asarray([XnP1]).ravel(),1000, window = 'flat'))
            XnP1 = XnP1.reshape(len(XnP1),1)
            if (np.average(np.abs(XnP1)) >= 1e2):
               print(np.average(np.abs(XnP1)), " Average val of XnP1", eval(checkLine))
               plt.plot(XnP1)
               sys.exit()
            XnP1_fin = np.real(XnP1)
           
            
          #  Add in float var to move np.real XnP1 to prevent stdout discarding imag
            
        for ii in range(len(V.Ex)-1):
            if sparse.issparse(XnP1):
                XnP1_fin= XnP1_fin.todense()
              # maybe don't discard imaginary part, use just real part for vid
            #breakpoint()
            V.Ex[ii] = XnP1_fin[ii][0]*ratio
         
            V.Hy[ii] = XnP1_fin[ii+len(V.Ex)][0]
        V.Ex_History[jj] = V.Ex
        
        
    return V.Ex, V.Ex_History, V.Hy, UnP1



#PRIMA THEN SPRIM.





def vonNeumannAnalysisMOR(V,P,C_V,C_P, coEfMat, name="Not provided"):
    #breakpoint()
    if sparse.issparse(coEfMat) == False:
        coEfMat = sparse.csc_matrix(coEfMat)
    #breakpoint()
    print("VonNeumann Analysis for MOR...", name)
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
    print(name, " contains ", counter,  " unstable eigenvalues, perturbing..." )
    """      
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
    """
    if counter >= 1:
        H, HeigValsDiag, HeigVecsInv, HeigVecs = eigenvalueDecomp(coEfMat)
        coEfMat = H
    counter =0
    coEfMat = sparse.csc_matrix(coEfMat)
    print("checking if eigenvalues stabilised: ")
    for ii in range(noOfVals):
        stabilityVals[ii]=(abs(sparse.linalg.eigs(coEfMat, which = 'LM', k = 3, tol = 1e-2, maxiter = 100000, return_eigenvectors = False)[ii]))
    toc = time.perf_counter()
    print("time for eigenvals: ", toc-tic)
    for jj in range(noOfVals):
        if np.isclose(stabilityVals[jj], 1, rtol = 1e-4):
            counter +=1
    print(counter, " eigenvalues >= 1 remaining")
    return  coEfMat
    

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

def eigenvalueDecomp(H):   # takes in reduced matrix

    if sparse.issparse(H): 
        H = H.todense()
    if np.iscomplex(H.any()):
        print("Input to Eigval decomp is complex.")
    HeigVals = li.eigvals(H)
    #breakpoint()
    HeigVecs =li.eig(H)[1] #default right vec, columns are eigenvectors associated with each matching eigenval
   # HeigVecs = sparse.csc_matrix(HeigVecs)
    #HeigVecs, R = np.linalg.qr(HeigVecs)
    
    print("condition number of HeigVecs: ", np.linalg.cond(HeigVecs))
    
    #Per,  HeigVecsL, HeigVecsU = li.lu(HeigVecs)
    print("inverting for eigenvalue decomposition")
   # HeigVecsInvL = np.linalg.inv(HeigVecsL)
    #HeigVecsInvU = np.linalg.inv(HeigVecsU)
   # HeigVecsInv = HeigVecsInvU@HeigVecsInvL
    #HeigVecsInv =sparse.linalg.spsolve()
    
    HeigVecsInv=luInvert(HeigVecs, " eigenval decomp")
    print("matrices inverted")
    check = np.allclose(HeigVecsInv@HeigVecs, np.identity(len(HeigVecs)), atol = 1e-2)
    if check ==False:
        print("Inversion incorrect! Using pseudoinverse")
    HeigVecsInv = np.linalg.pinv(HeigVecs)
    shrinkFactor = 0.999
    
    
    #Check sparsity, size etc 
    for i in range(len(HeigVals)):
        if abs(HeigVals[i]) >= 1:
            HeigVals[i] = (shrinkFactor*HeigVals[i])/abs(HeigVals[i])
    HeigValsDiag = np.diag(HeigVals)

    H1 = HeigVecs@HeigValsDiag
    H = H1@HeigVecsInv
    #HVecs @ Hvalspert @HVecs^-1#
    return H, HeigValsDiag, HeigVecsInv, HeigVecs
    
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
def postTests(V, P, ranTIME, interv, ratio, prev = 'no', start =0, windType = 'hanning', stype = 1, chain = False):
    fig, ax = plt.subplots()
    ax.axvspan(P.nzsrc, P.nzsrc+15, alpha = 0.5)
    ax.grid()
    if prev == 'no':
        test = np.zeros((1,len(V.Ex)))   ## BLANK SPACE AS WE ARE APPENDING ONTO THIS WITH VSTACK
        for hh in range(P.delayMOR):
            test = np.vstack([test, V.Ex_History[hh]*ratio ])  ## pulseInit constructor?
        
    else:
        tester  = np.zeros((1,len(V.EX)))
        # SECONDARY COUNTER INDEX FOR TEST AS TEST DOESN'T HAVE ALL THE INDICES?
        
    
    j = -1   
    for i in range(start+ P.delayMOR, ranTIME,  interv):
        j +=1
        if prev == 'no': 
            if stype == 1:
              # breakpoint()
               print(i,j)
              
               print('shape of smooth', ((smooth(V.Ex_History[i].ravel(), 150, window = windType)*ratio)).shape)
              
               smoothVal =  smooth(V.Ex_History[i].ravel()*ratio, 150, window = windType)
               
               test= np.vstack([test, smoothVal*ratio])
              
               ax.plot(test[j])
            if stype == 2: 
                pf= np.polyfit(np.arange(len(V.Ex_History[i])), V.Ex_History[i].ravel(), 21)*ratio
                test =np.vstack([test, np.poly1d(pf)(np.arange(len(V.Ex_History[i])))])
                #breakpoint()
                #print("j counter: ",j)
                ax.plot(test[j])
            if stype == 3: 
                b, a = sig.butter(8, 0.125)
                sigClean = smooth((V.Ex_History[i]*ratio).ravel(), 150, window = windType)
                test = np.vstack([test, sig.filtfilt(b,a, sigClean*ratio, method = 'gust')])
                
        elif prev != 'no':
            if stype == 1:
                tester = np.vstack([test, smooth(test[j].ravel(), 150, window = windType)*ratio])
                ax.plot(tester[j])
            if stype == 2: #REFACTOR TESTER
               # tester=np.vstack([np.poly1d(np.polyfit(np.arange(len(test[j])))), test[j].ravel(), 7))(np.arange(len(test[j])))])
                ax.plot(tester[j])
    if prev == 'no':       
        quickAnim(test, ranTIME, interv, start)
        return test
    else:
        tester = quickAnim(tester, ranTIME, interv, start)
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
        #ax.set_ylim(-2,2)
        plt.savefig(path + "/" + str(i) + ".png")
        image_folder = path
        video_name = 'MOR Check.avi'
    
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
    
   

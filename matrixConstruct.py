# -*- coding: utf-8 -*-
"""
Creating logic for matrix construction for MOR methods
Using vectorise and ogrid 
"""


import numpy as np
import time 
from numba import njit as nj
from scipy import sparse
import BaseFDTD11 as BF
import sys


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

def BAndSourceVector(V,P, C_V, C_P):
    B = np.zeros((P.Nz*2+2, P.Nz*2+2))
    UnP1A = np.ones(P.Nz+1)
    UnP1B = np.ones(P.Nz+1)
    B[P.nzsrc, P.nzsrc] = 1
    B[P.Nz+1+P.nzsrc-1, P.nzsrc-1 +P.Nz+1] =1
    
    return UnP1A, UnP1B, B
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
        UnP1 = sparse.csc_matrix(UnP1)
        B = sparse.csc_matrix(B)
        Xn = sparse.csc_matrix(Xn)
        XnP1 = np.zeros(2*P.Nz+2)
        XnP1 = sparse.csc_matrix(XnP1)
        if jj == 0:
            XnP1 = A@Xn.T
            XnP1 += B@UnP1.T
        elif jj >0:
            XnP1 = A@XnP1.T 
            XnP1 += B@UnP1.T
        
        XnP1 = XnP1.todense()    
        for ii in range(len(V.Ex)):
            V.Ex[ii] = XnP1[ii]
            V.Hy[ii] = XnP1[ii+len(V.Ex)]
        V.Ex_History[jj] = V.Ex
        
        Xn = XnP1
        
    return B, V.Ex, V.Ex_History, V.Hy, UnP1, Xn




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
    
    XnP1 =sparse.csc_matrix(XnP1)
    Xn = sparse.csc_matrix(Xn)
    R = sparse.csc_matrix(R)
    F = sparse.csc_matrix(F)
    for jj in range(0,P.timeSteps):
        print("Timestepping, step number: ", jj)
        
       
       # y = y.reshape(len(y),1)
                #XnP1 = sparse.linalg.spsolve((R+F), y)
        
        #XnP1 = sparse.csc_matrix(XnP1)
        UnP1A, UnP1B, B = BAndSourceVector(V, P, C_V, C_P)
        if B.shape != (2*P.Nz+2, 2*P.Nz+2):
            print("B is wrong shape")
            sys.exit()
       
        UnP1A = (Hys[jj]*UnP1A*7*P.Nlam)
        UnP1B = -(Exs[jj]*UnP1B*7*P.Nlam)/P.CharImp
        UnP1 = np.block([UnP1A, UnP1B])
        UnP1 = UnP1.reshape(len(UnP1), 1)
        UnP1 = sparse.csr_matrix(UnP1)
        B = sparse.csr_matrix(B)
        #breakpoint()
        
        y = (R-F)@XnP1 +B@UnP1
        #XnP1 = A@XnP1 + B@UnP1
        #breakpoint()
        
        #R= R.todense()
        #F = F.todense()
        #XnP1 = XnP1.todense()
        summer = R+F
        #y = y.todense()
        
        XnP1 = sparse.linalg.spsolve(summer, y )  # outputs NONSPARSE
        XnP1 = XnP1.reshape(len(XnP1), 1)
        #XnP1 = sparse.csc_matrix(XnP1)
        
        #print(XnP1[np.nonzero(XnP1)], "NONZERO")
        #breakpoint()
         
        #XnP1 = XnP1.reshape(len(XnP1), 1)
        
        for ii in range(len(V.Ex)-1):
            V.Ex[ii] = XnP1[ii]
            V.Hy[ii] = XnP1[ii+len(V.Ex)]
        V.Ex_History[jj] = V.Ex
        
        XnP1 = sparse.csc_matrix(XnP1)
            
        #Xn = XnP1
        
    return V.Ex, V.Ex_History, V.Hy, UnP1









def vonNeumannAnalysisMOR(V,P,C_V,C_P, coEfMat):
    print("VonNeumann Analysis for MOR...")
    stabilityVals = np.zeros(len(sparse.linalg.eigs(coEfMat,  k = 15, ncv = 60, return_eigenvectors = False)))
    simpleRoot = False
    """
    Eigenvalue pertubation, there are no eigenvalues!
    """
    
    counter = 0
   
   # if abs(np.linalg.det(coEfMat)) != 0:    
    
    for ii in range(len(sparse.linalg.eigs(coEfMat, which = 'LM', k = 15, ncv = 80, return_eigenvectors = False))):
        stabilityVals[ii]=(abs(sparse.linalg.eigs(coEfMat, which = 'LM', k = 15, ncv = 80, return_eigenvectors = False)[ii]))
        
    for jj in range(len(stabilityVals)):
        if np.isclose(stabilityVals[jj], 1, rtol = 1e-8):
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
      
    



#x = [(0,0), (1,1)]
#print(A[tuple(x)])

#@nj

#def testBuild(i, j, A):
    
    
 #   return j

#ticc = time.perf_counter()

#ff = testBuild(2,0,A)
#tocc =  time.perf_counter()


#print(ff)

#print("completed loop in: ", tocc-ticc)
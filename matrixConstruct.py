# -*- coding: utf-8 -*-
"""
Creating logic for matrix construction for MOR methods
Using vectorise and ogrid 
"""


import numpy as np
import time 
from numba import njit as nj
from scipy import linalg as li
from scipy import sparse
import BaseFDTD11 as BF
import sys
from sklearn.linear_model import Ridge

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
    
    #XnP1 =sparse.csc_matrix(XnP1)
    #Xn = sparse.csc_matrix(Xn)
    R = sparse.csc_matrix(R)
    F = sparse.csc_matrix(F)
    k = 100 # verify too big or small? A priori if better
    Q, H, v0, k = MORViaSPRIM(Xn, XnP1, 2*P.Nz+2,R, k, F, P, V, C_V, C_P)
    for jj in range(0,P.timeSteps):
        print("Timestepping, step number: ", jj, "/ ", P.timeSteps)
        
       
       # y = y.reshape(len(y),1)
                #XnP1 = sparse.linalg.spsolve((R+F), y)
        
        #XnP1 = sparse.csc_matrix(XnP1)
        UnP1A, UnP1B, B = BAndSourceVector(V, P, C_V, C_P)
        if B.shape != (2*P.Nz+2, 2*P.Nz+2):
            print("B is wrong shape")
            sys.exit()
       
        UnP1A = (Hys[jj]*UnP1A*P.Nlam)/P.courantNo
        UnP1B = -((Exs[jj]*UnP1B*P.Nlam)/P.CharImp)/P.courantNo
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
def MORViaSPRIM(Xn, XnP1, n, R, k, F, P, V, C_V, C_P):  #from Li Xihao and SPRIM FREUND and intro to MOR
    """
    Takes in matrices E and A from LTI state vectors from matrix
    eqns.
    
    Arnoldi process and Krylov space 
    
    include semi-def and definite tests of R and F etc
    """
    summer = (R+F)
    winter = (R-F)
    summer = summer.todense()
    A = np.linalg.inv(summer)@winter
    #A = sparse.csc_matrix(A)
    Q = np.ones((n,k+1))
     # Normalisation (sets L2 norm = 1)
    
   
    Q, H, k = ArnoldiProcess(A, Q, k)
    H = eigenvalueDecomp(H)
    
   #CONSIDER EXPLICIT RESTART
   
   #SPRIM -> partition V into blocks size of E. H etc, set as diagonal block matrix 
    breakpoint()
    Q1 = Q[:n, :n]
    Q2 = Q[n+1:, n+1:]
    Q_sp = np.kron(Q1, Q2)
    breakpoint()
    Xn_red = Q_sp@Xn
    XnP1_red = Q_sp@XnP1
    breakpoint()
   #Matrix multiply Vsprim by Xn
   
   #Eigenvalue perturbation -> eigenvalue decomposition 
   # Vec.T diagEigVal Vec 
   # eigs>1 = (gamma * eigVal[i])/(norm(eigVal[i]))
   #where gamma = 0.999
   #Vec.T diagEigs>1 Vec =  H_stab
   # use to create reduced system (Xn+1 = H_stab@Xn + V.T@B@UnP1)
   # result is obtained by V@Xn+1 roughlEq Xn+1 feed into V.Ex, V.Hy
   # solve system for Xn+1 follow eqn through 
    
    return Q, H, v0, k

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
    
    inputType = A.dtype.type
     # Q =  [q1 q2 q3... qn] where q1.n are column vectors
    H = np.asmatrix( np.zeros((k+1,k), dtype=inputType) ) # AmQn = Qn+1Hn
    A = sparse.csc_matrix(A)
    Q[:,0] = Q[:,0]/np.linalg.norm(Q[:,0])
    
    
    for m in range(k):
        q = Q[:,m]
        qp1 = A@q # take the column vector and multiply by A
        for j in range( m+1):
            
            H[ j, m] = (q.T.conj()* qp1 )[0]
            
            qp1 -= H[ j, m] * q
        H[ m+1, m] = np.linalg.norm(qp1)  
        tol0 = 1e-12
        if m is not k-1:
            if H[m+1, m] > tol0:
                
                Q[:, m+1] = Q[:,m]/H[m+1,m]  #normalise
            else: 
                return Q, H[:-1, :], k
    return Q, H[:-1,:], k

def eigenvalueDecomp(H):
    HeigVals = np.linalg.eigvals(H)
    HeigVecs = np.asmatrix(np.linalg.eig(H)[1]) #columns are eigenvectors associated with each matching eigenval
    
    HeigValsDiag = np.diag(HeigVals)
    HeigVecsInv = np.linalg.inv(HeigVecs)
    shrinkFactor = 0.99
    
    
    #Check sparsity, size etc 
    for i in range(len(HeigVals)):
        if abs(HeigVals[i]) > 1:
            HeigVals[i] = (shrinkFactor*HeigVals[i])/abs(HeigVals[i])

    H = HeigVecsInv@HeigValsDiag@HeigVecs
    
    return H
    
    

#ticc = time.perf_counter()

#ff = testBuild(2,0,A)
#tocc =  time.perf_counter()


#print(ff)

#print("completed loop in: ", tocc-ticc)
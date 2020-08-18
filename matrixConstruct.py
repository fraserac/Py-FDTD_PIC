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
    #R = sparse.csc_matrix(R)
    #F = sparse.csc_matrix(F)
    k = 1000 # verify too big or small? A priori if better
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
    Per, stenL, stenU = li.lu(winter)
    invStenL = np.linalg.inv(stenL)
    invStenU = np.linalg.inv(stenU)
    invSten = invStenU@invStenL
    invSten = sparse.csc_matrix(invSten)
    M = invSten@summer #multiply summer by inverse of LU decomp
    B = BGenerator(P)
    B = sparse.csc_matrix(B)
   # Q = invSten@B
    V_a = np.random.randint(5, size=(len(winter), k)) # redundant
  
    #Q = Q.astype(float)
    #M = M.astype(float)
    
    breakpoint()
    #M = (S0(summer)-winter)^-1@summer, R = (s0((summer)-winter)^-1@B)
    #LU DECOMP OF M 
    #inv = sparse.csc_matrix(inv)
    
    
    
    #A = sparse.csc_matrix(A)
    #Q = np.random.randint(5, size=(2*P.Nz+2,k+1))
    #Q= Q.astype(float)
    
    #if(sparse.issparse(M)):
     #   M = M.todense()
    print("Starting Arnoldi process")
    V_sp, H, k = GlobalArnoldi(M, V_a, k)
   
    #V_sp, R = np.linalg.qr(V_sp)#

   # print("Performing Orthonormal tests")
    #check = np.allclose(V_sp.T @ V_sp, np.eye(V_sp.shape[1]), rtol =1e-2)
   # print(" ortho check: ", check)
     #unit tests, check hessenberg/orthonorm
    print("Projector check:" )
    
    
    check = np.allclose(V_sp.T@V_sp, V_sp)
    print(check)
    breakpoint()
    
    V_spt = V_sp.T
    #Xn_red = np.dot(Q_spt, Xn)
   # Q_sp = sparse.csc_matrix(Q_sp)
    #Q_spt = sparse.csc_matrix(Q_spt)
    #breakpoint()
    XnP1_red =  V_spt@XnP1
   # M = sparse.csc_matrix(M)
    #A_red = M#Q_spt@M@Q_sp
    winter_red = V_spt@winter@V_sp
    summer_red = V_spt@summer@V_sp
   
    #A_red = sparse.csc_matrix(A_red)
    #if (sparse.issparse(A_red)):
      #  A_red = A_red.todense()
    print("Eigenvalue decomp...")
    #winter_red= eigenvalueDecomp(winter_red)
    #summer_red  = eigenvalueDecomp(summer_red)
    winter_red = sparse.csc_matrix(winter_red)
    #summer_red = sparse.csc_matrix(summer_red)
    #B_a_red = Q_spt@inv@Q_sp
    for jj in range(0,P.timeSteps):
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
        
        
        XnP1_red = H@XnP1_red +B_red@UnP1_red
        #XnP1 = A@XnP1 + B@UnP1
        #breakpoint()
        
        #XnP1_red = np.linalg.solve(summer_red, y)
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
        
        XnP1 = np.real(V_sp@XnP1_red)
        
        
        for ii in range(len(V.Ex)-1):
           # breakpoint()
            V.Ex[ii] = XnP1[ii][0]
            V.Hy[ii] = XnP1[ii+len(V.Ex)][0]
        V.Ex_History[jj] = V.Ex
        
        #XnP1 = sparse.csc_matrix(XnP1)
            
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
def MORViaSPRIM(Q, Xn, XnP1, num, R, k, F, P, V, C_V, C_P):  #from Li Xihao and SPRIM FREUND and intro to MOR
    """
    Takes in matrices E and A from LTI state vectors from matrix
    eqns.
    
    Arnoldi process and Krylov space 
    
    include semi-def and definite tests of R and F etc
    """
    summer = (R+F)
    winter = (R-F)
    summer = summer.todense()
   # A = np.linalg.inv(summer)@winter
    #A = sparse.csc_matrix(A)
   # Q = np.ones((num,k+1))
     # Normalisation (sets L2 norm = 1)
    
   
    Q, H, k = ArnoldiProcess(A, Q, k)
    # H = eigenvalueDecomp(H)
   
   #CONSIDER EXPLICIT RESTART
    blocks = 2
   #SPRIM -> partition V into blocks size of E. H etc, set as diagonal block matrix 
    
    Q1 = Q[:int(num/blocks), :int(k/blocks)]
    Q2 = Q[int(num/blocks):, :int(k/blocks)]
    
    Q_sp = np.block( [  [Q1, np.zeros(Q1.shape)], [np.zeros(Q2.shape), Q2]      ])
    
    
    Xn_red = Q_sp.T@Xn
    XnP1_red = Q_sp.T@XnP1
    
   #Matrix multiply Vsprim by Xn
   
   #Eigenvalue perturbation -> eigenvalue decomposition 
   # Vec.T diagEigVal Vec 
   # eigs>1 = (gamma * eigVal[i])/(norm(eigVal[i]))
   #where gamma = 0.999
   #Vec.T diagEigs>1 Vec =  H_stab
   # use to create reduced system (Xn+1 = H_stab@Xn + V.T@B@UnP1)
   # result is obtained by V@Xn+1 roughlEq Xn+1 feed into V.Ex, V.Hy
   # solve system for Xn+1 follow eqn through 
    
    return Q, H, k, Xn_red, XnP1_red, Q_sp

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
    HeigVals = np.linalg.eigvals(H)
    HeigVecs = np.asmatrix(np.linalg.eig(H)[1]) #columns are eigenvectors associated with each matching eigenval
   # HeigVecs = sparse.csc_matrix(HeigVecs)
    HeigValsDiag = np.diag(HeigVals)
    Per,  HeigVecsL, HeigVecsU = li.lu(HeigVecs)
    print("inverting for eigenvalue decomposition")
    HeigVecsInvL = np.linalg.inv(HeigVecsL)
    HeigVecsInvU = np.linalg.inv(HeigVecsU)
    HeigVecsInv = HeigVecsInvU@HeigVecsInvL
    print("matrices inverted")
    shrinkFactor = 0.999
    
    
    #Check sparsity, size etc 
    for i in range(len(HeigVals)):
        if abs(HeigVals[i]) > 1:
            HeigVals[i] = (shrinkFactor*HeigVals[i])/abs(HeigVals[i])

    H1 = HeigVecsInv@HeigValsDiag
    H = H1@HeigVecs
    
    return H
    
def GlobalArnoldi(A, V_a, k):
    
    H = np.asmatrix( np.ones((k+1,k)) ) # AmQn = Qn+1Hn
    A = sparse.csc_matrix(A) # may cause issue
    #fnrm = np.linalg.norm(Q[:,0])
    breakpoint()
#    Q[:,0]= Q[:,0]/fnrm
    
   #
    V, R = np.linalg.qr(V_a) #R from kr(M,R) is fed in as Q, output as V then Q_sp
   # breakpoint()
    
    
    for j in range(k-1):
        
        W = A@V[:,j]
        print("progress through Arnoldi: ", j, "/", k)
        for i in range(j+1):
            H[i,j] = np.dot(V[:,i].T.conj(), W)
            W -= H[i,j]*V[:,i].T.conj()
            
        #end for i
        tol0 = 1e-12
        H[j+1, j] =  np.linalg.norm(W)
        if abs(H[j+1, j]) >= tol0:
         V[:,j]= V[:,j]/H[j+1, j]
        else:
            print("invariant subspace found ", j)
            return V, H[:-1,:], j
    return V, H[:-1,:], j

#ticc = time.perf_counter()

#ff = testBuild(2,0,A)
#tocc =  time.perf_counter()


#print(ff)

#print("completed loop in: ", tocc-ticc)
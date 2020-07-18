# -*- coding: utf-8 -*-
"""
Creating logic for matrix construction for MOR methods
Using vectorise and ogrid 
"""


import numpy as np
import time 
from numba import njit as nj
from scipy import sparse

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



def ABuild(A, P, selfCoEFHy, selfCoEfEx, curlinExCoEf, curlinHyCoef, blocks=2):  # VERY SPARSE UPDATE MATRIX OVER ALL SPACE
    dRange = P.Nz-1
    A = A.todense()
    
    for j in range(0, dRange-1): #HY BLOCK OVER SPACE
         A[j, 1+j]=selfCoEFHy # hyself[nz]
    for j in range(0, dRange-1):
                A[j, j +dRange+1] = -80
                A[j, j+2 +dRange] = +40 #spatial curl operator [nz] CHECK
    """
    #E block over space
    for j in range(0, dRange-3):
        A[j + dRange, j+1 +dRange] = selfCoEfEx # SelfExCo[nz]        
        A[j + dRange, j +1] = -2000*curlinExCoEf[j] #spatial curl Hy, nz
        A[j +dRange, j +2]= curlinExCoEf[j]       
    """
    A = sparse.csc_matrix(A)
    return A, blocks


def InitMatCalc(A, blocks, source, V, P, C_V, C_P):
    dum = np.zeros((blocks*(P.Nz), 1))
    dum = sparse.csc_matrix(dum)
    source = sparse.csc_matrix(source)
   # A = sparse.csc_matrix(A)
   # breakpoint()
    
    
    #B = sparse.csc_matrix(B)
    
    
    return A
    
def MatCalcIter(A, B, source):
    A = A.todense()
    
    
   # breakpoint()
    Bs = A@B
    #breakpoint()
   # B = sparse.csc_matrix(B)
    source = np.asmatrix(source)
    Bs = np.array(B) + np.array(source)
    #Bs =sparse.csc_matrix(B)
    
    
    #B = sparse.csc_matrix(B) ## is this step necessary?
    return Bs


def TimeIter(A, B, source, V, P, C_V, C_P):
    
    idSrc = [(P.nzsrc), (P.nzsrc -1+P.Nz)]
    
    #breakpoint()
    for jj in range(P.timeSteps):
        #source =  source.toarray()
        
        source[P.nzsrc] = np.exp(-(jj - 30)*(jj-30)/100)
        source[P.nzsrc-1+ P.Nz] = -np.exp(-(jj - 30)*(jj-30)/100)*(1/377)
        #source = sparse.csc_matrix(source)
       # breakpoint()
        top = int(P.Nz+1)
        B = MatCalcIter(A, B, source) 
        B = B.flatten()
        #breakpoint()
        
        
        
        V.Ex  =B[0:top]
        V.Ex_History[jj] = V.Ex
        V.Hy = B[top: 2*(top)]
       
    return B, V.Ex, V.Ex_History, V.Hy



  
    



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
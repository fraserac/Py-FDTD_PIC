# -*- coding: utf-8 -*-
"""
Material definition

@author: Fraser
"""
import numpy as np
from MasterController import *


epsRe = 1
epsIm = 1
epsilon = 1#complex(epsRe,epsIm)
muRe = 1
muIm = 1
mu = 1#complex(muRe, muIm)
MatEps = 9
MatMu = 1
permit_0 = 8.85e-12
permea_0 = 1.26e-6
c0 = 1/(np.sqrt(permit_0*permea_0)) 
freq_in = 5e9
maxFreq = 10e9
MaxGridSize = 1e5
lamMin = c0/(np.sqrt(abs(epsRe)*abs(muRe))*maxFreq)
Nlam = 30
dz =lamMin/Nlam  
delT = (dz)/(c0)   # LOOK INTO HOW DZ AND DELT ARE USED, COURANT NO?
Nz = 200

courantNo = c0*delT/dz
CharImp =np.sqrt(permea_0)/np.sqrt(permit_0)
MaterialFrontEdge = 100  # Discrete tile where material begins (array index) 
MaterialRearEdge = 200
UpHyMat = np.zeros(Nz)#,dtype=complex)
UpExMat = np.zeros(Nz)#,dtype=complex)

UpHyFree = ((1/mu)/CharImp)*courantNo
UpExFree = ((1/epsilon)*CharImp)*courantNo

for k in range(0, MaterialFrontEdge-1):   
    UpExMat[k] =UpExFree 
    UpHyMat[k] =UpHyFree
for jj in range(MaterialFrontEdge-1, MaterialRearEdge-1):
    UpExMat[jj] = (UpExFree/MatEps)
    UpHyMat[jj] = (UpHyFree/MatMu)
for ii in range(MaterialRearEdge-1, Nz):
    UpExMat[ii] = UpExFree
    UpHyMat[ii] = UpHyFree

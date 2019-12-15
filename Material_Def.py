# -*- coding: utf-8 -*-
"""
@author: Fraser
"""
import numpy as np

#from MasterController import *


permit_0 = 8.85e-12
permea_0 = 1.26e-6
epsRe = 1
epsIm = 1
epsilon = 1#complex(epsRe,epsIm)
muRe = 5
muIm = 1
mu = 1#complex(muRe, muIm)
MatEps = epsRe
MatMu = muRe
permit_0 = 8.85e-12
permea_0 = 1.26e-6
c0 = 1/(np.sqrt(permit_0*permea_0))
freq_in = 4e9
maxFreq = 8e9
MaxGridSize = 1e5
lamMin = c0/(np.sqrt(abs(epsRe)*abs(muRe))*maxFreq)
Nlam = 30
dz =lamMin/Nlam  
delT = (dz)/(c0)   # LOOK INTO HOW DZ AND DELT ARE USED, COURANT NO?

courantNo = c0*delT/dz
CharImp =np.sqrt(permea_0)/np.sqrt(permit_0)
MaterialFrontEdge = 80  # Discrete tile where material begins (array index)
MaterialRearEdge = 95
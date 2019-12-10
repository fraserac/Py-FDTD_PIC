# -*- coding: utf-8 -*-
"""
Material definition

@author: Fraser
"""
import numpy as np

permit_0 = 8.85e-12
permea_0 = 1.26e-6
epsRe = 1
epsIm = 1
epsilon = 1#complex(epsRe,epsIm)
muRe = 1
muIm = 1
mu = 1#complex(muRe, muIm)
MatEps = 1
MatMu = 1

CharImp =np.sqrt(permea_0)/np.sqrt(permit_0)
MaterialFrontEdge = 100  # Discrete tile where material begins (array index) 
MaterialRearEdge = 120


# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:36:07 2020

@author: Owner

This script tests basefdtd script functions, with as many unit tests as possible, physics 
tests are done as full or partial model tests, whereby bulk code is run and output is tested against 
either an analytical value for a known result or/and checked against physical conservation laws.

"""

import unittest
from BaseFDTD import FieldInit, SmoothTurnOn, EmptySpaceCalc, Material, UpdateCoef
import Material_Def as matDef
import numpy as np
import sys
import inspect
class Variables_test(object):
    def __init__(self, UpHyMat, UpExMat, Ex, Hy, Ex_History, Hy_History,Psi_Ex_History, Psi_Hy_History, Hys, Exs, x1ColBe, x1ColAf, epsilon, mu, UpExHcompsCo, UpExSelf, UpHyEcompsCo, UpHySelf):
        self.UpHyMat = UpHyMat
        self.UpExMat = UpExMat
        self.UpExHcompsCo = UpExHcompsCo
        self.UpExSelf = UpExSelf
        self.UpHySelf = UpHySelf
        self.UpHyEcompsCo = UpHyEcompsCo 
        self.Ex = Ex
        self.Hy = Hy
        self.Ex_History = Ex_History
        self.Hy_History = Hy_History
        self.Psi_Ex_History= Psi_Ex_History
        self.Psi_Hy_History = Psi_Hy_History
        self.Exs = Exs
        self.Hys = Hys
        self.x1ColBe = x1ColBe
        self.x1ColAf = x1ColAf
        self.epsilon = epsilon
        self.mu = mu
        self.inputVec = []
        self.outputPlots = []
    def __str__(self):
        return 'Contains data that will change during sim'
    
    def __repr__(self):
        return (f'{self.__class__.__name__}', ": Contains field variables that change during sim")
       
        
    # methods to handle user input errors during instantiation.
    
class Params_test(object):
    permit_0 =8.854187817620389e-12
    permea_0 =1.2566370614359173e-06
    CharImp =np.sqrt(permea_0/permit_0)
    c0 = 299792458.0
    
    def __init__(self, epsRe, muRe, freq_in, lMin, nlm, dz, delT, courantNo, domainSize,  matRear, matFront, gridNo, timeSteps, x1Loc, nzsrc, period, eLoss, eSelfCo, eHcompsCo, mLoss, hEcompsCo, hSelfCo, pmlWidth, x2Loc ):
        self.epsRe = epsRe
        self.muRe = muRe
        self.freq_in = freq_in
        self.lamMin = lMin
        self.Nlam = nlm
        self.dz = dz
        self.delT = delT
        self.courantNo= courantNo
        self.materialFrontEdge = matFront
        self.materialRearEdge = matRear
        self.Nz = gridNo
        self.timeSteps = timeSteps
        self.x1Loc = x1Loc
        self.x2Loc = x2Loc
        self.nzsrc = nzsrc
        self.period = period
        self.eLoss = eLoss
        self.eSelfCo = eSelfCo
        self.eHcompsCo = eHcompsCo
        self.mLoss = mLoss
        self.hSelfCo = hSelfCo
        self.hEcompsCo = hEcompsCo
        self.pmlWidth = pmlWidth
        self.domainSize = domainSize
        
    def __repr__(self):
        return (f'{self.__class__.__name__}'(f'{self.epsRe!r}, {self.muRe!r}'))
    
    def __str__(self):
        return 'Class containing all values that remain constant throughout a sim' 


class CPML_Params_test(object):
    
    
    def __init__(self, kappaMax, sigmaEMax, sigmaHMax, sigmaOpt, alphaMax, r_scale, r_a_scale ):
        self.kappaMax=kappaMax
        self.sigmaEMax=sigmaEMax
        self.sigmaHMax=sigmaHMax
        self.sigmaOpt=sigmaOpt
        self.alphaMax=alphaMax
        self.r_scale=r_scale
        self.r_a_scale=r_a_scale
    def __repr__(self):
        return (f'{self.__class__.__name__}')
    
    def __str__(self):
        return 'Class containing all CPML values that remain constant throughout a sim' 
    
    
class CPML_Variables_test(object):
    
    
    def __init__(self, kappa_Ex, kappa_Hy, psi_Ex, psi_Hy, alpha_Ex, alpha_Hy, sigma_Ex, sigma_Hy, beX, bmY, ceX, cmY, Ca, Cb, Cc, C1, C2, C3, eLoss_CPML, mLoss_CPML, den_Hydz, den_Exdz ):
        self.kappa_Ex=kappa_Ex
        self.kappa_Hy=kappa_Hy
        self.psi_Ex=psi_Ex
        self.psi_Hy=psi_Hy
        self.alpha_Ex= alpha_Ex
        self.alpha_Hy=alpha_Hy
        self.sigma_Ex=sigma_Ex
        self.sigma_Hy=sigma_Hy
        self.beX=beX
        self.bmY=bmY
        self.ceX=ceX
        self.cmY=cmY
        self.Ca=Ca
        self.Cb= Cb
        self.Cc=Cc
        self.C1=C1
        self.C2=C2
        self.C3=C3
        self.eLoss_CPML=eLoss_CPML
        self.mLoss_CPML = mLoss_CPML
        self.den_Hydz = den_Hydz
        self.den_Exdz = den_Exdz
    def __repr__(self):
        return (f'{self.__class__.__name__}')
     
    def __str__(self):
        return 'Class containing all CPML values that vary throughout a sim' 


#classes will be called via a Setup (not tearDown function yet), which will be called by each test 
        #func in turn by the unittest module.

class TestBaseFDTD(unittest.TestCase):
    def SetUpInitial(self):
        Pt = Params_test(matDef.epsRe, matDef.muRe, matDef.freq_in, matDef.lamMin, matDef.Nlam, matDef.dz, matDef.delT, matDef.courantNo, matDef.domainSize, matDef.MaterialRearEdge, matDef.MaterialFrontEdge, matDef.Nz, matDef.timeSteps, matDef.x1Loc, matDef.nzsrc, matDef.period, matDef.eLoss, matDef.eSelfCo, matDef.eHcompsCo, matDef.mLoss, matDef.hEcompsCo, matDef.hSelfCo, matDef.pmlWidth, matDef.x2Loc )    
        Vt = Variables_test(matDef.UpHyMat, matDef.UpExMat, matDef.Ex, matDef.Hy, matDef.Ex_History, matDef.Hy_History, matDef.Psi_Ex_History, matDef.Psi_Hy_History, matDef.Hys, matDef.Exs, matDef.x1ColBe, matDef.x1ColAf, matDef.epsilon, matDef.mu, matDef.UpExHcompsCo, matDef.UpExSelf, matDef.UpHyEcompsCo, matDef.UpHySelf)
        C_Pt =  CPML_Params_test(matDef.kappaMax, matDef.sigmaEMax, matDef.sigmaHMax, matDef.sigmaOpt, matDef.alphaMax, matDef.r_scale, matDef.r_a_scale)
        C_Vt = CPML_Variables_test(matDef.kappa_Ex, matDef.kappa_Hy, matDef.psi_Ex, matDef.psi_Hy, matDef.alpha_Ex, matDef.alpha_Hy, matDef.sigma_Ex, matDef.sigma_Hy,matDef.beX, matDef.bmY, matDef.ceX, matDef.cmY, matDef.Ca, matDef.Cb, matDef.Cc, matDef.C1, matDef.C2, matDef.C3, matDef.eLoss_CPML, matDef.mLoss_CPML, matDef.den_Hydz, matDef.den_Exdz )
        return Pt, Vt, C_Pt, C_Vt
    
    def test_FieldInit(self):
        Pt, Vt, C_Pt, C_Vt = self.SetUpInitial()
        T1, T2, T3, T4, T5, T6, T7, T8 = FieldInit(Vt, Pt) # need to feed v,p classes into this 
        self.assertEqual(len(T1), Pt.Nz +1 )
        self.assertEqual(len(T3), Pt.Nz +1) #CHECK THIS IS RIGHT
        self.assertEqual(len(T2), Pt.timeSteps)
        self.assertEqual(len(T4), Pt.timeSteps)
        del Pt, Vt, C_Vt, C_Pt
        
    def test_SmoothTurnOn(self):
        Pt, Vt, C_Pt, C_Vt = self.SetUpInitial()
        self.assertIsNotNone(SmoothTurnOn(Vt,Pt))
        del Pt, Vt, C_Vt, C_Pt
        
        
    def test_EmptySpaceCalc(self):
        Pt, Vt, C_Pt, C_Vt = self.SetUpInitial()
        self.assertEqual(len(EmptySpaceCalc(Vt,Pt)[1]), Pt.Nz+1)
        del Pt, Vt, C_Vt, C_Pt
        
    def test_Material(self):
        Pt, Vt, C_Pt, C_Vt = self.SetUpInitial()
        for i in range(len(Material(Vt,Pt))):
            if not isinstance(Material(Vt,Pt)[i], np.ndarray):
               print("Output from Material function, ", i,", is not a number.") 
               del Pt, Vt, C_Vt, C_Pt
               sys.exit() 
        
    
    def test_UpdateCoef(self):
        Pt, Vt, C_Pt, C_Vt = self.SetUpInitial()
        
        del Pt, Vt, C_Vt, C_Pt
        
    
    def HyTfsfCorr(self):
        Pt, Vt, C_Pt, C_Vt = self.SetUpInitial()
        
        del Pt, Vt, C_Vt, C_Pt
        
    
        
        
        
if __name__ == '__main__':
    unittest.main() 
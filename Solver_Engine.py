# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:16:33 2021

@author: Fraser
Solver_Engine
controls integration steps based off of media in simulation. 
"""
import BaseFDTD11 
import genericStability as gStab
import numpy as np
from numba import njit as nj
from numba import jit
import JuliaHandler as JH

def probeSim(V, P, C_V, C_P, counts, val = "null", af = False, granAttn = 25,  whichField = "Ex", attenRead = False ):
    """
    Take current value of whatever quantity is being probed from
    and add to x1ColBe or x1ColAf array. 
    """
    
    
        #Multi layer probe plot choice of layer granularity? 10 as default. 
        #Functions basically the same as the other probes, but starts at 
        #matfront +5 then adds probes at chosen spaces and frequency.
        #eventually fourier transform each row of 2d array and return 
        #amplitude relative to amplitude of free space (x1ColBe) 
        #build in optional stuff later, beers law for proper 
        # create list of inputs: if this works, move granAttn and attenAmnt to 
        #P class. 
    if attenRead == True:    
        inds = np.arange(P.materialFrontEdge, granAttn*V.attenAmnt + P.materialFrontEdge, granAttn)
       # listOfIndAtten = [inds]
   
        for i in range(len(V.x1Atten)):
            if whichField == "Ex":
                V.x1Atten[i][counts] =  V.Ex[inds[i]]
        return V.x1Atten
                
    elif af == True:
        V.x1ColAf[counts] = val
        return V.x1ColAf

    
    elif af == False:
       # breakpoint()
        V.x1ColBe[counts] = val
    
        return V.x1ColBe
    
    return "Alternative option unavailable, probeSim func"
    
def vidMake(V, P, C_V, C_P, counts, field, whichField = "Ex"):
    # COUNTS WON'T MATCH UP WITH INDEX OF SHRUNK HISTORY. COUNTS/P.vidInterval?
    # counts = zero should not be here

    
    shrunkCnt = int(counts/P.vidInterval)
    if whichField == "Ex":
        if shrunkCnt< len(V.Ex_History):
            V.Ex_History[shrunkCnt] = field
        return V.Ex_History
    
    return "vidMake has gone to the end of the returns "

def boundCondManager(V, P, C_V, C_P):
    # in here we set up boundary conditions, if cpml, call relevant cpml stuff,
    # or call other boundary condition update funcs.
    # if no B.Cs needed, set dummy var arrays as ones. 
    
    #integrators call this to set up boundary conditions and call appropriate
    #features. switch case? 
    if P.CPMLXp  == True or P.CPMLXm == True:
        
        C_V.sigma_Ex, C_V.sigma_Hy, C_V.alpha_Ex,  C_V.alpha_Hy, C_V.kappa_Ex, C_V.kappa_Hy= BaseFDTD11.CPML_ScalingCalc(V, P, C_V, C_P)
        
        C_V.beX, C_V.ceX = BaseFDTD11.CPML_Ex_RC_Define(V, P, C_V, C_P)
        C_V.bmY, C_V.cmY = BaseFDTD11.CPML_HY_RC_Define(V, P, C_V, C_P)
        C_V.eLoss_CPML, C_V.Ca, C_V.Cb, C_V.Cc = BaseFDTD11.CPML_Ex_Update_Coef(V,P, C_V, C_P)
        C_V.mLoss_CPML, C_V.C1, C_V.C2, C_V.C3 = BaseFDTD11.CPML_Hy_Update_Coef(V,P, C_V, C_P)
       
        C_V.den_Exdz, C_V.den_Hydz = BaseFDTD11.denominators(V, P, C_V, C_P)
    return C_V
    
def SourceManager(V, P, C_V, C_P):
    ## boolean fed into argument which chooses source to use and creates appropiately.
    if P.SineCont == True:
        Exs, Hys1 = BaseFDTD11.SmoothTurnOn(V,P)
        Exs = np.asarray(Exs)*P.courantNo
        Hys1 = np.asarray(Hys1)*P.courantNo
        #breakpoint()
        if P.TFSF ==True:
            Hys1 = Hys1*(1/P.CharImp)
        return Exs, Hys1
        #
    #if SineCont == True
         #run SineCont function with no of repeatsm time +interval between
    if P.Gaussian == True: # Generate Exs and use
         Exs = BaseFDTD11.Gaussian(V, P)
         Hys = np.zeros(len(Exs))
         if P.TFSF == True:
             Hys = BaseFDTD11.Gaussian(V,P)
        
         #plt.plot(Exs)
        # Width and other components should be defaulted in construct of sourcemanager
        # source manager changes the source point then returns whole field 
    #if Ricker == True 
        # Ricker func in basefdtd11. 
         return Exs, Hys
     
    return [],[]

def SechArr(a):
    #protect overflow: 
    #breakpoint()
    a= np.where(a>50, 50, a)
    out = 1/np.cosh(a)
    return out

def Sig_Mod(V, P, sig, AmpCarr =1, AmpMod =1, tau = 14.6e-10):
        
        # initially just sech envelope
        trang= np.arange(len(sig))*(P.delT)
        omega_mod = 2*np.pi*(1/tau)
        pulseEnv =(AmpMod + sig)*SechArr(omega_mod*trang) 
        sigOut = pulseEnv
        return sigOut

def IntegratorFreeSpace1D(V,P,C_V, C_P, probeReadFinishBe, probeReadStartAf):
    for i in range(0,2):
        V.tempVarPol, V.tempTempVarE, V.tempVarE, V.tempTempVarPol, V.polarisationCurr, V.Ex, V.Dx, V.Hy = BaseFDTD11.FieldInit(V,P)
        #analReflectCo, dispPhaseVel, realV = BaseFDTD11.AnalyticalReflectionE(V,P)
        
        V.UpHyMat, V.UpExMat = BaseFDTD11.EmptySpaceCalc(V,P)   
        
        V.epsilon, V.mu, V.UpExHcompsCo, V.UpExSelf, V.UpHyEcompsCo, V.UpHySelf = BaseFDTD11.Material(V,P)
        V.UpHyMat, V.UpExMat = BaseFDTD11.UpdateCoef(V,P)
    # move these into bc manager, call bc manager from here
        C_V = BaseFDTD11.CPML_FieldInit(V,P, C_V, C_P)
        C_V = boundCondManager(V, P, C_V, C_P)
      
        # PROBABLY BETTER TO RUN THIS IN LINEAR DISP INTEGRATOR
        #lamCont, lamDisc, diff, V.plasmaFreqE, fix = gStab.spatialStab(P.timeSteps,P.Nz,P.dz, P.freq_in, P.delT, V.plasmaFreqE, V.omega_0E, V.gammaE)
        Exs, Hys = SourceManager(V, P, C_V, C_P)
        tauIn=1/(P.freq_in/5)
        Exs = Sig_Mod(V, P, Exs, tau= tauIn)
        Hys = Sig_Mod(V,P, Hys, AmpMod = 1/P.CharImp, tau = tauIn)
        
        #gStab.vonNeumannAnalysis(V,P,C_V,C_P)

        if P.julia:
            V,P,C_V,C_P = JH.Jul_Integrator_Prep(V, P, C_V, C_P, Exs, Hys, i)
        else:
            for counts in range(0,P.timeSteps):

                    #if counts%(int(P.timeSteps/10)) == 0:
                     #   print("timestep progress:", counts, "/", P.timeSteps)
                    V.Ex =BaseFDTD11.ADE_ExUpdate(V,P, C_V, C_P, counts)
                    if P.CPMLXp == True or P.CPMLXm == True: # Go into cpml field updates to choose x+ and x-
                        C_V.psi_Ex, V.Ex  = BaseFDTD11.CPML_Psi_e_Update(V,P, C_V, C_P)

                    ## SOURCE MANAGER COMES IN HERE

                    V.Ex[P.nzsrc] += Exs[counts]/P.courantNo
                    if P.TFSF == True:
                        V.Hy[P.nzsrc-1] -= Hys[counts]/P.courantNo
                    ####
                    V.Hy = BaseFDTD11.ADE_HyUpdate(V,P, C_V, C_P)
                    if P.CPMLXp == True or P.CPMLXm == True:
                       C_V.psi_Hy, V.Hy = BaseFDTD11.CPML_Psi_m_Update(V,P, C_V, C_P)

                    ##################
                   # V.tempTempVarPol, V.tempVarPol, V.tempVarE, V.tempTempVarE, V.tempTempVarHy, V.tempVarHy, V.tempTempVarJx, V.tempVarJx, C_V.tempTempVarPsiEx, C_V.tempVarPsiEx, C_V.tempTempVarPsiHy, C_V.tempVarPsiHy = BaseFDTD11.ADE_TempPolCurr(V,P, C_V, C_P)
                    #V.Ex = BaseFDTD11.MUR1DEx(V, P, C_V, C_P)

                    #########################



                    if counts >0:
                        if counts % P.vidInterval ==0:
                            if i == 1:
                                V.Ex_History =vidMake(V, P, C_V, C_P, counts, V.Ex, whichField = "Ex")

                    # option to not store history, and even when storing, only store in
                    #intervals



                    if i == 0:
                         if counts <= P.timeSteps-1:
                             if counts <= probeReadFinishBe:
                                     V.x1ColBe= probeSim(V, P, C_V, C_P, counts, V.Ex[P.x1Loc])
                                 #change this from Ex history
                    elif i ==1:
                             if counts <= P.timeSteps-1:
                                 if counts >= probeReadStartAf:
                                         V.x1ColAf= probeSim(V, P, C_V, C_P, counts, V.Ex[P.x2Loc], af = True)
                                         if P.atten == True:
                                             V.x1Atten = probeSim(V, P, C_V, C_P, counts, V.Ex[P.x2Loc], attenRead = True)

            #FDdata, FDXaxis, FDdataPow = gft(V,P, C_V, C_P, V.x1ColBe)   # freq dom stuff for reflection

    return V.Ex, V.Hy, Exs, Hys,  C_V.psi_Ex,C_V.psi_Hy, V.x1ColBe,  V.x1ColAf   #Do I need to return C_V?



# Need to set up options so can choose field with nonlinearity/dispersion etc
def IntegratorNL1D(V,P,C_V,C_P, probeReadFinishBe, probeReadStartAf):
     for i in range(0,2):
        
        V.tempVarPol, V.tempTempVarE, V.tempVarE, V.tempTempVarPol, V.polarisationCurr, V.Ex, V.Dx, V.Hy = BaseFDTD11.FieldInit(V,P)  #Not necessary?
        
        
        V.UpHyMat, V.UpExMat = BaseFDTD11.EmptySpaceCalc(V,P)   
    # move these into bc manager, call bc manager from here
        C_V = BaseFDTD11.CPML_FieldInit(V,P, C_V, C_P)
        C_V = boundCondManager(V, P, C_V, C_P)
        
        lamCont, lamDisc, diff, V.plasmaFreqE, fix = gStab.spatialStab(P.timeSteps,P.Nz,P.dz, P.freq_in, P.delT, V.plasmaFreqE, V.omega_0E, V.gammaE)
        Exs, Hys = SourceManager(V, P, C_V, C_P)
        Exs = Sig_Mod(V,P, Exs)
        Hys = Sig_Mod(V,P, Hys, AmpMod = 1/P.CharImp)
        
        for counts in range(0,P.timeSteps):
          
           
           if i == 1:
                V.tempTempVarPol, V.tempVarPol, V.tempVarE, V.tempTempVarE, V.tempTempVarHy, V.tempVarHy, V.tempTempVarJx, V.tempVarJx, C_V.tempTempVarPsiEx, C_V.tempVarPsiEx, C_V.tempTempVarPsiHy, C_V.tempVarPsiHy = BaseFDTD11.ADE_TempPolCurr(V,P, C_V, C_P)
                V.polarisationCurr = BaseFDTD11.ADE_PolarisationCurrent_Ex(V, P, C_V, C_P, counts)
                
           V.Ex =BaseFDTD11.ADE_ExUpdate(V, P, C_V, C_P, counts) 
                # V.Jx = BaseFDTD11.ADE_JxUpdate(V,P, C_V, C_P)  
           
           if P.CPMLXp == True or P.CPMLXm == True: # Go into cpml field updates to choose x+ and x- 
                    C_V.psi_Ex, V.Ex  = BaseFDTD11.CPML_Psi_e_Update(V,P, C_V, C_P)
                  
           V.Ex[P.nzsrc] += Exs[counts]/P.courantNo
           
           if P.TFSF == True:
                V.Hy[P.nzsrc-1] -= Hys[counts]/P.courantNo
                
           
           
           V.Dx = BaseFDTD11.ADE_DxUpdate(V, P, C_V, C_P)  # Linear bit
          # V.Ex =BaseFDTD11.ADE_ExCreate(V, P, C_V, C_P) 
           V.Acubic = BaseFDTD11.AcubicFinder(V,P)
           
           V.Ex = BaseFDTD11.NonLinExUpdate(V,P)
           V.Hy = BaseFDTD11.ADE_HyUpdate(V,P, C_V, C_P)
           if P.CPMLXp == True or P.CPMLXm == True:
                   C_V.psi_Hy, V.Hy = BaseFDTD11.CPML_Psi_m_Update(V,P, C_V, C_P)
          
           if counts >0:
               if counts % P.vidInterval ==0:
                   if i == 1:
                       V.Ex_History =vidMake(V, P, C_V, C_P, counts, V.Ex, whichField = "Ex")
           
           # option to not store history, and even when storing, only store in 
           #intervals 
           
           
   
           if i == 0:
                if counts <= P.timeSteps-1:
                    if counts <= probeReadFinishBe:
                            V.x1ColBe= probeSim(V, P, C_V, C_P, counts, V.Ex[P.x1Loc])
                        #change this from Ex history
           elif i ==1:
                    if counts <= P.timeSteps-1:
                        if counts >= probeReadStartAf:
                                V.x1ColAf= probeSim(V, P, C_V, C_P, counts, V.Ex[P.x2Loc], af = True)
                                if P.atten == True:
                                    V.x1Atten = probeSim(V, P, C_V, C_P, counts, V.Ex[P.x2Loc], attenRead = True)
                   
     return V.Ex, V.Hy, Exs, Hys,  C_V.psi_Ex,C_V.psi_Hy, V.x1ColBe,  V.x1ColAf


    
def IntegratorLinLor1D(V, P, C_V, C_P, probeReadFinishBe, probeReadStartAf):
    for i in range(0,2):
        
        V.tempVarPol, V.tempTempVarE, V.tempVarE, V.tempTempVarPol, V.polarisationCurr, V.Ex, V.Dx, V.Hy = BaseFDTD11.FieldInit(V,P)
        
        
        V.UpHyMat, V.UpExMat = BaseFDTD11.EmptySpaceCalc(V,P)   
    # move these into bc manager, call bc manager from here
        C_V = BaseFDTD11.CPML_FieldInit(V,P, C_V, C_P)
        C_V = boundCondManager(V, P, C_V, C_P)
        
        lamCont, lamDisc, diff, V.plasmaFreqE, fix = gStab.spatialStab(P.timeSteps,P.Nz,P.dz, P.freq_in, P.delT, V.plasmaFreqE, V.omega_0E, V.gammaE)
        Exs, Hys = SourceManager(V, P, C_V, C_P)
        tauIn = 1/(P.freq_in/5)
        #Exs = Sig_Mod(V,P, Exs,tau =tauIn)
       # Hys = Sig_Mod(V,P, Hys, AmpMod = 1/P.CharImp, tau = tauIn)
        #gStab.vonNeumannAnalysis(V,P,C_V,C_P)
        V.test = 0
        
        for counts in range(0,P.timeSteps):
           
           
           if i == 1:
                V.tempTempVarPol, V.tempVarPol, V.tempVarE, V.tempTempVarE, V.tempTempVarHy, V.tempVarHy, V.tempTempVarJx, V.tempVarJx, C_V.tempTempVarPsiEx, C_V.tempVarPsiEx, C_V.tempTempVarPsiHy, C_V.tempVarPsiHy = BaseFDTD11.ADE_TempPolCurr(V,P, C_V, C_P)
                V.polarisationCurr = BaseFDTD11.ADE_PolarisationCurrent_Ex(V, P, C_V, C_P, counts)
                
           V.Ex =BaseFDTD11.ADE_ExUpdate(V, P, C_V, C_P, counts) 
                # V.Jx = BaseFDTD11.ADE_JxUpdate(V,P, C_V, C_P)  
           
           if P.CPMLXp == True or P.CPMLXm == True: # Go into cpml field updates to choose x+ and x- 
                    C_V.psi_Ex, V.Ex  = BaseFDTD11.CPML_Psi_e_Update(V,P, C_V, C_P)
                  
           V.Ex[P.nzsrc] += Exs[counts]/P.courantNo
           
           if P.TFSF == True:
                V.Hy[P.nzsrc-1] -= Hys[counts]/P.courantNo
                
           V.Dx = BaseFDTD11.ADE_DxUpdate(V, P, C_V, C_P)
           V.Ex =BaseFDTD11.ADE_ExCreate(V, P, C_V, C_P) 
           V.Hy = BaseFDTD11.ADE_HyUpdate(V,P, C_V, C_P)
           if P.CPMLXp == True or P.CPMLXm == True:
                   C_V.psi_Hy, V.Hy = BaseFDTD11.CPML_Psi_m_Update(V,P, C_V, C_P)
            
           #V.Ex = BaseFDTD11.MUR1DEx(V, P, C_V, C_P)
           
            #C_V.psi_Ex, V.Ex  = BaseFDTD11.CPML_Psi_e_Update(V,P, C_V, C_P)
           # V.Ex = BaseFDTD11.MUR1DEx(V, P, C_V, C_P)
           
           
           
           """
           Re write probes to not require Ex_History.
           Re write History to be optional and be taken in interval steps
           Re write probes into a different function 
           Create transmission probes and attenuation probes. 
           
           Spatial dispersion plot alongside looping parameters
           Function which plots epsilon, mu, refractive index etc with loop
           Rigorous reflection and transmission probe timings.
           Beer's law vs attenuation plot 
           
           clean up redundant code, split big functions into smaller ones,
           split scripts into multiple scripts. 
           
           Prepare for nonlinear ade (Harmonics, then resonance tuning
           Manley-Rowe verification, resonance tuning verification?)
           Prepare for 2D, (Extra update fields, plots, dispersion checks, 
           2D checks angles stuff, geometry designer)
           Simple charged particle distribution evolution. 25th Jan?
           
           
           should free space pre-reflection be running Dx? Maybe just call 
           free space integrator once?
           
           """
           if counts >0:   # MOVE TO FUNCTION 
               if counts % P.vidInterval ==0:
                   if i == 1:
                       V.Ex_History =vidMake(V, P, C_V, C_P, counts, V.Ex, whichField = "Ex")
           
           # option to not store history, and even when storing, only store in 
           #intervals 
           
           
   
           if i == 0:
                if counts <= P.timeSteps-1:
                    if counts <= probeReadFinishBe:
                            V.x1ColBe= probeSim(V, P, C_V, C_P, counts, V.Ex[P.x1Loc])
                        #change this from Ex history
           elif i ==1:
                    if counts <= P.timeSteps-1:
                        if counts >= probeReadStartAf:
                                V.x1ColAf= probeSim(V, P, C_V, C_P, counts, V.Ex[P.x2Loc], af = True)
                                if P.atten == True:
                                    V.x1Atten = probeSim(V, P, C_V, C_P, counts, V.Ex[P.x2Loc], attenRead = True)
    return V.Ex, V.Hy, Exs, Hys,  C_V.psi_Ex,C_V.psi_Hy, V.x1ColBe,  V.x1ColAf


# Need to set up options so can choose field with nonlinearity/dispersion etc




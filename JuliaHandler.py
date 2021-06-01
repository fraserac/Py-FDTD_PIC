# orders julia funcs
#TODO pass classes to here, and in here create dicts of each class. Pass dict to JuliaBaseFDTD

import julia
import re
import numpy as np

import time
from ipyjulia_hacks import get_api
def PyDictToJuliaDict(d):
    jul = julia.Julia()
    jul.eval('include("JuliaBaseFDTD.jl")')
    jul.eval('oof = Dict()')

    for key in d.keys():
        if type(d[key]) == float:

            k = key
            v = d[k]
        #create a dict from classdict of only variables and parameters you actually need, iterate through this
        #pass through as text, removing \n, within julia split into list separated by , or whitespace whilst parsing to
        #float or whatever
        #pass array comp push into val, push val into dictionary.
    #print(v)
        #jul.eval("val = []")
            print("UpExMat")
            v=d["UpExMat"]
            #pattern = r'\n'
          #  v = re.sub(pattern, "", v)
            jul.Jul_Freespace1DIntegrator(Vd, Pd, C_Vd, C_Pd, i)
            print(vout, len(vout), type(vout))


        #jul.eval(f"push!(val,{i})")
        #pattern = r'\n'
        #v = re.sub(pattern, "", v)
        #jul.eval(f'val =  {v} ')



#jul.eval(f'println({n}["UpExMat"][80])')
    #jul.eval("println(values(oof))")
    #jul.eval(f'merge({n}, oof)')

def PyDictToTupleList(d):
    d_view = d.items()
    d_list = list(d_view)
    return d

def Jul_Integrator_Prep(V,P,C_V, C_P, Exs, Hys, i):
    Vd = V.__dict__
    Pd = P.__dict__
    C_Vd = C_V.__dict__
    C_Pd = C_P.__dict__

    
    jul = julia.Julia()
    jul.eval('include("JuliaBaseFDTD.jl")')
   # jul.eval("Vd =Dict{Any, Any}()")




    #jul.eval(f"Vd = jul")


    #TODO fix the syntax error here, maybe pass as json? manually convert
    if P.LorentzMed:

        #V.Ex, V.Hy, Exs, Hys, C_V.psi_Ex, C_V.psi_Hy, V.x1ColBe, V.x1ColAf = SE.IntegratorLinLor1D(V, P, C_V, C_P,
                                                                                                   #probeReadFinishBe,
                                                                                                  # probeReadStartAf)
        pass
    elif P.FreeSpace:
        Vd, Pd, C_Vd, C_Pd = jul.Jul_Freespace1DIntegrator(Vd, Pd, C_Vd, C_Pd, Exs, Hys, i)
        for key in C_Vd.keys():
            print(key)
        V.Ex = Vd["Ex"]
        V.Hy = Vd["Hy"]
        C_V.psi_Ex = C_Vd["psi_Ex"]
        C_V.psi_Hy = C_Vd["psi_Hy"]
        V.x1ColBe = Vd["x1ColBe"]
        V.x1ColAf = Vd["x1ColAf"]
        V.Ex_History = Vd["Ex_History"]
        return V, P, C_V, C_P

    elif P.nonLinMed:
        #V.Ex, V.Hy, Exs, Hys, C_V.psi_Ex, C_V.psi_Hy, V.x1ColBe, V.x1ColAf = SE.IntegratorNL1D(V, P, C_V, C_P,
                                                                                              # probeReadFinishBe,
                                                                                               #probeReadStartAf)
        pass




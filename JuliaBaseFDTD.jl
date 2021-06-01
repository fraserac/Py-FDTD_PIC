#=
JuliaBaseFDTD:
- Julia version: 
- Author: frasercripps
- Date: 2021-05-30
=#
##ADE_ExUpdate
#ADE_HyUpdate
#Exs
#pass all classes through as dicts
#videomake
#Probesim
using Revise

function PrintIt(a)
    for i=1:2000
        if mod(i,1000) ==0
            println(a)
        end
    end
end

#function DictConverter(Vd, Pd, C_Vd, C_Pd)



function Jul_Freespace1DIntegrator(Vd, Pd, C_Vd, C_Pd, Exs, Hys, i)
    for counts=1:Pd["timeSteps"]  # zero index to one index?!
        println(counts)
        return Vd, Pd, C_Vd, C_Pd
        Vd["Ex"] = ADE_ExUpdate(Vd, Pd, C_Vd, C_Pd)
        r
        Vd["Ex"][Pd["nzsrc"]] += Exs[i]/P["courantNo"]
        Vd["Hy"] = ADE_HyUpdate(Vd, Pd, C_Vd, C_Pd)
        if counts >0
            if mod(counts, Pd["vidInterval"]) ==0
                if i == 1
                    Vd["Ex_History"]=vidMake(Vd, Pd, C_Vd, C_Pd, counts, Vd["Ex"], whichField = "Ex")
                end
            end
        end #end of if tree

    if i == 0
        if counts <= Pd["timeSteps"]-1
            if counts <= probeReadFinishBe
                 Vd["x1ColBe"]= probeSim(Vd, Pd, C_Vd, C_Pd, counts, Vd["Ex"][Pd["x1Loc"]])
            end
        end
             #change this from Ex history
    elseif i ==1
         if counts <= Pd["timeSteps"]-1
             if counts >= probeReadStartAf
                 Vd["x1ColAf"]= probeSim(Vd, Pd, C_Vd, C_Pd, counts, Vd["Ex"][Pd[""]], af=true)
                 if Pd["atten"]
                     Vd["x1Atten"] = probeSim(Vd, Pd, C_Vd, C_Pd, counts, Vd["Ex"][Pd["x2Loc"]], attenRead=true)
                 end
             end
         end
    end

    end # out timestepping
end


function ADE_ExUpdate(Vd, Pd, C_Vd, C_Pd)
    println("HERE!")
    end

function ADE_HyUpdate()
    end

function vidMake()
    end

function probeSim()
    end


"""
V.Ex = BaseFDTD11.ADE_ExUpdate(V, P, C_V, C_P, counts)
V.Ex[P.nzsrc] += Exs[counts]/P.courantNo
V.Hy = BaseFDTD11.ADE_HyUpdate(V,P, C_V, C_P)
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

"""
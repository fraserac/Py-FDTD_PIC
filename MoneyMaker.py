# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:57:07 2020

@author: Wolowizard
MoneyMaker
"""
import numpy as np
import itertools as it
import sys
import math


title = "Swansea vs Luton"
dictWin = {"Sky bet, W1" : 19/11, "VirginBet, W2,":8/11, "William Hill, W3":12/10}
dictLose = {"Sky bet, L1" : 18/5, "VirginBet, L2,":4/1, "William Hill, L3": 19/5}
dictDraw = {"Sky bet, D1" : 5/4, "VirginBet, D2,":140/5, "William Hill, D3": 30/10}

maxStake = 5.00#
res = maxStake/100
#INITIAL STAKE BACK TOO?


resVec = np.linspace(0, maxStake, 40)

winVec = [dictWin["Sky bet, W1"], dictWin["VirginBet, W2,"], dictWin["William Hill, W3"]]   #Contains odds in the form of coefficient of stake
loseVec = [dictLose["Sky bet, L1"], dictLose["VirginBet, L2,"], dictLose["William Hill, L3"]]
drawVec =[dictDraw["Sky bet, D1"], dictDraw["VirginBet, D2,"], dictDraw["William Hill, D3"]]
Si = 0.0
Sj = 0.0
Sk = 0.0
Smat = np.zeros(len(resVec)*len(winVec)).reshape(len(resVec), len(winVec))

X=[] # vector of earnings
outputDict = {}
bigM = np.zeros(len(winVec)**3, dtype=object).reshape(len(winVec), len(winVec)**2)

permList = list((it.product('012', repeat=3)))
permW = np.zeros(len(permList))
permL = np.zeros(len(permList))
permD = np.zeros(len(permList))

for ii in range(len(permList)):
    permW[ii] = winVec[int(permList[ii][0])]# 
    permL[ii] = loseVec[int(permList[ii][1])]
    permD[ii] = drawVec[int(permList[ii][2])]
    for jj in range(len(winVec)**2):
        for kk in range(len(winVec)):
            
            bigM[kk, jj] = -1*np.ones(len(winVec)**2).reshape(len(winVec), len(winVec))
            bigM[kk, jj][0,0] = permW[kk]+1
            bigM[kk, jj][1,1] = permL[kk]+1
            bigM[kk, jj][2,2] = permD[kk]+1
    
            

sep = " " 



"""
FIX BELOW: SAME STAKES FOR EVERY THING 
"""
smPermList = list((it.product(resVec, repeat=3)))
Si = 0.0
Sj = 0.0
Sk = 0.0
Smat = np.zeros(len(smPermList)*len(winVec)).reshape(len(smPermList), len(winVec))

#ORIGINAL STAKE? 

jjk = 0
iMod =0
jMod = 0
count = 0

"""
this bit finds the S vector by permuation, replace with s vector via 
eigenvector analysis 
"""
for i in range(len(bigM.flatten())):
         A = bigM.flatten()
         for items in A:
             CC = np.linalg.eig(A[i])
              
             for eVecs in range(len(CC[1])):
                  
                  B = np.where(CC[1][eVecs]>0)
                  if len(B[0]) > 2:
                        count += 3
                       # print("perfect bet!")
                        Si = CC[1][eVecs][0]
                        Sj = CC[1][eVecs][1]
                        Sk = CC[1][eVecs][2]
                        Smat[i] = [Si, Sj, Sk]
                        Smat[i][np.where(Smat[i]<0)]=0
                        d = [i]
                        e = sep.join([str(elem) for elem in d])
                        X = np.matmul(A[i], Smat[i])
                        outputDict[e] = [Smat[i], X, np.linalg.norm(X)]
                  if len(B[0]) ==2:
                      
                        count +=2
                        Si = CC[1][eVecs][0]
                        Sj = CC[1][eVecs][1]
                        Sk = CC[1][eVecs][2]
                        Smat[i] = [Si, Sj, Sk]
                        Smat[i][np.where(Smat[i]<0)]=0
                        d = [i]
                        e = sep.join([str(elem) for elem in d])
                        X = np.matmul(A[i], Smat[i])
                        outputDict[e] = [Smat[i], X, np.linalg.norm(X)]
                  if count == 2:    
                       # print("Good bet", CC[1][eVecs], CC[0])
                        #move to output dict 
                        Si = CC[1][eVecs][0]
                        Sj = CC[1][eVecs][1]
                        Sk = CC[1][eVecs][2]
                        Smat[i] = [Si, Sj, Sk]
                        Smat[i][np.where(Smat[i]<0)]=0
                        d = [i]
                        e = sep.join([str(elem) for elem in d])
                        X = np.matmul(A[i], Smat[i])
                        outputDict[e] = [Smat[i], X, np.linalg.norm(X)]
                        
                  
            
                    
        #now take search eigenvectors for more than two negative vals 
        
                
"""
            
        if count == 2:    
            print("Good bet", A[1][eVecs], A[0])
            #move to output dict 
            Si = A[1][eVecs][0]
            Sj = A[1][eVecs][1]
            Sk = A[1][eVecs][2]
            Smat[i] = [Si, Sj, Sk]
            d = [i, j]
            e = sep.join([str(elem) for elem in d])
            outputDict[e] = [Smat[i], X, np.linalg.norm(X)]

for i in range(len(smPermList)):
            Si = smPermList[i][0]
            Sj =  smPermList[i][1]
            Sk = smPermList[i][2]
            Smat[i] = [Si, Sj, Sk]
            jMod = i%8
            if jMod == 0:
                if i >0:
                    if iMod <2:
                        iMod+=1
        
            X = np.matmul(bigM[iMod,jMod], Smat[i])
            for vals in range (len(X)):
                if(all(aaa >= 0 for aaa in X)):
                    d = [iMod,jMod]
                    e = sep.join([str(elem) for elem in d])
                    outputDict[e] = [Smat[i], X, np.linalg.norm(X)]
"""            

#permutation matrix for submatrix selection   
# row val kept constant, column varied 
# 0-2, 0-8  iVec = 
        
curVal = 0.0 
curInd = 0       

for keys in outputDict:
    if outputDict[keys][2] > curVal:
        curVal = outputDict[keys][2]
        curInd = keys[0]

ind = sep.join([str(eleme) for eleme in curInd])       
indNo = [int(ind[0])]


if ind == '0 0':
    print("No viable candidate", ind)
    sys.exit()

#REPLACE BIGM WITH FLATTENED MATRIX 

print("Winning odds plus optimal stake:", A[indNo[0]][0,0] , "£", outputDict[ind][0][0]) # last term is S vector
print("Losing odds plus optimal stake:", A[indNo[0], indNo[1]][1,1], "£",outputDict[ind][0][1])             
print("Drawing odds plus optimal stake:", A[indNo[0], indNo[1]][2,2], "£",outputDict[ind][0][2] )

valListWin = list(dictWin.values())
valListLose = list(dictLose.values())
valListDraw = list(dictDraw.values())
keyListWin = list(dictWin.keys())
keyListLose = list(dictLose.keys())
keyListDraw = list(dictDraw.keys())


for item  in valListWin:
        if math.isclose(item, bigM[indNo[0], indNo[1]][0,0], rel_tol= 1e-1):
            winComp = keyListWin[valListWin.index(item)]
            
for item  in valListLose:
        if math.isclose(item, bigM[indNo[0], indNo[1]][1,1], rel_tol= 1e-1):
            loseComp = keyListLose[valListLose.index(item)]

for item  in valListDraw:
        if math.isclose(item,bigM[indNo[0], indNo[1]][2,2], rel_tol= 1e-1):
            drawComp = keyListDraw[valListDraw.index(item)]

print("Companies for win, loss, draw bets for game ", title, ": ", winComp, loseComp, drawComp )



print("Net loss/gain if all bets are placed and team wins/loses/draws: ", "£", X)
print("remaining  total stake if team wins/loses/draws", X)
#print("Full stake over all three result bets:", "£", outputDict[ind][0][0]+ outputDict[ind][0][1]+ outputDict[ind][0][2])
print("prize based off of winning/losing/drawing", (outputDict[ind][0][0]+ outputDict[ind][0][1]+ outputDict[ind][0][2])  )
"""
#iterate through each output element and find vector with max magnitude and
#associated data.
"""
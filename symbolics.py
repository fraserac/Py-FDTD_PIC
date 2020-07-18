# -*- coding: utf-8 -*-
"""
Symbolic stuff 


@author: Owner


Co efficients for coupled recursive matrix 
"""

from sympy import symbols as sym,  Array as Arr, Eq, solve, symarray, Matrix as mat, latex
from sympy.matrices.dense import list2numpy

import numpy as np
from sympy import init_printing, linsolve
from IPython.display import display, Markdown, Latex


"""
A = B + C + D
E = A + D + F
P = E + Y + Z

H1+ = H1 - E1 + E2
E1+ = E1 + H1+ -H2 
H2+ = H2+ -E1+ +E2
            
            selfH         curlH 
H[0]' =     H[0]
H[1]' =     H[1]        - E[1] + E[2]
H[2]' =     H[2]        - E[2] + E[3]
H[3]' =     H[3]        - E[3] + E[4]
H[4]' =     H[4]        - E[4] + E[5]
E[0]' =                 + E[0]
E[1]' =    -H[0] + H[1]'       + E[1]
E[2]' =    -H[1] + H[2]'       + E[2]
E[3]' =    -H[2] + H[3]'       + E[3]    
E[4]' =    -H[3] + H[4]'       + E[4]

            curlE                selfE
            
            
            
            
######################################
######################################

###########
#
#
###########
               selfH[i]      curlH[i]

XN+1[0]' =     XN[0]
XN+1[1]' =     XN[1]        - XN[1] + XN[2]
XN+1[2]' =     XN[2]        - XN[2] + XN[3]
XN+1[3]' =     XN[3]        - XN[3] + XN[4]
XN+1[4]' =     XN[4]        - XN[4] + XN[5]
.
.
.
.  nz-4 later
.
.
.
.
XN+1[nz+1]' =                 + E[0]
XN+1[nz+2]' =    -XN[0] + XN[1]'       + XN[1]
XN+1[nz+3]' =    -XN[1] + XN[2]'       + XN[2]
XN+1[nz+4]' =    -XN[2] + XN[3]'       + XN[3]
XN+1[nz+5]' =    -XN[3] + XN[4]'       + XN[4]

                  curlE[i]              selfE[i]
                  
                  
                  

ROW JJ:
substitution occurs in lower half H quadrant, 1-nz

Step one, construct array of equations as above, with a key referring to each summation

every sub point, sub in relevant previous equation, fill in other terms as above.

col 1 first half plus second half negative 0 -nz-1
col 2 first half empty second half sub in previous equations with for loop 
col 3 first half negative  1-nz-1 second half zero 
col 4  positive 2-nz, positive, E 1-nz+1




COLUMN II: 



 selfCoEFHy, selfCoEfEx, curlinExCoEf, curlinHyCoef,
"""
def matBuilderSym(Nz, A, blocks =2):
    dRange = Nz
    As = A
    A = mat.zeros(blocks*dRange+2, blocks*dRange+2)
    
    #Hy stamp
    for j in range(1, dRange): #HY BLOCK OVER SPACE
         A[j, 1+j]= As[j, 1+j] # hyself[nz]
    for j in range(1, dRange-2):
                A[j, j +dRange+1] = -1*As[j, j +dRange+1]
                A[j, j+2 +dRange] = As[j, j+2 +dRange]
                
    #E block            
    for j in range(1, dRange):
        A[j + dRange, j+1 +dRange] = As[j + dRange, j+1 +dRange] # SelfExCo[nz]        
        A[j + dRange, j +1] = -1*As[j + dRange, j +1] #spatial curl Hy, nz
        A[j +dRange, j +2]= As[j +dRange, j +2] #soln to be subbed    
    return A 

def matSubberSym():
    pass

"""
def coEfFinder(V, P, C_V, C_P):
      betaE = (0.5*V.plasmaFreqE**2*P.permit_0*P.delT)/(1+0.5*V.gammaE*P.delT)
      kapE = (1-0.5*V.gammaE*P.delT)/(1+0.5*V.gammaE*P.delT)
      selfCoEfHy = 1
      selfCoEfEx = ((2*P.permit_0-betaE*P.delT)/(2*P.permit_0+betaE*P.delT))
      curlinExCoEf=((2*P.delT)/(2*P.permit_0+betaE*P.delT))*(1/P.courantNo)*C_V.den_Exdz
      curlinHyCoef = (1/P.courantNo)*C_V.den_Hydz
      
      return selfCoEfHy, selfCoEfEx, curlinExCoEf, curlinHyCoef

for i in range(2*Nz-4, 2*Nz+2):
    AFin[i, 2]= Eqn.rhs[i].subs('Hy_{}'.format(i), Eqn.lhs[i-2*Nz-3])
    
    
Step 1: Create symbolic matrix A. 
Step 2: Create Xn, B, UnP1, XnP1
Step 3: For loop sub in blocks of A with envsetup values and recursives 
Step 4: Generate AFin, sub in RHS into recursives 
Step 5: Solve for basis vectors of Xn, save into XnP1
Step 6: Extract 1-Nz from XnP1 and place into V.Ex
Step 7: V.Ex into V.Ex_History[counts]
Step 8: Loop for counts
Step 9: Videomaker
    
"""

blocks = 2

Nz = 5
AFin = symarray('AFin', (2*Nz+2, 2*Nz+2))
As= symarray('A', (2*Nz+2,2*Nz+2))
A = matBuilderSym(Nz, As)

Exn = symarray('Ex', (Nz))
Hyn = symarray('Hy', (Nz))
zeros = mat(Arr([0,0]))
Exn = mat(Exn)
Hyn = mat(Hyn)
Xn = Hyn.row_insert(Nz, Exn)
Xn = Xn.row_insert(blocks*Nz, zeros)
B = symarray('B', (2*Nz+2, 2*Nz+2))
UnP1 = symarray("U", (2*Nz+2))
UnP1 = mat(UnP1)
ExnSol = symarray('Ex_sol ', (Nz))
HynSol = symarray('Hy_sol ' , (Nz))
ExnSol = mat(ExnSol)
HynSol = mat(HynSol)
XnP1 = HynSol.row_insert(Nz, ExnSol)
XnP1 = XnP1.row_insert(blocks*Nz, zeros)
#setup matrix for source and B 

Eqn = Eq(A*Xn + B*UnP1  , XnP1)

for i in range(2*Nz-4, 2*Nz+2):
    AFin[i, 2]= Eqn.lhs[i].subs('Hy_{}'.format(i), Eqn.rhs[i-2*Nz-3])
#create matrix A with dashed terms
#replaced dashed terms with relevant row*col + row+col data 
#sub new symbolic value back into A
# linsolve to find XnP1

system = Eq(AFin*Xn + B*UnP1, XnP1)

#print(Eqn)






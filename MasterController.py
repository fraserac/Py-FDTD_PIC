# -*- coding: utf-8 -*-
"""
Fields controller, this script is the master that guides all the processes, 
calling the field updates and eventually particle updaters and field interpolater 
as well as any memory saving methods like sparse matrices etc. 
BC functions called as well 
If GUI is incorporated it will act as a direct interface with this script. 
"""

import numpy as np
import scipy as sci
import math 
from BaseFDTD import *
  

# IN HERE WE NEED FOR LOOP OVER TIME CONTAINING UPDATE FUNCTION CALLS FOR FIELDS

  ##M MAKE THIS MORE ACCURATE LATER

Nz = 200

timeSteps =449

#Ex =np.zeros(Nz)#,dtype=complex)
#Hy=np.zeros(Nz)#,dtype=complex)
#Ex_History= [[]]*timeSteps
nzsrc = 50#round(Nz/2)

if timeSteps > 5000:
    print('time stepping too damn fine')
    stahp
    
for count in range(timeSteps):
    #print(count
   FieldInit(Nz, timeSteps)
   HyBC(Nz)
   #function H field update feed in H field 
   #Tf/sf H correction
   #function ABC E field 
   #function update E field 
   #function Prepare and perform animation  (ani inside this function with plot?)
   
     #make T0 and tau into variables
    Ex[nzsrc]= Ex[nzsrc] + np.exp(-(count +0.5 -(-0.5)-30)*(count +0.5 -(-0.5)-30)/100) #tf/sf correction Ex
    Ex_History[count] = np.insert(Ex_History[count], 0, Ex)
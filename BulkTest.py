# -*- coding: utf-8 -*-
"""
BULK TEST
takes in tuple of vars of interest, list of types each iterant should be, 
takes in condition that needs to be met 
checks all are true and performs other analyses

"""
import numpy as np
import pandas 

def genericTest( inputDict, inputTimeVal= "",nameofVar= "", inputSpaceVal = "", typeList = "", conditionList ="", checkList = "", minList= "", maxList = "", growthList = ""):
        inDataFrame = pandas.DataFrame(inputDict, columns =inputSpaceVal, index= inputTimeVal)
        #print(inDataFrame)
        #print(inDataFrame.describe())
        # print(np.max(inputDict), np.argmax(inputDict), "max value and location in history")
       
        #print(inDataFrame, "base df")
        alpha = inDataFrame.pct_change()
        
        alpha = alpha.replace([np.inf, -np.inf], np.nan)
        #alpha = alpha.replace(0.0, np.nan)
        alpha = alpha.drop(alpha.columns[0.000000], axis=1)
        
        beta=alpha.idxmax()
        
        gamm =(alpha.max(skipna = True)) 
        #gamm = alpha.to_numpy()
        #print(alpha, "rate of change dataframe", nameofVar)
        print(beta, "largest rate of change", nameofVar)
        print((alpha.max(skipna = True)),"largest changes percent", nameofVar)
        bean = alpha.to_numpy()
        beans = np.where(bean > 50)# indices of values
        if len(beans)>0:
            print("rate of change is high at: " , bean[beans])
 #fixxxxx       
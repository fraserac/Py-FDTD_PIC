# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 17:36:07 2020

@author: Owner
"""

import unittest
from BaseFDTD import FieldInit
import Material_Def
import numpy as np


class TestBaseFDTD(unittest.TestCase):
    
    def test_FieldInit(self): 
        T1, T2, T3, T4, T5, T6 = FieldInit(Mc.Nz,Mc.timeSteps)
        self.assertEqual(len(T1), Mc.Nz )
        self.assertEqual(len(T3), Mc.Nz )
        self.assertEqual(len(T2), Mc.timeSteps)
        self.assertEqual(len(T4), Mc.timeSteps)

if __name__ == '__main__':
    unittest.main() 
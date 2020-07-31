# -*- coding: utf-8 -*-
"""
Cython set up 
"""
from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("helloworld.pyx", languagelevel = 3)
)

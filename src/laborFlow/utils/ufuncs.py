# -*- coding: utf-8 -*-
"""
Numba ufuncs for speeding up certain calculations via multi-threading


Note that these functions are named according to the methods they assist

When importing, import them as: 
    
    from laborFlows.utils import ufuncs


Dealing with pointers:
https://numba.pydata.org/numba-doc/latest/reference/utils.html#dealing-with-pointers


One approach, if continuing to use this framework is to write the u_func for the lowest 
level and implement upwards, i.e the calcWage function can be written such that it is 
usable at the Economies or Firm level. In fact, it must be both in order to minimize 
the amount of code written and to keep the implementations consistent


@author: Zach Modig
"""



# import numpy as np
from numba import jit # , cuda, vectorize # May eventually add GPU computing, but not now



## This probably isn't going to see any speed up...need to apply this to Economy calculation that puts this in a loop
@jit(nogil=True, nopython=True)
def calcUtility(wage, rent, A, pref, utility):
    """
    Pass in vectors of wages, rents, Ammenities, and preferences and return 
    corresponding utilities
    
    Utility should be a numpy array of zeros with the same length as the other
    inputs
    """
    
    n_pts = len(wage)
    
    for ii in range(n_pts):
        
        utility[ii] = wage[ii] - rent[ii] + A[ii] + pref[ii]
    
    
    return utility








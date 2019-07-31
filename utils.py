# -*- coding: utf-8 -*-
import math
import random
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d

import core,update,visualization,measures




def f(nDimensions,distrib):
    """
        The random distribution for which the quantizer is made.
    """
    if distrib == 'gaussian':
        return np.random.normal(0, 1, nDimensions)
    elif distrib == 'uniform':
        return np.random.uniform(-1. , 1. , nDimensions)
    else:
        print('ERROR ! no distribution defined')


def findRegion(x,hyperplanes):
    """
        Returns the region in which the point x is located.
        The region is modelised as an array of booleans, as described at the top
        of this file.
    """
    return np.array([( np.dot( h , np.append(x,1.) ) < 0 ) for h in hyperplanes])




def squareDistance(x,y):
    """Returns the square of the distance between 2 points x and y."""
    return sum((x-y)**2)



def getExistingRegions(hyperplanes,param,distrib):
    """
        Return the regions formed by the hyperplanes as arrays of booleans
        (similar to centroids()'s first part)
    """
    output = []
    for i in range(param):
        x = f(len(hyperplanes[0])-1 , distrib)
        r = findRegion(x,hyperplanes)
        alreadyExists = False # index of r in the array output
        for j in range(len(output)): # check if r is already registered
            if np.all(output[j] == r):
                alreadyExists = True
                break
        if not alreadyExists:
            output.append(r)
    return np.array(output)

def getMaxMeasureVariations(hp,pCentroids,pMeasure,distrib):
    '''
        Returns the maximal variation of the Distortion measure given the 
        parameters minus 50%, as an estimation of distortion function.
    '''
    e=[]
    for k in range(40):
        e.append(measure("mse",hp,pCentroids,pMeasure,distrib))
    e = np.array(e)
    e = (e - np.mean(e))**2
    return math.sqrt(np.max(e)) * 0.5

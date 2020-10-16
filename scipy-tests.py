# -*- coding: utf-8 -*-

# Tests based on scipy optimization functions

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import core, init, LBG
import visualization as visu
import measures as ms

def S_measure(x,*args):
    """Wrapper of the measure functions"""
    nHyperplanes,nDimensions,m,pCentroids,pMeasure,distrib = args
    global current_cost
    
    # convert x (flattened vector) to hyperplane matrix
    hyperplanes = x.reshape(nHyperplanes,nDimensions+1)
    
    if m == "mse":
        current_cost = ms.MSE(hyperplanes,pCentroids,pMeasure,distrib)
    elif m == "entropy":
        current_cost = ms.negEntropy(hyperplanes,pCentroids,pMeasure,distrib)
    else:
        print("Error: the measure parameter is unknown")
    return current_cost

def S_callback(xk):
    """callback function for scipy test. xk is the state of """
    global measureEvol
    measureEvol.append(current_cost)

def test_scipy(nHyperplanes,nDimensions,distrib,pCentroids,pMeasure,m,initMethod='doublePoint'):
    global measureEvol
    global scipy_method
    measureEvol = []
    if initMethod == 'doublePoint':
        hp = init.doublePoint(nHyperplanes, nDimensions, distrib)
    else: print('init methods not existing')
    
    x0 = hp.flatten() #convert to a 1-D vector    
    result = minimize( S_measure, x0, 
        args=(nHyperplanes,nDimensions,m,pCentroids,pMeasure,distrib), 
        method=scipy_method,
        callback=S_callback,
        options={'maxiter':1000}
    )
    
    
    # store results in a file
    d = 'G' if distrib == 'gaussian' else 'U'
    file = open("scipydata/"+str(d)+"_"+str(nDimensions)+"D_"+str(nHyperplanes)+"Hp_"+m+".txt",'a')
    file.write('\n')
    file.write( str(measureEvol) )
    file.close()


measureEvol = [] # Needs to be declared here so that it can be used module-wide


# scipy_method should be in {'SLSQP', 'CG', 'COBYLA', 'BFGS', 'Powell', 'Nelder-Mead', 'TNC', 'L-BFGS-B'}
scipy_method = 'SLSQP'

repeats = 20
#dimension from 2 to 5
for dimensions in range(2,6):
    for r in range(repeats):
        for k in range(dimensions,8):
            test_scipy(k,dimensions,
                        distrib='gaussian',
                        pCentroids=1000,
                        pMeasure=10000,
                        m='mse',
                        initMethod='doublePoint'
                      )





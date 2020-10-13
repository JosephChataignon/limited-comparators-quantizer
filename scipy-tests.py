# -*- coding: utf-8 -*-

# Tests based on scipy optimization functions

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import core, init, LBG
import visualization as visu
import measures as ms

def S_measure(x,args):
    """Wrapper of the measure functions"""
    nHyperplanes,nDimensions,m,pCentroids,pMeasure,distrib = args
    # convert x (flattened vector) to hyperplane matrix
    hyperplanes = x.reshape(nHyperplanes,nDimensions+1)
    
    if m == "mse":
        return MSE(hyperplanes,pCentroids,pMeasure,distrib)
    elif m == "entropy":
        return negEntropy(hyperplanes,pCentroids,pMeasure,distrib)
    else:
        print("Error: the measure parameter is unknown")

def S_callback(xk):
    """callback function for scipy test. xk is the state of """
    

def test_scipy(nHyperplanes,nDimensions,distrib,pCentroids,pMeasure,m,nIterations,initMethod='doublePoint'):
    if initMethod == 'doublePoint':
        hp = init.doublePoint(nHyperplanes, nDimensions, distrib)
    else: print('init methods not existing')
    
    x0 = hp.flatten() #convert to a 1-D vector    
    result = minimize( S_measure, x0, 
        args=(nHyperplanes,nDimensions,m,pCentroids,pMeasure,distrib), 
        method='COBYLA',
        callback=S_callback,
        options={'maxiter':10000}
        )
    
    # TODO: callbacks of minimization
    
    measureEvol
    # store results in a file
    d = 'G' if distrib == 'gaussian' else 'U'
    file = open("optidata/"+str(d)+"_"+str(nDimensions)+"D_"+str(nHyperplanes)+"Hp_"+m+"_"+str(nIterations)+"iter"+".txt",'a') 
    file.write('\n')
    file.write( str(measureEvol) )
    file.close()







repeats = 25
#dimension from 2 to 5
dimensions=3
for r in range(repeats):
    for k in range(dimensions,8):
        pass
    #    runGenetic(k,dimensions,'gaussian','mse')
    #     testLBG(k,dimensions,'gaussian','mse',10)
    #     testOpti(k,dimensions,distrib='gaussian',pCentroids=1000,pMeasure=10000,m='mse',nIterations=3)
    

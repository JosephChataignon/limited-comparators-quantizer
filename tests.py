# -*- coding: utf-8 -*-

# With n dimensions, each hyperplane P is modelised by an array of length n+1,
# such that (sum_of P[k]*X[k] for_k<n) + P[n] = 0 for any X in P
# Each region (ie, intersection of halfspaces defined by the hyperplanes) is
# modelised by an array of booleans indicating, for each hyperplane, on which
# side the region is, for ex:
# r = [0,1,0], for X in r
# sum(P1[k]*X[k])+P1[n] < 0
# sum(P2[k]*X[k])+P2[n] > 0
# sum(P3[k]*X[k])+P3[n] < 0
# Note that some regions defined this way may be empty or with an area of zero

import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import core,update,visualization,utils,measures




distribution = 'gaussian'




def standardTest2D(distrib,nHyperplanes,initNotRandom=False):
    #initialisation
    if initNotRandom:
        hp = initNotRandom(nHyperplanes,2,1000,5000,20,distribution)
    else:
        hp = init(nHyperplanes,2)
    visualiseHyperplanes(hp,'initial configuration',5,distribution)

    #title
    d = 'G' if distrib == 'gaussian' else 'U'
    t = 'test %d%s'%(nHyperplanes,d)
    print(t,'\n')

    curve,saveHyperplanes = optimisation(hp,1000,5000,9,
        [False,False,3],
        'hps',distribution,
        updateMethod="oneVarInterpolation",
        precisionCheck=False,
        structureCheck=True)
    plt.figure(); plt.plot(curve); plt.title(t); plt.show()
    print('\nsaveHyperplanes:\n',saveHyperplanes)
    print('\n',t,'\n')
standardTest2D('gaussian',3)


def mseEstimations(numberOfEstimations):
    e=[]
    for k in range(numberOfEstimations):
        e.append(measure("mse",hp,1000,10000,'gaussian'))
    plt.figure(); plt.plot(e); plt.title('%d mse estimations')%numberOfEstimations; plt.show()
# mseEstimations(50)








# possible improvements:
# change format of hyperplanes to a 2-coordinates point and and an angle (at least when applying the random changes)
# change varbyvar update so to chose one hyperplane randomly instead of testing them all
# particle filters

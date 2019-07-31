# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d

import core,update,visualization,utils



# mesures of distortion

def MSE(hyperplanes,pCentroids,pMSE,distrib):
    """
        Returns MSE given the hyperplanes separating regions.
        Parameter pCentroids is the number of realisations of f used for
        determining the centroids of each region.
        Parameter pMSE is the number of realisations used for computing the MSE.
    """
    error = 0.
    # countURE = 0 # Unregistrated Region Error
    numberOfPointsUsed = 0
    regionsWithCentroids = centroids(hyperplanes,pCentroids,distrib)
    for i in range(pMSE):
        x = f(len(hyperplanes[0])-1 , distrib)
        # while (x[0]>0.5) or (x[0]<-0.5) or (x[1]>0.5) or (x[1]<-0.5):
        #     x = f(len(hyperplanes[0])-1 , distrib)
        r = findRegion(x,hyperplanes)
        regionRegistered = False
        for j in range(len(regionsWithCentroids)):
            if np.all(regionsWithCentroids[j,0] == r):
                c = regionsWithCentroids[j,1]
                regionRegistered = True
                break;
        if regionRegistered:
            error += squareDistance(x,c)
            numberOfPointsUsed += 1
        # else:
        #     print("ERROR: unregistered region - non adaptative centroids")
        #     countURE += 1
    error /= float(len(x))
    error /= float(numberOfPointsUsed)
    return error

def MSEforDirection(directions, pCentroids, pMSE, distrib):
    """A more efficient way to compute MSE for different directions"""
    directionsMSE = np.zeros(len(directions))
    numberOfPointsUsed = np.zeros(len(directions))
    # define regions and centroids
    regionsWithCentroids = []
    for dir in directions:
        regionsWithCentroids.append(centroids(dir,pCentroids,distrib) )
    # calculate error
    for k in range(pMSE):
        x = f(len(directions[0][0])-1 , distrib) # pick a random point x
        for i in range(len(directions)): #for each direction i
            r = findRegion(x,directions[i])
            regionRegistered = False
            for j in range(len(regionsWithCentroids[i])):
                if np.all(regionsWithCentroids[i][j,0] == r):
                    c = regionsWithCentroids[i][j,1]
                    regionRegistered = True
                    break;
            if regionRegistered:
                directionsMSE[i] += squareDistance(x,c)
                numberOfPointsUsed[i] += 1.
    directionsMSE /= float(len(x))
    directionsMSE /= numberOfPointsUsed
    return directionsMSE













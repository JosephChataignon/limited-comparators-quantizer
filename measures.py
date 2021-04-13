# -*- coding: utf-8 -*-

# mesures of distortion

import numpy as np

import core,utils



def measure(m,hyperplanes,pCentroids,pMeasure,distrib, dataset):
    if m == "mse":
        return MSE(hyperplanes,pCentroids,pMeasure,distrib, dataset)
    elif m == "entropy":
        return negEntropy(hyperplanes,pCentroids,pMeasure,distrib, dataset)
    else:
        print("Error: the measure parameter is unknown")



def negEntropy(hyperplanes,pCentroids,pMeasure,distrib, dataset):
    """
        Returns the opposite (negative) of the overall entropy of the
        hyperplane configuration inducted by the parameter hyperplanes.
        The method used to calculate entropy is similar to that of MSE:
        generate a big amount of random points following the distribution to
        estimate the probabilities.
    """
    regionsWithCentroids = core.centroids(hyperplanes,pCentroids,distrib)
    entropies = np.zeros(len(regionsWithCentroids))
    numberOfPointsUsed = 0
    
    for i in range(pMeasure):
        
        # generate point and find its region
        x = utils.f(len(hyperplanes[0])-1 , distrib, dataset)
        r = utils.findRegion(x,hyperplanes)
        
        # match region with an already known one
        for j in range(len(regionsWithCentroids)):
            if np.all(regionsWithCentroids[j,0] == r):
                
                #increase entropies counter
                entropies[j] += 1
                numberOfPointsUsed += 1
                break;
    
    # adjust values and convert to an actual entropy
    entropies /= float(len(x))
    entropies /= float(numberOfPointsUsed)
    logEntropies = np.log2(entropies) 
    entropies *= logEntropies
    
    return sum(entropies)





def MSE(hyperplanes,pCentroids,pMeasure,distrib, dataset):
    """
        Returns MSE given the hyperplanes separating regions.
        Parameter pCentroids is the number of realisations of f used for
        determining the centroids of each region.
        Parameter pMeasure is the number of realisations used for computing the MSE.
    """
    error = 0.
    numberOfPointsUsed = 0
    regionsWithCentroids = core.centroids(hyperplanes,pCentroids,distrib)
    for i in range(pMeasure):
        x = utils.f(len(hyperplanes[0])-1 , distrib, dataset)
        r = utils.findRegion(x,hyperplanes)
        regionRegistered = False
        for j in range(len(regionsWithCentroids)):
            if np.all(regionsWithCentroids[j,0] == r):
                c = regionsWithCentroids[j,1]
                regionRegistered = True
                break;
        if regionRegistered:
            error += utils.squareDistance(x,c)
            numberOfPointsUsed += 1
    error /= float(len(x))
    error /= float(numberOfPointsUsed)
    return error

def MSEforDirection(directions, pCentroids, pMeasure, distrib, dataset):
    """
        A more efficient way to compute MSE for different directions (for a
        specific update function)
    """
    directionsMSE = np.zeros(len(directions))
    numberOfPointsUsed = np.zeros(len(directions))
    # define regions and centroids
    regionsWithCentroids = []
    for dir in directions:
        regionsWithCentroids.append(core.centroids(dir,pCentroids,distrib) )
    # calculate error
    for k in range(pMeasure):
        x = utils.f(len(directions[0][0])-1 , distrib, dataset) # pick a random point x
        for i in range(len(directions)): #for each direction i
            r = utils.findRegion(x,directions[i])
            regionRegistered = False
            for j in range(len(regionsWithCentroids[i])):
                if np.all(regionsWithCentroids[i][j,0] == r):
                    c = regionsWithCentroids[i][j,1]
                    regionRegistered = True
                    break;
            if regionRegistered:
                directionsMSE[i] += utils.squareDistance(x,c)
                numberOfPointsUsed[i] += 1.
    directionsMSE /= float(len(x))
    directionsMSE /= numberOfPointsUsed
    return directionsMSE

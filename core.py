# -*- coding: utf-8 -*-

import numpy as np

import update,visualization,utils,measures


def init(nHyperplanes,nDimensions):
    """
        Returns a set of hyperplanes with random orientations. nHyperplanes is
        the number of hyperplanes to return, and nDimension the number of
        dimensions of the space.
    """
    return np.random.normal(0,10,(nHyperplanes,nDimensions+1))


def initNotRandom(nHyperplanes, nDimensions, pCentroids, pMeasure, nConfigs, distrib):
    '''
        Initialise hyperplanes by generating nConfigs random configurations and
        selecting the one with lowest measure of distortion.
    '''
    for k in range(nConfigs):
        hp = init(nHyperplanes,nDimensions)
        e = measure("mse",hp,pCentroids,pMeasure,distrib)
        if k == 0:
            minDistortion = e
            minHp = hp
        else:
            if minDistortion >= e:
                minDistortion = e
                minHp = hp
    return minHp



def centroids(hyperplanes,param,distrib):
    """
        Gives the centroid of every non-void region. The returned object is an
        array containing every non-void regions with the coordinates of its
        centroid.
        param is the number of realisations of f on the whole space, that are
        used for computing the centroids. Augmenting it will augment the
        precision but also computation time.
    """
    output = []
    rpr = [] # realisations per region
    for i in range(param):
        x = f(len(hyperplanes[0])-1 , distrib)
        r = findRegion(x,hyperplanes)
        ir = -1 # index of r in the array output
        for j in range(len(output)): # check if r is already registered
            if np.all(output[j][0] == r):
                ir = j
                break
        if ir == -1:
            output.append([r,x])
            rpr.append(1.)
        else:
            output[ir][1] += x
            rpr[ir] += 1.
    # divide the coordinates for each region by the rpr for that region
    for k in range(len(output)):
        output[k][1] /= rpr[k]
    return np.array(output)






def optimisation(hp,pCentroids,pMeasure,pOptimisation,visualisation=[False,False,10],wTitle='',distrib='gaussian',updateMethod='random directions',precisionCheck=False,structureCheck=False):
    '''
        Uses an update function to update the hyperplanes hp, with pCentroids
            and pMeasure as parametersof the functions centroids() and measure()
        pOptimisation: number of iterations
        visualisation = [bool visualiseInitial, bool visualiseSteps, int stepsInterval]
        wTitle: the title used for visualisation
        distrib: the random distribution studied
        updateMethod: the method to use
        If precisionCheck, it is checked wether or not the new config generated
            is better than the previous one.
        If structureCheck, it is checked wether or not the structure of the new
            config (hyperplanes intersections, regions) is different from the
            previous one.
    '''
    measureEvolution = [measure("mse",hp,pCentroids,pMeasure,distrib)]; saveHyperplanes = [hp]

    if visualisation[0]:
        visualiseHyperplanes(hp,wTitle+', iteration= %d, error= %f'%(0,measureEvolution[-1]),5,distrib)

    # optimization steps
    for k in range(1,pOptimisation+1):

        if np.all(saveHyperplanes[-1] == hp):
            print('identical configurations ***')
        print('optimisation function: iteration',k,'of',pOptimisation)
        saveHyperplanes.append(hp)

        u = choseUpdateFunction(updateMethod,k)
        if u == 'oneVarInterpolation':
            for i in range(10):
                hp,newMeasure = oneVarInterpolation(hp,pCentroids,pMeasure*k,k,measureEvolution[-1],distrib,precisionCheck,structureCheck)
        elif u == 'indVarByVar':
            for i in range(10):
                hp,newMeasure = updateHyperplanes(hp,pCentroids,pMeasure*k,k,measureEvolution[-1],u,distrib,precisionCheck,structureCheck)
        else:
            hp,newMeasure = updateHyperplanes(hp,pCentroids,pMeasure*k,k,measureEvolution[-1],u,distrib,precisionCheck,structureCheck)

        measureEvolution.append(newMeasure)

        # display result
        if (k % visualisation[2] == 0) and visualisation[1]:
            visualiseHyperplanes( hp , wTitle+', iteration= %d, error= %f'%(k,measureEvolution[-1]) , 5 , distrib)
        print('measureEvolution[-1]',measureEvolution[-1])

    return measureEvolution,saveHyperplanes

# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d

import core,utils
import visualization as visu
import measures as ms





# Functions to update the configuration

def getAlpha(optimisationIteration,method='linear decrease'):
    '''returns alpha, the learning speed parameter'''
    if method == 'linear decrease':
        alpha = 10-optimisationIteration
        if alpha<1:
            alpha = 1.
    elif method == 'exponential decrease':
        alpha = np.exp(-optimisationIteration/10.)
    return alpha

def oneVarInterpolation(hp,pCentroids,pMeasure,optimisationIteration,lastMeasure,distrib,m,precisionCheck,structureCheck,var=None):
    '''
        Use interpolation to estimate the optimal value of a randomly chosen
        variable.
    '''
    # Note: tried applying the function recursively on its local minima, not a good idea
    #       (lack of precision+infinite time)

    if var == None:
        i = np.random.random_integers(0,len(hp)-1)
        j = np.random.random_integers(0,len(hp[0])-1)
        var = hp[i][j]
    else:
        i = var[0]
        j = var[1]
        var = var[2]
    x = []
    y = []
    for k in [-30.,-20.,-10.,-5.,-2.,-1.5,-1.,-0.75,-0.5,0.01,0.25,0.5,0.75,1.,1.5,2.,3.,6.,10.,20.,30.,40.,50.]:
        direction = hp; direction[i][j] = var * k
        x.append( var * k )
        y.append( ms.measure(m, direction ,pCentroids ,pMeasure , distrib ) )
    #take successive 4 x's to interpolate
    xmins = []#np.zeros(0)
    ymins = []#np.zeros(0)
    for k in range(len(x)-3):
        xReduced = x[k:k+4]
        yReduced = y[k:k+4]
#        print(xReduced)
#        print(yReduced)
        fApprox = interp1d(xReduced, yReduced, kind='cubic')
        xnew = np.linspace(min(xReduced), max(xReduced), num=100, endpoint=True)
        ynew = fApprox(xnew)
        xmins.append( xnew[np.argmin(ynew)] )
        ymins.append( min(ynew) )
    smallestValues = [ [xmins[l] for l in np.argpartition(ymins,5)[:5]] , [ymins[l] for l in np.argpartition(ymins,5)[:5]] ]
    smallestValuesMeasure = []
    smallDirections = []
    for smallX in smallestValues[0]:
        direction = hp; direction[i][j] = smallX; smallDirections.append(direction)
        smallestValuesMeasure.append( ms.measure(m,direction,pCentroids,pMeasure*10,distrib) )
    n = np.argmin( smallestValuesMeasure )
    return smallDirections[n] , smallestValuesMeasure[n]
    # Visualisation (comment return statement to activate)
    # f3 = interp1d(x, y, kind='cubic')
    # xnew = np.linspace(min(x), max(x), num=500, endpoint=True)
    # ynew = f3(xnew)
    # plt.figure(), plt.plot(x,y,'o',xnew,ynew,'-', xmins, ymins ,'v', smallestValues[0], smallestValues[1], 'v', smallestValues[0], smallestValuesMeasure, 's' ), plt.show()
# oneVarInterpolation(init(3,2),1000,10000,1,10,'gaussian',False,False)

def directionsVarByVar(hp,numberOfDirections,optimisationIteration,distrib,m,pCentroids,structureCheck):
    '''
        Update the hyperplanes by looking at each parameter's potential
        improvements individually. 2D only
    '''
    if structureCheck:
        regionsStructure = utils.getExistingRegions(hp,pCentroids,distrib)
    alpha = getAlpha(optimisationIteration,'linear decrease')
    directions = [hp]
    for i in range(len(hp)):
        for j in range(len(hp[i])):
            for k in range(numberOfDirections):
                newdirection = hp
                newdirection[i][j] += np.random.normal(0,alpha)
                if structureCheck:
                    if np.all( regionsStructure == utils.getExistingRegions(newdirection,pCentroids,distrib) ):
                        directions.append(newdirection)
                else:
                    directions.append(newdirection)
    return directions

def directionsGlobalRandom(hp,numberOfDirections,optimisationIteration):
    '''
        Generates random moves for the hyperplanes.
    '''
    # initialise parameters and directions
    alpha = getAlpha(optimisationIteration)
    directions = [hp]
    for k in range(numberOfDirections):
        directions.append( hp + np.random.random(np.shape(hp)) * alpha )
    return directions

def directionsIndVarByVar(hp,numberOfDirections,optimisationIteration):
    '''
        Pick one variable and and generate different directions for it.
    '''
    directions = [hp]
    i = np.random.random_integers(0,len(hp)-1)
    j = np.random.random_integers(0,len(hp[0])-1)
    x = hp[i][j]
    for k in range(1,10):
        directionPosSm = hp; directionPosGr = hp; directionNegSm = hp; directionNegGr = hp
        directionPosSm[i][j] = x * (1/k)
        directionPosGr[i][j] = x * (1+k/2)
        directionNegSm[i][j] = x * (-1/k)
        directionNegGr[i][j] = x * -(1+k/2)
        directions.append(directionPosSm)
        directions.append(directionPosGr)
        directions.append(directionNegSm)
        directions.append(directionNegGr)
    return directions


def choseUpdateFunction(updateMethod,iteration):
    '''Choses the update function to use based on the parameter updateMethod'''
    if updateMethod == 'globalRandom':
        return 'globalRandom'
    elif updateMethod == 'varByVar':
        return 'varByVar'
    elif updateMethod == "indVarByVar":
        return "indVarByVar"
    elif updateMethod == "oneVarInterpolation":
        return "oneVarInterpolation"
    elif updateMethod == 'mixed1':
        if (iteration % 2 == 1) and (iteration < 8) :
            return 'globalRandom'
        else:
            return 'varByVar'
    elif updateMethod == 'mixed random':
        if np.random.uniform() < np.exp(-iteration/10.):
            return 'globalRandom'
        else:
            return 'varByVar'

def updateHyperplanes(hp,pCentroids,pMeasure,optimisationIteration,lastMSE,updateFunction,distrib,m,precisionCheck,structureCheck):
    '''
        Actually update the hyperplanes by selecting the lowest MSE between
        several directions. Only uses MSE as measure.
    '''
    # generate directions to test
    if updateFunction == 'globalRandom':
        directions = directionsGlobalRandom(hp,10+10*optimisationIteration,optimisationIteration)
    elif updateFunction == 'varByVar':
        directions = directionsVarByVar(hp,10+10*optimisationIteration,optimisationIteration,distrib,pCentroids,structureCheck)
    elif updateFunction == 'indVarByVar':
        directions = directionsIndVarByVar(hp,10+10*optimisationIteration,optimisationIteration)
    # evaluate distortion
    if m =='mse':
        directionsMeasure = ms.MSEforDirection(directions, pCentroids, pMeasure , distrib)
    elif m == 'entropy':
        print('il y a un probleme')
    # select the lowest MSE configuration
    indexMin = np.argmin(directionsMeasure)
    newMSE = ms.measure(m, directions[indexMin] ,pCentroids ,pMeasure*2 , distrib )
    return checkPrecision(hp,pCentroids,pMeasure,optimisationIteration,distrib,m,directions,precisionCheck,newMSE,lastMSE,indexMin)

def checkPrecision(hp,pCentroids,pMeasure,optimisationIteration,distrib,m,directions,precisionCheck,structureCheck,newMeasure,lastMeasure,indexMin):
    """Deprecated - might not work"""
    if not precisionCheck:
        if newMeasure > lastMeasure :
            print('Error: new Distortion bigger than at previous iteration')
            indexMin = 0 # keep previous configuration
            return (hp,lastMeasure)
        else:
            return (directions[indexMin],newMeasure)
    else:
    #check that new configuration is better than the old one, taking imprecision into account
        variations = utils.getMaxMeasureVariations(hp,pCentroids,pMeasure,distrib)
        if newMeasure < lastMeasure-variations:
            return directions[indexMin],newMeasure
        else:
            print('Not enough precision - going deeper*******************')
            newHp,newMeasure = updateHyperplanes(hp,pCentroids,pMeasure*10,optimisationIteration,lastMeasure,updateFunction,distrib,m,precisionCheck,structureCheck)
            return newHp,newMeasure








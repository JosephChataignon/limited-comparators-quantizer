# -*- coding: utf-8 -*-
import math
import random
import scipy
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from scipy.interpolate import interp1d

import core,visualization,utils,measures

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

def oneVarInterpolation(hp,pCentroids,pMSE,optimisationIteration,lastMSE,distrib,precisionCheck,structureCheck,var=None):
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
    for k in [-30.,-20.,-10.,-5.,-2.,-1.5,-1.,-0.75,-0.5,0,0.25,0.5,0.75,1.,1.5,2.,3.,6.,10.,20.,30.,40.,50.]:
        direction = hp; direction[i][j] = var * k
        x.append( var * k )
        y.append( MSE( direction ,pCentroids ,pMSE , distrib ) )
    #take successive 4 x's to interpolate
    xmins = []#np.zeros(0)
    ymins = []#np.zeros(0)
    for k in range(len(x)-3):
        xReduced = x[k:k+4]
        yReduced = y[k:k+4]
        fApprox = interp1d(xReduced, yReduced, kind='cubic')
        xnew = np.linspace(min(xReduced), max(xReduced), num=100, endpoint=True)
        ynew = fApprox(xnew)
        xmins.append( xnew[np.argmin(ynew)] )
        ymins.append( min(ynew) )
    smallestValues = [ [xmins[l] for l in np.argpartition(ymins,5)[:5]] , [ymins[l] for l in np.argpartition(ymins,5)[:5]] ]
    smallestValuesMSE = []
    smallDirections = []
    for smallX in smallestValues[0]:
        direction = hp; direction[i][j] = smallX; smallDirections.append(direction)
        smallestValuesMSE.append( MSE(direction,pCentroids,pMSE*10,distrib) )
    n = np.argmin( smallestValuesMSE )
    return smallDirections[n] , smallestValuesMSE[n]
    # Visualisation (comment return statement to activate)
    # f3 = interp1d(x, y, kind='cubic')
    # xnew = np.linspace(min(x), max(x), num=500, endpoint=True)
    # ynew = f3(xnew)
    # plt.figure(), plt.plot(x,y,'o',xnew,ynew,'-', xmins, ymins ,'v', smallestValues[0], smallestValues[1], 'v', smallestValues[0], smallestValuesMSE, 's' ), plt.show()
# oneVarInterpolation(init(3,2),1000,10000,1,10,'gaussian',False,False)

def directionsVarByVar(hp,numberOfDirections,optimisationIteration,distrib,pCentroids,structureCheck):
    '''
        Update the hyperplanes by looking at each parameter's potential
        improvements individually. 2D only
    '''
    if structureCheck:
        regionsStructure = getExistingRegions(hp,pCentroids,distrib)
    alpha = getAlpha(optimisationIteration,'linear decrease')
    directions = [hp]
    for i in range(len(hp)):
        for j in range(len(hp[i])):
            for k in range(numberOfDirections):
                newdirection = hp
                newdirection[i][j] += np.random.normal(0,alpha)
                if structureCheck:
                    if np.all( regionsStructure == getExistingRegions(newdirection,pCentroids,distrib) ):
                        directions.append(newdirection)
                else:
                    directions.append(newdirection)
    return directions

def directionsGlobalRandom(hp,numberOfDirections,optimisationIteration):
    '''
        Update the hyperplanes by looking at random moves and selecting the one
        lowering the most the error.
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

def updateHyperplanes(hp,pCentroids,pMSE,optimisationIteration,lastMSE,updateFunction,distrib,precisionCheck,structureCheck):
    '''
        Actually update the hyperplanes by selecting the lowest MSE between
        several directions
    '''
    # generate directions to test
    if updateFunction == 'globalRandom':
        directions = directionsGlobalRandom(hp,10+10*optimisationIteration,optimisationIteration)
    elif updateFunction == 'varByVar':
        directions = directionsVarByVar(hp,10+10*optimisationIteration,optimisationIteration,distrib,pCentroids,structureCheck)
    elif updateFunction == 'indVarByVar':
        directions = directionsIndVarByVar(hp,10+10*optimisationIteration,optimisationIteration)
    # evaluate MSE
    directionsMSE = MSEforDirection(directions, pCentroids, pMSE , distrib)
    # select the lowest MSE configuration
    indexMin = np.argmin(directionsMSE)
    newMSE = MSE( directions[indexMin] ,pCentroids ,pMSE*2 , distrib )
    return checkPrecision(hp,pCentroids,pMSE,distrib,directions,precisionCheck,newMSE,lastMSE,indexMin)

def checkPrecision(hp,pCentroids,pMSE,distrib,directions,precisionCheck,newMSE,lastMSE,indexMin):
    if not precisionCheck:
        if newMSE > lastMSE :
            print('Error: new MSE bigger than at previous iteration')
            indexMin = 0 # keep previous configuration
            return (hp,lastMSE)
        else:
            return (directions[indexMin],newMSE)
    else:
    #check that new configuration is better than the old one, taking imprecision into account
        variations = getMaxMSEVariations(hp,pCentroids,pMSE,distrib)
        if newMSE < lastMSE-variations:
            return directions[indexMin],newMSE
        else:
            print('Not enough precision - going deeper*******************')
            newHp,newMSE = updateHyperplanes(hp,pCentroids,pMSE*10,optimisationIteration,lastMSE,updateFunction,distrib)
            return newHp,newMSE


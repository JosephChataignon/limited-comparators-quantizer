# -*- coding: utf-8 -*-
import numpy as np
import random

import measures as ms




def f(nDimensions,distrib,dataset=None):
    '''
        The random distribution for which the quantizer is made.
    '''
    if distrib == 'gaussian':
        return np.random.normal(0, 1, nDimensions)
    elif distrib == 'uniform':
        return np.random.uniform(-1. , 1. , nDimensions)
    elif distrib == 'dataset':
        return next(dataset)
    else:
        print('ERROR ! no distribution defined')    

def load_dataset(filename='dataset/noaa-daily-weather-data.csv',
                 dimension=3,loop=False):
    '''
        loads the full dataset at once
        loop = True makes the generator loop back to the beginning of the file
        NOAA dataset is from:
        https://data.opendatasoft.com/explore/dataset/noaa-daily-weather-data%40public/information/?q=&refine.country_code=FR
    '''
    with open(filename,'r') as f:
        lines = f.readlines()
    while True:
        c, datapoint = 0, []
        random.shuffle(lines)
        while len(datapoint) < dimension:
            try:
                datapoint.append(float(lines[c].strip()))
            except ValueError: pass
            c += 1
        yield datapoint
        if c >= len(lines)-dimension: continue
            


def distribCenter(nDimensions,distrib):
    '''
        The center of the random distribution distrib.
    '''
    if distrib == 'gaussian':
        return np.zeros(nDimensions)
    elif distrib == 'uniform':
        return np.zeros(nDimensions)
    else:
        print('ERROR ! no distribution defined')

def hyperplaneFromPoints(points):
    '''
        Returns the hyperplane passing by the set of points "points".
        points must be a square matrix (n points of dimension n).
    '''
    n = points.shape[0]
    k = np.ones((n,1))
    a = np.matrix.dot(np.linalg.inv(points), k)
    a = a.flatten()
    return np.concatenate((a,[-1.]))

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
        e.append(ms.measure("mse",hp,pCentroids,pMeasure,distrib))
    e = np.array(e)
    e = (e - np.mean(e))**2
    return np.sqrt(np.max(e)) * 0.5

def hyperplaneOrder(hp,distrib,orderBy):
    '''
        Returns a value that can be used to order hyperplanes by similarity.
        orderBy is the type of order that is used to order the hyperplanes. 
        Different ordering methods can be used, giving different results. The 
        simplest one is the distance of the hyperplane to the coordinate zero.
    '''
    if orderBy == "distanceToZero": # distance to the coordinate zero
        print("not implemented yet")
    elif orderBy == "distanceToDistribCentroid":# distance to the centroid of the distribution
        print("not implemented yet")

def normalizeHpNotation(hp):
    ''' normalizes hp such that its last value is 1. '''
    hp = np.array(hp)
    return hp / hp[-1]

def normalizeConfigNotation(config):
    ''' normalize every hyperplane of config such that its last value is 1. '''
    return np.array([hp/hp[-1] for hp in config])

def norm1(hp):
    '''Norm 1 of hp'''
    return hp / np.sqrt(np.sum(hp**2))

def angleHps(hp1,hp2):
    '''returns the angle between 2 hyperplanes (need to normalize them first)'''
    return np.abs(np.arccos(np.dot(hp1,hp2)))

def nearestPoint(point, hp):
    '''returns the point of hp that is nearest to point'''
    p1 = np.append(point , 1.)
    p2 = np.dot(p1,hp)
    p3 = np.sum( hp[:-1]**2 )
    l = -p2/p3
    return point + l * hp[:-1]

def distanceBetweenHps(hp1, hp2, distrib):
    '''
        Measure the distance between 2 hyperplanes as the distance of their 
        points nearest to the distribution center
    '''
    point1 = nearestPoint( distribCenter( len(hp1)-1 , distrib ) , hp1 )
    point2 = nearestPoint( distribCenter( len(hp2)-1 , distrib ) , hp2 )
    return np.sqrt(squareDistance(point1,point2))

def dissimilarityHps(hp1, hp2, distrib):
    '''
        Measure the dissimilarity between 2 hyperplanes as the distance of their 
        points nearest to the distribution center, multiplied with their angle.
    '''
    angle = angleHps(norm1(hp1),norm1(hp2))
    distance = distanceBetweenHps(hp1,hp2,distrib)
    return angle * distance

def distancePoint2Hp(point, hp):
    '''
        Distance between point and hyperplane.
        d = abs(H.p) / norm(H)
    '''
    hp = np.array(hp)
    p1 = np.append(point , 1.)
    p2 = np.abs( np.dot(p1,hp) )
    p3 = np.sum( hp[:-1]**2 )
    return p2/np.sqrt(p3)











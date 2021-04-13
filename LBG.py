import numpy as np
import matplotlib.pyplot as plt
import utils


def initLBG(nRegions,nDimensions,distrib,dataset):
    if distrib == 'gaussian':
        return np.random.normal(size=(nRegions,nDimensions))
    elif distrib == 'uniform':
        return np.random.uniform(size=(nRegions,nDimensions))
    elif distrib == 'dataset':
        ds = utils.load_dataset(loop=True)
        return np.array([next(ds) for x in range(nRegions)])
    else:
        print('ERROR: no distribution defined')

def findRegion(point,regions):
    '''returns the index of the region containing point'''
    for k in range(len(regions)): # for each region k
        isin = True # whether point is in the region k
        for l in range(len(regions)): #for each region l distinct from k
            if k != l:
                if np.dot(regions[k,l], np.append(point,1.0)) > 0: #if point out of bounds
                    isin = False
                    break
        if isin:
            return k
    return -1

def MSE(regions, germs, pMeasure, distrib, dataset):
    '''
        Returns the mean squared error based on an approximation made with 
        random points generated according to the random distribution being 
        studied.
    '''
    nDimensions = len(regions[0,0])-1
    error = 0.
    for k in range(pMeasure):
        x = utils.f(nDimensions, distrib, dataset)
        r = findRegion(x, regions)
        error += utils.squareDistance(x,germs[r])
    error /= float(nDimensions)
    error /= float(pMeasure)
    return error

def centroids(regions, pCentroids, distrib, dataset):
    '''
        This function computes the positions of the centroids of regions, based 
        on an estimation of pCentroids points.
    '''
    nRegions, nDimensions = len(regions), len(regions[0,0])-1 # number of regions or germs, and dimensions
    germs = np.zeros((nRegions,nDimensions))
    rpr = np.ones((nRegions)) #realisations per region - should be zero but then shit happens
    for _ in range(pCentroids):
        x = utils.f(nDimensions , distrib, dataset)
        r = findRegion(x, regions)
        germs[r] += x
        rpr[r] += 1
    if 0. in rpr:
        print('attention, a 0 in rpr:',rpr)#!!!!!!!!!!
    germs = [germ/x for germ,x in zip(germs,rpr)]
    return np.array( germs )

def adjustRegions(germs):
    '''
        This function adjusts the delimitations of the regions based on their
        center. It builds a Voronoi diagram from an array containing a germ for 
        each region.
        Regions delimiters are a set of variables [a1,a2,...an,b] , which are 
        parameters of the equation a1*x1 + a2*x2 + ... + an*xn + b <= 0
    '''
    nRegions, nDimensions = len(germs), len(germs[0]) # number of regions or germs, and dimensions
    regions = np.zeros( (nRegions,nRegions,nDimensions+1) )
    for i in range(nRegions):
        for j in range(nRegions):
            if i != j:
                regions[i,j] = np.append( 2*(germs[j]-germs[i]) , np.sum(germs[i]**2-germs[j]**2) )
                if regions[i,j,-1] != 0:
                    regions[i,j,:] = regions[i,j,:] / np.abs(regions[i,j,-1])
    return regions

def maxlloyd(germs,iterations,pCentroids,pMeasure,distrib,dataset):
    '''
        Actual implementation of Max-Lloyd algorithm.
        griddimensions indicates the number of points on each axis that are used for
        estimating the centroid of a region.
    '''
    nRegions, nDimensions = len(germs), len(germs[0]) # number of regions and dimensions
    print("LBG algorithm with",nRegions,"regions and",nDimensions,"dimensions.")
    regions = adjustRegions(germs)
    measureEvolution = [MSE(regions,germs,pMeasure,distrib,dataset)]
    savegerms=[germs]
    for k in range(iterations):
        print('LBG iteration',k,'of',iterations)
        # Step 1: adjust regions
        regions = adjustRegions(germs)
        # Step 2: adjust germs
        germs = centroids(regions, pCentroids, distrib, dataset)
        # measure distortion
        measureEvolution.append(MSE(regions,germs,pMeasure,distrib,dataset))
        pMeasure = int(np.sqrt(2)*pMeasure)
    print("LBG finished")
    return measureEvolution,germs,regions

#germs = initLBG(7,2,'gaussian')
#measureEvolution,germs,regions = maxlloyd(germs,5,10000,100000,'gaussian')

def displayregions(griddimensions,regions):
    '''visualizes the regions, in 2D only, for a number of regions <= 8'''
    plt.figure(2)
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', 'w']
    for i in range(griddimensions[0]):
        for j in range(griddimensions[1]):
            i2 = float(i)*10.0/griddimensions[0]-5 ; j2 = float(j)*10.0/griddimensions[1]-5; point = np.array([i2,j2])
            plt.plot(i2,j2,marker='o',color=colors[findRegion(point,regions)])
    plt.show()





















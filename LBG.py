import numpy as np

import utils



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

def MSE(regions, germs, pMeasure, distrib):
    '''
        Returns the mean squared error based on an approximation made with 
        random points generated according to the random distribution being 
        studied.
    '''
    nDimensions = len(regions[0,0])-1
    error = 0.
    for k in range(pMeasure):
        x = utils.f(nDimensions, distrib)
        r = findRegion(x, regions)
        error += utils.squareDistance(x,germs[r])
    error /= float(nDimensions)
    error /= float(pMeasure)
    return error

def centroids(regions, pCentroids, distrib):
    '''
        This function computes the positions of the centroids of regions, based 
        on an estimation of pCentroids points.
    '''
    nRegions, nDimensions = len(regions), len(regions[0,0])-1 # number of regions or germs, and dimensions
    germs = np.zeros((nRegions,nDimensions))
    rpr = np.zeros() #realisations per region
    for _ in range(pCentroids):
        x = utils.f(nDimensions , distrib)
        r = findRegion(x, regions)
        germs[r] += x
        rpr[r] += 1
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
    regions = np.zeros( (nRegions,nRegions,nDimensions) )
    for i in range(nRegions):
        for j in range(nRegions):
            if i != j:
                regions[i,j] = np.append( 2*(germs[j]-germs[i]) , np.sum((germs[i]-germs[j])**2) )
                if regions[i,j,-1] != 0:
                    regions[i,j,:] = regions[i,j,:] / np.abs(regions[i,j,-1])
    return regions

def maxlloyd(germs,iterations,pCentroids,pMeasure,distrib):
    '''
        Actual implementation of Max-Lloyd algorithm.
        griddimensions indicates the number of points on each axis that are used for
        estimating the centroid of a region.
    '''
    nRegions, nDimensions = len(germs), len(germs[0]) # number of regions and dimensions
    print("LBG algorithm, number of regions:", nRegions)
    regions = np.random.normal(0,1,(nRegions,nRegions,nDimensions)) #random init with a normal distribution !
    germs = centroids(regions, pCentroids, distrib)
    measureEvolution = [MSE(regions,germs,pMeasure, distrib)]
    for k in range(iterations):
        # Step 1: adjust regions
        regions = adjustRegions(germs)
        # Step 2: adjust germs
        germs = centroids(regions, pCentroids, distrib)
        # measure distortion
        measureEvolution.append(MSE(regions,germs,pMeasure, distrib))
    return measureEvolution,germs,regions




























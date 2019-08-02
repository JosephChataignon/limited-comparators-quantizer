# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

import core,utils





def visualiseRegions(hp,iteration,error,nbPoints,width,distrib):
    '''
        visualize the regions as a colored grid, with their centroids as
        triangles, 2D only
    '''
    plt.figure()
    c = core.centroids(hp,1000,distrib)
    regions=[]
    for k in c:
        regions.append(k[0])
    colors = ['b', 'c', 'y', 'm', 'r', 'g', 'k', 'w']
    for i in range(nbPoints):
        for j in range(nbPoints):
            i2 = float(i)*width/nbPoints-(width/2.) ; j2 = float(j)*width/nbPoints-(width/2.); point = np.array([i2,j2])
            region = utils.findRegion(point,hp)
            for z in range(len(regions)):
                if np.all(region==regions[z]):
                    plt.plot(i2,j2,marker='.',color=colors[z])
    for k in range(len(c)):
        plt.plot(c[k][1][0],c[k][1][1],marker='v',color=colors[k+1%7])
    plt.title('iteration %d, error = %d'%(iteration,error))
    print(error)
    plt.show()



def visualiseHyperplanes(hps,windowTitles,width,distrib,numberPlots=1):
    '''visualize the hyperplanes, 2D only'''
    plt.figure()
    for k in range(numberPlots):
        if numberPlots == 1:
            wTitle = windowTitles
            hp = hps
        else:
            wTitle = windowTitles[k]
            hp = hps[k]
            plt.subplot(int(np.sqrt(numberPlots)),int(np.sqrt(numberPlots))+1,k+1)
        c = core.centroids(hp,1000,distrib)
        plt.plot(0,0,marker='o',color='silver') # plot center of the distribution
        if distrib == 'uniform': # plot the limits of the uniform distribution
            plt.plot( [1,1],[-1,1],[-1,1],[-1,-1],[-1,-1],[1,-1],[1,-1],[1,1] , color='silver')

        for i in range(len(hp)): # plot hyperplanes
            plt.plot( [-width,width] , [-1.*(hp[i][2]-hp[i][0]*width)/hp[i][1] , -1.*(hp[i][2]+hp[i][0]*width)/hp[i][1] ] )

        for k in range(len(c)): # plot centroids
            plt.plot(c[k][1][0],c[k][1][1],marker='+',color='k')
        plt.axis([-width,width,-width,width])
        plt.title(wTitle)
    plt.show()
# visualiseHyperplanes(init(3,2),'test',10,'uniform')


# -*- coding: utf-8 -*-

# With n dimensions, each hyperplane P is modeled by an array of length n+1,
# such that (sum_of P[k]*X[k] for_k<n) + P[n] = 0 for any X in P
# Each region (ie, intersection of halfspaces defined by the hyperplanes) is
# modelised by an array of booleans indicating, for each hyperplane, on which
# side the region is, for ex:
# r = [0,1,0], for X in r
# sum(P1[k]*X[k])+P1[n] < 0
# sum(P2[k]*X[k])+P2[n] > 0
# sum(P3[k]*X[k])+P3[n] < 0
# Note that some regions defined this way may be empty or with an area of zero

import matplotlib.pyplot as plt
import numpy as np

import core, init, LBG
import visualization as visu
import measures as ms



def standardTest2D(distrib,m,nHyperplanes,initNotRandom=False):
    '''
        With distribution distrib and measure m, run optimisation for 
        nHyperplanes hyperplanes and 2 dimensions.
        Plots a figure of the evolution of distortion.
    '''
    #initialisation
    if initNotRandom:
        hp = init.poolSelect(nHyperplanes,2,800,2500,20,distrib,m)
    else:
        hp = init.normal(nHyperplanes,2)
    visu.visualiseHyperplanes(hp,'initial configuration',5,distrib)

    #title
    d = 'G' if distrib == 'gaussian' else 'U'
    t = 'test %d%s'%(nHyperplanes,d)
    print(t,'\n')

    curve,saveHyperplanes = core.optimisation(hp,100,500,6,
        [True,True,2],
        'hps',distrib,m,
        updateMethod="oneVarInterpolation",
        precisionCheck=False,
        structureCheck=False)
    plt.figure(); plt.plot(curve); plt.title(t); plt.show()
    #print('\nsaveHyperplanes:\n',saveHyperplanes)
    #print('\n',t,'\n')
#standardTest2D(distribution, measure, 5)



def multipleTest2D(numberOfTests,distrib,m,nHyperplanes,initNotRandom=False):
    '''
        Same as standardTest2D() but runs several times the optimisation.
    '''
    measureEvolutions = []
    hyperplanesEvolutions = []
    for k in range(numberOfTests):
        #initialisation
        if initNotRandom:
            hp = init.poolSelect(nHyperplanes,2,800,2500,20,distrib,m)
        else:
            hp = init.normal(nHyperplanes,2)
        #visu.visualiseHyperplanes(hp,'initial configuration',5,distribution)
    
        #title
        d = 'G' if distrib == 'gaussian' else 'U'
        t = 'test %d%s, model %d'%(nHyperplanes,d,k)
        print(t,'\n')
        
        #optimisation
        curve,saveHyperplanes = core.optimisation(hp,1000,10000,15,
            [False,False,5],
            t,distrib,m,
            updateMethod="oneVarInterpolation",
            precisionCheck=False,
            structureCheck=False)
        measureEvolutions.append(curve)
        hyperplanesEvolutions.append(saveHyperplanes)
    
    #display results
    stepsToDisplay = [0,10,-1]
    for model in hyperplanesEvolutions:
        for step in stepsToDisplay:
            visu.visualiseHyperplanes(model[step],'optimisation step %d'%(step),5,distrib)
    plt.figure(); 
    for curve in measureEvolutions:
        plt.plot(curve); 
    plt.title(t); plt.show()
# multipleTest2D(4, distribution, measure, 5)





def testOpti(nHyperplanes,nDimensions,distrib,pCentroids,pMeasure,m,nIterations,initMethod='doublePoint'):
    '''Execute optimization function'''
    #initialisation
    if initMethod == 'doublePoint':
        hp = init.doublePoint(nHyperplanes, nDimensions, distrib)
    else:
        print('need to code other init methods')
    
    measureEvol,saveHyperplanes = core.optimisation(hp,pCentroids,pMeasure,nIterations,
                                            [False,False,1],
                                            'test higher dimensions',distrib,m,
                                            updateMethod="oneVarInterpolation",
                                            precisionCheck=False,
                                            structureCheck=False)
    # store results in a file
    d = 'G' if distrib == 'gaussian' else 'U'
    file = open("optidata/"+str(d)+"_"+str(nDimensions)+"D_"+str(nHyperplanes)+"Hp_"+m+"_"+str(nIterations)+"iter"+".txt",'a') 
    file.write('\n')
    file.write( str(measureEvol) )
    file.close()




def mseEstimations(numberOfEstimations,m="mse"):
    ''' estimations of MSE on the same set of points to visualise its variance '''
    e=[]
    hp = init.normal(3,2)
    for k in range(numberOfEstimations):
        e.append(ms.measure(m,hp,1000,10000,'gaussian'))
    plt.figure(); plt.plot(e); plt.title('%d mse estimations')%numberOfEstimations; plt.show()
# mseEstimations(50)


def runGenetic(nHyperplanes,nDimensions,distrib,measure):
    order = 'dissimilarity'
    pCentroids = 10000
    pMeasure   = 30000
    genIter = 30 #iterations of the genetic algorithm
    nConfigs = 100
    
    hps , geneticMeasureEvolution = init.genetic(
            nHyperplanes,nDimensions, pCentroids, pMeasure,distrib,measure,
            nConfigs,genIter,1,1,order)
    # configs number, total iterations, crossover points, mutation param
    
    # save genetic algorithm results
    d = 'G' if distrib == 'gaussian' else 'U'
    file = open("Initialisation_performance_data/genetic_evol_data_"+str(genIter)+"Iter/"
                +"genetic_"
                +d+"_"+str(nDimensions)+"D_"+str(nHyperplanes)+"Hp_"+measure
                +"_"+nConfigs+"geneticConfigs"+"_"+str(genIter)+"geneticIter"+"_1crossover"
                +".txt",'a') 
    file.write('\n')
    file.write( str(geneticMeasureEvolution) )
    file.close()

def testLBG(nRegions,nDimensions, distrib,measure,iterations):
    
    pCentroids = 10000
    pMeasure   = 30000
    germs = LBG.initLBG(nRegions,nDimensions,'gaussian')
    measureEvolution,germs,regions = LBG.maxlloyd(germs,iterations,pCentroids,pMeasure,'gaussian')
    d = 'G' if distrib == 'gaussian' else 'U'
    file = open("LBGdata/"
                +d+"_"+str(nDimensions)+"D_"+str(nRegions)+"Reg_"+measure+"_"
                +str(iterations)+"iter"
                +".txt",'a') 
    file.write('\n')
    file.write( str(measureEvolution) )
    file.close()



repeats = 25
#dimension from 2 to 5
dimensions=3
for r in range(repeats):
    # for k in range(dimensions,8):
    #     runGenetic(k,dimensions,'gaussian','mse')
    #     testLBG(k,dimensions,'gaussian','mse',10)
    #     testOpti(k,dimensions,distrib='gaussian',pCentroids=1000,pMeasure=10000,m='mse',nIterations=3)
    runGenetic(7, dimensions, 'gaussian', 'mse')#7hp in 3D


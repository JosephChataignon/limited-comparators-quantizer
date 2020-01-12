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

import core, init
import visualization as visu
import measures as ms


distribution = 'uniform'
measure = 'entropy'




def standardTest2D(distrib,m,nHyperplanes,initNotRandom=False):
    '''
        With distribution distrib and measure m, run optimisation for 
        nHyperplanes hyperplanes and 2 dimensions.
    '''
    #initialisation
    if initNotRandom:
        hp = init.poolSelect(nHyperplanes,2,800,2500,20,distribution,m)
    else:
        hp = init.normal(nHyperplanes,2)
    visu.visualiseHyperplanes(hp,'initial configuration',5,distribution)

    #title
    d = 'G' if distrib == 'gaussian' else 'U'
    t = 'test %d%s'%(nHyperplanes,d)
    print(t,'\n')

    curve,saveHyperplanes = core.optimisation(hp,100,500,6,
        [True,True,2],
        'hps',distribution,measure,
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
            hp = init.poolSelect(nHyperplanes,2,800,2500,20,distribution,m)
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
            t,distribution,measure,
            updateMethod="oneVarInterpolation",
            precisionCheck=False,
            structureCheck=False)
        measureEvolutions.append(curve)
        hyperplanesEvolutions.append(saveHyperplanes)
    
    #display results
    stepsToDisplay = [0,10,-1]
    for model in hyperplanesEvolutions:
        for step in stepsToDisplay:
            visu.visualiseHyperplanes(model[step],'optimisation step %d'%(step),5,distribution)
    plt.figure(); 
    for curve in measureEvolutions:
        plt.plot(curve); 
    plt.title(t); plt.show()
# multipleTest2D(4, distribution, measure, 5)





def higherDimensions(nHyperplanes,nDimensions,distrib,m,nIterations,initNotRandom=False):
    #initialisation
    if initNotRandom:
        hp = init.poolSelect(nHyperplanes,nDimensions,1000,10000,20,distrib,m)
    else:
        hp = init.normal(nHyperplanes,nDimensions)
    
    curve,saveHyperplanes = core.optimisation(hp,5000,20000,nIterations,
                                            [False,False,1],
                                            'test higher dimensions',distribution,measure,
                                            updateMethod="oneVarInterpolation",
                                            precisionCheck=False,
                                            structureCheck=False)
    # store results in a file
    d = 'G' if distrib == 'gaussian' else 'U'
    plt.figure(); plt.plot(curve); plt.title("results_data/"+d+"_"+str(nDimensions)+"D_"+str(nHyperplanes)+"Hp_"+measure); plt.show()
    print("results_data/"+d+"_"+str(nDimensions)+"D_"+str(nHyperplanes)+"Hp_"+measure)
    
    file = open("results_data/"+d+"_"+str(nDimensions)+"D_"+str(nHyperplanes)+"Hp_"+measure+".txt",'a') 
    file.write('\n')
    file.write( str(curve[-1]) )
    file.close()

#while True:
#    for nHyperplanes in range(4,11):
#        nDimensions = 3
#        nIterations = 8
#        distribution = 'uniform'
#        measure = 'entropy'
#        higherDimensions(nHyperplanes,nDimensions,distribution,measure,nIterations)




def mseEstimations(numberOfEstimations,m="mse"):
    ''' estimations of MSE on the same set of points to visualise its variance '''
    e=[]
    hp = init.normal(3,2)
    for k in range(numberOfEstimations):
        e.append(ms.measure(m,hp,1000,10000,'gaussian'))
    plt.figure(); plt.plot(e); plt.title('%d mse estimations')%numberOfEstimations; plt.show()
# mseEstimations(50)

def initPerformance(paramEval,nDimensions,nHyperplanes,distrib,measure,updateMethod,geneticInit=False):
    '''
        Define the performance of an initialisation method. Run optimization
        several times to get values about convergence speed, optimum quality...
        paramEval is the number of times the optimization is to be run.
    '''
    measureEvols = []
    typeInit = "genetic_"
    for k in range(paramEval):
        
        # initialization
        if geneticInit:
            order = 'dissimilarity'
            pCentroids = 10000
            pMeasure   = 30000
            hps , geneticMeasureEvolution = init.genetic(
                    nHyperplanes,nDimensions, pCentroids, pMeasure,distrib,measure,
                    10,5,1,1,order) # configs number, total iterations, crossover points, mutation param
            # initialization method can be changed here
            # Don't forget to change the file names accordingly !
            
            # save genetic algorithm results
            d = 'G' if distrib == 'gaussian' else 'U'
            file = open("Initialisation_performance_data/genetic_evol_data/"
                        +"genetic_"
                        +d+"_"+str(nDimensions)+"D_"+str(nHyperplanes)+"Hp_"+measure
                        +"_10geneticConfigs"+"_5geneticIter"+"_1crossover"
                        +".txt",'a') 
            file.write('\n')
            file.write( str(geneticMeasureEvolution) )
            file.close()
        else:
            typeInit,hps = "randomInit_",[]
            hps = init.doublePoint(nHyperplanes,nDimensions,distrib)
        
        # optimize
        pCentroids = 10000
        pMeasure   = 100000
        measureEvolution,saveHyperplanes = core.optimisation(hps,pCentroids,pMeasure,
                                                             5,distrib=distrib,m=measure,
                                                             updateMethod=updateMethod)
        measureEvols.append(measureEvolution)
        
        # save
        d = 'G' if distrib == 'gaussian' else 'U'
        file = open("Initialisation_performance_data/"
                    +typeInit
                    +d+"_"+str(nDimensions)+"D_"+str(nHyperplanes)+"Hp_"+measure
                    +"_10geneticConfigs"+"_5geneticIter"+"_1crossover"
                    +".txt",'a') 
        file.write('\n')
        file.write( str(measureEvolution) )
        file.close()

#dimensions=2
#for k in range(dimensions,7): # dimensions to 6 hyperplanes
#    initPerformance(1,dimensions,k,distrib='gaussian',
#                    measure='mse',updateMethod='oneVarInterpolation')




#dimensions=2
#dimensions=3
#dimensions=4
dimensions=5
for k in range(dimensions,7):
    initPerformance(1,dimensions,k,distrib='gaussian',measure='mse',updateMethod='oneVarInterpolation')
#    initPerformance(1,dimensions,k,distrib='gaussian',measure='mse',updateMethod='oneVarInterpolation',geneticInit=True)







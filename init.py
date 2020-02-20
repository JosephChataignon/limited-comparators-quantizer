#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy

import utils
import measures as ms

def normal(nHyperplanes,nDimensions):
    """
        Returns a set of hyperplanes with random orientations. nHyperplanes is
        the number of hyperplanes to return, and nDimension the number of
        dimensions of the space.
        The hyperplanes are simply generated by setting their coordinates to
        random values following a normal distribution.
    """
    return np.random.normal(0,10,(nHyperplanes,nDimensions+1))

def doublePoint(nHyperplanes,nDimensions,distrib):
    """
        Returns a set of hyperplanes with random orientations. nHyperplanes is
        the number of hyperplanes to return, and nDimension the number of
        dimensions of the space.
        Here for each hyperplane, nDimensions random points are generated
        following the distribution distrib, and the unique hyperplane passing 
        by all these points is kept.
    """
    hyperplanes = []
    for k in range(nHyperplanes):
        points = np.array([utils.f(nDimensions,distrib) for n in range(nDimensions)])
        hyperplanes.append( utils.hyperplaneFromPoints(points) )
    return np.array(hyperplanes)

def poolSelect(nHyperplanes, nDimensions, pCentroids, pMeasure, poolSize, distrib, m,initType='doublePoint'):
    '''
        Initialize hyperplanes by generating a pool of poolSize random 
        configurations and selecting the one with lowest measure of distortion.
    '''
    for k in range(poolSize):
        if initType == 'normal':
            hps = normal(nHyperplanes,nDimensions) 
        elif initType == 'doublePoint': 
            hps = doublePoint(nHyperplanes,nDimensions,distrib) 
        else:
            print("ERROR! invalid initialization type")
        e = ms.measure(m,hps,pCentroids,pMeasure,distrib)
        if k < 1:
            minDistortion = e
            minConfig = hps
        else:
            if minDistortion >= e:
                minDistortion = e
                minConfig = hps
    return minConfig

def genetic(nHyperplanes, nDimensions, pCentroids, pMeasureInit, distrib, m,
            nConfigs, pGenetic, crossover, mutation, order='dissimilarity', selection='rank', 
            initType='doublePoint', mutationPercentage=50):
    '''
        Generates a partially optimized configuration of hyperplanes, with the 
        goal of having an approximatively equal repartition of the input 
        distribution over the regions.
        Here one hyperplane = one gene. At every iteration, one half of the old
        configs is kept and used to generate the next generation by crossover.
        -nConfigs is the number of configurations to generate and cross
        -pGenetic is the number of iterations
        -order is the type of ordering used to order hyperplanes before crossover
        -crossover is the number of crossing points in the crossover operations
        -mutation is the mutation method and intensity
        -selection is the selection method used to chose the configs that are 
            reproduced
    '''
    print('start initialisation (genetic)')
    configs = []
    measures = []
    geneticMeasureEvolution = []
    # Step 1: generate random configurations
    for k in range(nConfigs):
        if initType == 'normal':
            config = normal(nHyperplanes,nDimensions) 
        elif initType == 'doublePoint': 
            config = doublePoint(nHyperplanes,nDimensions,distrib) 
        else:
            print("ERROR! invalid initialization type")
        configs.append(config)
    print('finished generating random configurations')
    
    for k in range(pGenetic):
        pMeasure = (k+1)*pMeasureInit
        print('genetic: iteration '+str(k+1)+' of '+str(pGenetic))
        measures = [ms.measure(m, config, pCentroids, pMeasure, distrib) for config in configs]
        geneticMeasureEvolution.append( np.min(measures) )
        # Step 2: selecting configs to reproduce
        configs, measures = select(selection, configs, measures, percentageToKeep=80)
        # Step 3: crossing configurations
        newConfigs = cross(nDimensions, distrib, crossover, copy.deepcopy(configs), order, outputSize=nConfigs)
        configs = np.concatenate((configs,newConfigs),axis=0)
        
        # Step 4: mutation
        if mutationPercentage == 100:
            configs = mutateAll(mutation,configs)
        else:
            measures = [ms.measure(m, config, pCentroids, pMeasure, distrib) for config in configs]
            configs = mutate(mutation, configs, measures, mutationPercentage)
    
    # Step 5: return the best config
    measures = [ms.measure(m, config, pCentroids, pMeasure, distrib) for config in configs]
    
    print('final: ',np.min(measures))
    
    print('end initialisation')
    return configs[ np.argmin(measures) ], geneticMeasureEvolution
#print(genetic(3, 2, 1000, 10000, 'gaussian', 'mse',10, 5, 1, 1)) #test



## Genetic algorithm subfunctions

def select(selection, configs, measures, percentageToKeep=50):
    '''
        Returns the selected configurations that are kept for the next 
        generation.
        percentageToKeep is the persentage of the total of configuration, 
        representing the configurations that will be kept.
    '''
    n = int(len(configs)*percentageToKeep/100)
    if selection == 'rank':
        # sort by distortion measure and keep the lowest
        configs = [x for _,x in sorted(zip(measures,configs))]
        measures = sorted(measures)
        return configs[:n+1], measures[:n+1]
    elif selection == 'random':
        return configs[:n+1], measures[:n+1]
    else:
        print('ERROR: unknown selection method')

def cross(nDimensions, distrib, crossover, configs, order, outputSize='default'):
    '''
        Crosses the configs 'configs', repeating the operation 'outputSize' times,
        with 'crossover' crossing points each time.
        Hyperplanes can be ordered before the crossing.
    '''
    newGen = [] # next generation
    if outputSize == 'default':
        outputSize = len(configs)
    if order == 'distanceToDistribCenter':
        distribCenter = utils.distribCenter(nDimensions, distrib)
        for k in range(len(configs)):
            config = configs[k]
            ranks = [ utils.distancePoint2Hp(distribCenter,hp) for hp in config ]
            #order hyperplanes according to ranks
            ordConfig = [hp for _,hp in sorted(zip(ranks,config))]
            configs[k] = ordConfig        
    
    for k in range(outputSize):
        # select 2 configs to cross
        i,j = np.random.randint(len(configs)),np.random.randint(len(configs))
        if order == 'distanceToDistribCenter' or order == 'noOrder':
            crosspoints = np.random.randint(len(configs[0]), size=crossover)# chose crossing points
            newConfig = []
            useI = True # whether to include i or j genes
            for l in range(len(configs[0])):
                if useI:
                    newConfig.append(configs[i][l])
                else:
                    newConfig.append(configs[j][l])
                if l in crosspoints:
                    useI = not useI
        elif order == 'dissimilarity':
            dissimilarities, hpPairs = [], [] # list to store dissimilarity values and associated hyperplane pairs
            for k in range(1,len(configs[i])):
                for l in range(k):
                    dissimilarities.append(utils.dissimilarityHps(configs[j][l], configs[i][k], distrib))
                    hpPairs.append([k,l])
            hpPairs = [hpPair for _,hpPair in sorted(zip(dissimilarities,hpPairs))]
            newConfig = configs[i]
            for pair in hpPairs[:int(len(hpPairs)/2)]: #swap the most similar half of hyperplane pairs
                newConfig[pair[0]] = configs[j][pair[1]]
        newGen.append(newConfig)
    return newGen

def mutateAll(mutation, configs):
    '''Applies a random mutation to all configs'''
    configs *= np.random.normal(1.,0.2, np.shape(configs))
    return configs

def mutate(mutation, configs, measures, percentageToMutate=50):
    '''
        Applies a random mutation to a ceratin percentage of configs. For now,
        only multiplies every matrix coefficient with a random normal value.
    '''
    #newConfigs = []#muter uniquement les moins performants
    configs = [x for _,x in sorted(zip(measures,configs))] # Sort configs according to measure
    nHp,nDim1,nC = len(configs[0]), len(configs[0][0]), int(len(configs)/100*percentageToMutate)
    
    configs *= np.concatenate((
        np.ones(( nC,nHp,nDim1 )) ,
        np.random.normal(1.,0.2, (len(configs)-nC,nHp,nDim1) )
        ),axis=0)
    
    return configs

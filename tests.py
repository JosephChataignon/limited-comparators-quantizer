# -*- coding: utf-8 -*-

# With n dimensions, each hyperplane P is modelised by an array of length n+1,
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

import core
import visualization as visu
import measures as ms


distribution = 'gaussian'
measure = 'entropy'




def standardTest2D(distrib,m,nHyperplanes,initNotRandom=False):
    '''
        With distribution distrib and measure m, run optimisation for 
        nHyperplanes hyperplanes and 2 dimensions.
    '''
    #initialisation
    if initNotRandom:
        hp = initNotRandom(nHyperplanes,2,800,2500,20,distribution,m)
    else:
        hp = core.init(nHyperplanes,2)
    visu.visualiseHyperplanes(hp,'initial configuration',5,distribution)

    #title
    d = 'G' if distrib == 'gaussian' else 'U'
    t = 'test %d%s'%(nHyperplanes,d)
    print(t,'\n')

    curve,saveHyperplanes = core.optimisation(hp,1000,10000,20,
        [True,True,5],
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
            hp = initNotRandom(nHyperplanes,2,800,2500,20,distribution,m)
        else:
            hp = core.init(nHyperplanes,2)
        #visu.visualiseHyperplanes(hp,'initial configuration',5,distribution)
    
        #title
        d = 'G' if distrib == 'gaussian' else 'U'
        t = 'test %d%s, model %d'%(nHyperplanes,d,k)
        print(t,'\n')
        
        #optimisation
        curve,saveHyperplanes = core.optimisation(hp,100,300,3,
            [False,False,5],
            t,distribution,measure,
            updateMethod="oneVarInterpolation",
            precisionCheck=False,
            structureCheck=False)
        measureEvolutions.append(curve)
        hyperplanesEvolutions.append(saveHyperplanes)
    
    #display results
    stepsToDisplay = [0,2]
    for model in hyperplanesEvolutions:
        for step in stepsToDisplay:
            visu.visualiseHyperplanes(model[step],'optimisation step %d'%(step),5,distribution)
    plt.figure(); 
    for curve in measureEvolutions:
        plt.plot(curve); 
    plt.title(t); plt.show()
    
multipleTest2D(3, distribution, measure, 3)
    





def mseEstimations(numberOfEstimations,m="mse"):
    e=[]
    hp = core.init(3,2)
    for k in range(numberOfEstimations):
        e.append(ms.measure(m,hp,1000,10000,'gaussian'))
    plt.figure(); plt.plot(e); plt.title('%d mse estimations')%numberOfEstimations; plt.show()
# mseEstimations(50)








# possible improvements:
# change interpolation so as to apply it to each parameter at every iteration - done
# change format of hyperplanes to a 2-coordinates point and and an angle (at least when applying the random changes)
# change varbyvar update so to chose one hyperplane randomly instead of testing them all
# particle filters

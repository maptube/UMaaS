import timeit
import os.path
import numpy as np
from math import exp, fabs
from sys import float_info

from globals import *
from utils import loadMatrix, resizeMatrix

from models.SingleOrigin import SingleOrigin

"""
Benchmarks for the Single Origin Constrained model (models/SingleOrigin.py)
All code here is lifted from the original model code and changed to be
self-contained (no setup) so that timings of various optimisations are easy.
Code here is designed to be a test of timings, NOT necessarily a test of
return values, although real data has been used wherever possible i.e. instead
of an NxN matrix containing random values, I try to load in a real matrix
instead.
"""

#modelRunsDir = '../model-runs'
#TObsFilename = 'TObs.bin' #1 mode
#CijRoadMinFilename = 'Cij_road_min.bin'

#load and init
Tij=loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
cij=loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))
#end load and init

###############################################################################

"""
calculateCBar_slow
Mean trips calculation, straight conversion from original C# code, no python optimisation
@returns float
"""
def benchmark_calculateCBar_slow():
    #main code
    (M, N) = np.shape(Tij)
    CNumerator = 0.0
    CDenominator = 0.0
    for i in range(0,N):
        for j in range(0,N):
            CNumerator += Tij[i, j] * cij[i, j]
            CDenominator += Tij[i, j]
    CBar = CNumerator / CDenominator
    print("CBar=",CBar)

    return CBar

###############################################################################

"""
calculateCBar_fast
Mean trips calculation, python optimised version of "_slow"
@returns float (NOTE: the return value MUST be identical to the _slow version, to prove they're functionally identical)
"""
def benchmark_calculateCBar_fast():
    #load and init
    Tij=loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    cij=loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))
    #end load and init

    #main code
    CNumerator2 = np.sum(Tij*cij)
    CDenominator2 = np.sum(Tij)
    CBar2=CNumerator2/CDenominator2
    print("CBar2=",CBar2)

    return CBar2

###############################################################################

"""
This is a benchmark of the simple Python code for SingleOrigin using different matrix sizes.
It is a test for how long a single execution of the main loop takes. Timings are printed
to the console based on 1000 runs of the model code i.e. the timing you see in seconds
must be divided by 1000.
NOTE: this could take a VERY long time to run if you pass in a high number for Nfinish 
"""
def benchmarkSingleOriginMatrixSizes(Nstart,Nfinish,Nstep):
    print("benchmark_SingleDest running matrix Nstart=",Nstart," Nfinish=",Nfinish, " Nstep=",Nstep)

    #load testing matrices
    TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))

    for N in range(Nstart,Nfinish,Nstep):
        #print("TPred runModel N=",N)
        #set up the model
        testModel = SingleOrigin()
        (TPred, secs)=testModel.benchmarkRun(1000,resizeMatrix(TObs1,N),resizeMatrix(Cij1,N),1.0)
        #NOTE: timing printed to console based on 1000 iterations of the main loop in the above code
        #Should not contain any setup timings - only the actual algorithm run time.
        print(N,",1000,",secs) #all console logging from here - makes it nice and easy to import into excel

###############################################################################


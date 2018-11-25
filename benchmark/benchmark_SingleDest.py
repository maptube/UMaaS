import timeit
import os.path
import numpy as np
from math import exp, fabs
from sys import float_info

from globals import *
from utils import loadMatrix

"""
Benchmarks for the Single Destination Constrained model (models/SingleDest.py)
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


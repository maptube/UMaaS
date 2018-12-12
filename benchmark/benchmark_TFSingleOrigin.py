import timeit
import os.path
import numpy as np
from math import exp, fabs
from sys import float_info

from globals import *
from utils import loadMatrix, resizeMatrix

from models.TFSingleDest import TFSingleDest

"""
This is a benchmark of the TensorFlow code for TFSingleDest using different matrix sizes.
It is a test for how long a single execution of the main loop takes. Timings are printed
to the console based on 1000 runs of the model code i.e. the timing you see in seconds
must be divided by 1000.
NOTE: this could take a VERY long time to run if you pass in a high number for Nfinish 
"""
def benchmarkTFSingleDestMatrixSizes(Nstart,Nfinish):
    print("benchmark_TFSingleDest running matrix Nstart=",Nstart," Nfinish=",Nfinish)

    #TensorFlow tests - load testing matrices
    TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))

    for N in range(Nstart,Nfinish,100):
        print("TFTPred runModel N=",N)
        #set up the model
        testTFModel = TFSingleDest(N)
        TFTPred=testTFModel.runModel(resizeMatrix(TObs1,N),resizeMatrix(Cij1,N),1.0)
        #NOTE: timing printed to console based on 1000 iterations of the main loop in the above code
        #Should not contain any setup timings - only the actual algorithm run time.

###############################################################################



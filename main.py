#!/usr/bin/env python3
"""
 main.py
"""

import os.path
import numpy as np
from globals import *
from utils import loadZoneLookup, loadMatrix, saveMatrix
from models.SingleOrigin import SingleOrigin
from models.TFSingleOrigin import TFSingleOrigin
from benchmark.run_benchmarks import runBenchmarks
from benchmark.benchmark_SingleOrigin import benchmarkSingleOriginMatrixSizes
from benchmark.benchmark_TFSingleOrigin import benchmarkTFSingleOriginMatrixSizes
from unittests.test_TFSingleOrigin import testTFSingleOrigin
#from unittests.test_Movidius import testMovidius, testBuildMovidiusGraph, testRunMovidiusGraph
from unittests.test_KerasGravityANN import testKerasGravityANN


###############################################################################

def main():
    print("python main function")

    #EQUIVALENCE TESTS
    #testTFSingleOrigin() #this is an equivalence test of SingleDest vs TFSingleDest i.e. does the TF code produce equivalent result?
    #======MOVIDIUS======
    #testMovidius()
    #testBuildMovidiusGraph()
    #testRunMovidiusGraph()
    #======Keras ANN======
    #testKerasGravityANN('',500,[64],200000,10000) #model 1 500x500 3-64-1
    #testKerasGravityANN('',500,[32],200000,10000) #model 2 500x500 3-32-1
    #testKerasGravityANN('',500,[16],200000,10000) #model 3 500x500 3-16-1
    #testKerasGravityANN('',500,[8],200000,10000) #model 4 500x500 3-8-1
    #testKerasGravityANN('',500,[4],200000,10000) #model 5 500x500 3-4-1
    #testKerasGravityANN('',500,[4,4],200000,10000) #model 6 500x500 3-4-4-1 10,000 epoch on varying batch size
    #testKerasGravityANN('',500,[4,4],31250,10000) #model 6 10,000 to 40,000 epoch on batch 31250 (optimum)
    #Now some tests on the real matrix
    #testKerasGravityANN('KerasGravityANN_20190102_115120_500_3441_10000.h5',7201,[4,4],57608,100) - no good!
    testKerasGravityANN('',7201,[4,4],115216,100)


    #BENCHMARK TESTS
    #size test doing a sweep of matrix size from N=100 to N=16000
    #======CPU CPU CPU======
    #benchmarkSingleOriginMatrixSizes(100,2000,100) #NOTE: this takes 15 minutes to run
    #benchmarkSingleOriginMatrixSizes(7201,7202,1) #OK, hack a 7201 run for comparison with the real QUANT
    #benchmarkSingleOriginMatrixSizes(2000,13000,500) #NOTE: this takes a VERY long time to run (a day?)
    ##benchmarkSingleOriginMatrixSizes(7500,13000,500) #TEST
    #======GPU GPU GPU======
    #benchmarkTFSingleOriginMatrixSizes(100,2000,100)
    #benchmarkTFSingleOriginMatrixSizes(7201,7202,1) #OK, hack a 7201 run for comparison with the real QUANT
    #benchmarkTFSingleOriginMatrixSizes(2000,16000,500) #NOTE: this takes a VERY long time to run (days?)


    #write out TF compute graph for the paper
    #TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    #Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))
    #TFModel = TFSingleOrigin(7201)
    #TFModel.debugWriteModelGraph(TObs1,Cij1,1.0)
    #tensorboard --logdir log/TFSingleOrigin


    #print ("test run model")
    #how on earth did I do this?
    #calibrate?
    #load TObs x 3, cij x 3 and calibrate betas
    #MODES: 0=road, 1=bus, 2=rail
    #TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    #TObs2 = loadMatrix(os.path.join(modelRunsDir,TObs32Filename))
    #TObs3 = loadMatrix(os.path.join(modelRunsDir,TObs33Filename))
    #TObs = [TObs1, TObs2, TObs3]
    #Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))
    #Cij2 = loadMatrix(os.path.join(modelRunsDir,CijBusMinFilename))
    #Cij3 = loadMatrix(os.path.join(modelRunsDir,CijRailMinFilename))
    #Cij = [Cij1,Cij2,Cij3]
    #print("loaded matrices")
    #set up model information to calibrate
    #model = SingleOrigin()
    #model.TObs = TObs
    #model.Cij=Cij
    #model.isUsingConstraints=False
    #print("run model")
    #model.run()

###############################################################################


if __name__ == '__main__':
    main()
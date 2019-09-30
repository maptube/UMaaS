#!/usr/bin/env python3
"""
 main.py
"""

import os.path
import numpy as np
from globals import *
from utils import loadMatrix, saveMatrix, loadQUANTCSV
from zonecodes import ZoneCodes #you need loadZoneLookup function from this
from models.SingleOrigin import SingleOrigin
from models.TFSingleOrigin import TFSingleOrigin
from benchmark.run_benchmarks import runBenchmarks
from benchmark.benchmark_SingleOrigin import benchmarkSingleOriginMatrixSizes
from benchmark.benchmark_TFSingleOrigin import benchmarkTFSingleOriginMatrixSizes
from unittests.test_TFSingleOrigin import testTFSingleOrigin
#from unittests.test_Movidius import testMovidius, testBuildMovidiusGraph, testRunMovidiusGraph
from unittests.test_KerasGravityANN import testKerasGravityANN, testKerasGravityANNInference, testKerasCBarError
from unittests.test_drawgraphs import plotLogCijLogTij, plotOiDj, plotCijHistogram, plotTijHistogram, plotTijTail

#from graphserver.graphtest import graphtest1


###############################################################################

def main():
    print("python main function")

    #Build Matrix Data - this should probably be in databuilder
    #Cij, Tij = loadQUANTCSV("data/fromQUANT/trainingdata_road.csv",7201)
    #saveMatrix(Cij,"data/fromQUANT/Py_Cij_road.bin")
    #saveMatrix(Tij,"data/fromQUANT/Py_TObs_road.bin")
    #Cij, Tij = loadQUANTCSV("data/fromQUANT/trainingdata_bus.csv",7201)
    #saveMatrix(Cij,"data/fromQUANT/Py_Cij_bus.bin")
    #saveMatrix(Tij,"data/fromQUANT/Py_TObs_bus.bin")
    #Cij, Tij = loadQUANTCSV("data/fromQUANT/trainingdata_gbrail.csv",7201)
    #saveMatrix(Cij,"data/fromQUANT/Py_Cij_gbrail.bin")
    #saveMatrix(Tij,"data/fromQUANT/Py_TObs_gbrail.bin")
    ###

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
    #testKerasGravityANN('KerasGravityANN_20190107_101348.h5',7201,[4,4],230432,100) #was 28804
    #testKerasGravityANN('KerasGravityANN_20190111_124934.h5',7201,[4,4],230432,200)
    #testKerasGravityANN('',7201,[8],230432,100) #8 seems to train more reliably
    #testKerasGravityANN('KerasGravityANN_20190112_181256.h5',7201,[32],230432,100)
    
    #======Keras ANN Inference Times======
    #ANN Inference testing - OK, this should really be benchmarking as it's a speed test of inference
    #testKerasGravityANNInference(7201,[64]) #NOTE this is a long one
    #testKerasGravityANNInference(7201,[32])
    #testKerasGravityANNInference(7201,[16])
    #testKerasGravityANNInference(7201,[8])
    #testKerasGravityANNInference(7201,[4])
    #testKerasGravityANNInference(7201,[4,4])

    #======New Keras Tests======
    #this is for the 4340 data Tij>=10 and Cij>=30 mins
    #meanOi= 7741.307142857143 meanDj= 2406.9841013824885 meanCij= 41.98041474654378 meanTij= 16.243087557603687
    #sdOi= 7106.816265087332 sdDj= 728.4594067969223 sdCij 24.522797327549736 sdTij 9.358829127240579
    #this is in results/new-256-256-256 trained using mae default adam, relu, relu, linear
    #testKerasGravityANN('results/new-256-256-256/KerasGravityANN_20190923_115800.h5',7201,[256,256,256],104647,10000) #255782, 104647 in sample
    #further testing testKerasGravityANN('KerasGravityANN_20190926_202411.h5',7201,[256,256,256],104647,10000)
    #this is how it was trained: model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae','accuracy'])
    #testKerasCBarError('results/new-256-256-256/KerasGravityANN_20190923_115800.h5')
    #
    #testKerasGravityANN('KerasGravityANN_20190927_152942.h5',7201,[16],10240,10000) #16adagrad on ALL 52m data
    #meanOi= 4010.628981245261 meanDj= 2241.033513047048 meanCij= 12.390618035274228 meanTij= 26.760203675543433
    #sdOi= 4543.346693511267 sdDj= 719.5767651711014 sdCij= 11.059383993934794 sdTij= 50.89338364137762
    #testKerasGravityANN('',7201,[8],256,1000) #1987928


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


    #======These are numbers tests on the original data======
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

    #======PLOTTING GRAPHS FOR TESTING DATA======
    #Python has botched these ones in the loading - they're defaulted to int metrices when the Cij should be float - load direct from QUANT!
    #TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    #Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))
    TObs1 = loadMatrix("data/fromQUANT/Py_TObs_road.bin") #alternate matrices built directly from the QUANT training data
    Cij1 = loadMatrix("data/fromQUANT/Py_Cij_road.bin")
    #plotLogCijLogTij(TObs1,Cij1)
    #plotOiDj(TObs1)
    #plotCijHistogram(Cij1)
    #plotTijHistogram(TObs1)
    plotTijTail(TObs1)

    #======NETWORK TESTS======
    #ZoneCodes.deriveAreakeyToZoneCodeFromShapefile('data/geometry/EnglandWalesScotland_MSOA.shp') #create the zone codes csv file from the shapefile
    #graphtest1()


###############################################################################


if __name__ == '__main__':
    main()
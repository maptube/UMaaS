#!/usr/bin/env python3
"""
 main.py
"""

import os.path
import numpy as np
from globals import *
from utils import loadZoneLookup, loadMatrix, saveMatrix
from models.SingleDest import SingleDest
from models.TFSingleDest import TFSingleDest
from benchmark.run_benchmarks import runBenchmarks
from unittests.test_TFSingleDest import testTFSingleDest
#from unittests.test_Movidius import testMovidius, testBuildMovidiusGraph, testRunMovidiusGraph


###############################################################################

def main():
    print("python main function")

    #runBenchmarks()
    testTFSingleDest()
    #testMovidius()
    #testBuildMovidiusGraph()
    #testRunMovidiusGraph()


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
    #model = SingleDest()
    #model.TObs = TObs
    #model.Cij=Cij
    #model.isUsingConstraints=False
    #print("run model")
    #model.run()

###############################################################################


if __name__ == '__main__':
    main()
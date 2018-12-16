#Keras Gravity ANN Tests
#Unit tests test functionality i.e. one model against another and correlations and errors between results

import os.path
import math
import numpy as np

from globals import *
from utils import loadMatrix
from models.KerasGravityANN import KerasGravityANN

###############################################################################

def testKerasGravityANN():
    #load in data - we have 52 million points!
    #use mode 1 = road
    TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))
    (M, N) = np.shape(TObs1)
    KGANN = KerasGravityANN()
    Oi = KGANN.calculateOi(TObs1)
    Dj = KGANN.calculateDj(TObs1)
    #now we need to make an input set which is [Oi,Dj,Cij] with a target of Tij
    print("Building training set - this might take a while...")
    inputs = np.empty([N*N, 3], dtype=float)
    targets = np.empty([N*N,1], dtype=float)
    nextpct = 0
    for i in range(0,N):
        pct = i/N*100
        if pct>=nextpct:
            print(pct," percent complete")
            nextpct+=10
        for j in range(0,N):
            inputs[i*j,0]=Oi[i]
            inputs[i*j,1]=Dj[j]
            inputs[i*j,2]=Cij1[i,j]
            targets[i*j]=TObs1[i,j]
        #end for j
    #end for i

    #raw inputs must be normalised for input to the ANN [0..1]
    KGANN.normaliseInputsLinear(inputs,targets)
    #input is [ [Oi, Dj, Cij], ..., ... ]
    #targets are [ TObs, ..., ... ] to match inputs
    KGANN.trainModel(inputs,targets,2) #was 1000 ~ 20 hours!

    #todo: get the beta back out by equivalence testing and plot geographically

#Keras Gravity ANN Tests
#Unit tests test functionality i.e. one model against another and correlations and errors between results

import os.path
import math
import numpy as np

from globals import *
from utils import loadMatrix
from models.KerasGravityANN import KerasGravityANN

###############################################################################

"""
Compare two matrices and return mean square error
"""
def meanSquareError(TObs,TPred):
    #todo: probably should fail if shape TObs not the same as TPred - it's just going to crash though
    #(M, N) = np.shape(TObs)
    #e = 0
    #for i in range(0,N):
    #    for j in range(0,N):
    #        diff = TObs[i,j]-TPred[i,j]
    #        e+=diff*diff
    #return e/(N*N)
    return np.square(np.subtract(TObs, TPred)).mean()

###############################################################################

"""
Return a count of all the non-zero elements in TObs
"""
def countNonZero(TObs):
    (M, N) = np.shape(TObs)
    count=0
    for i in range(0,N):
        for j in range(0,N):
            if TObs[i,j]>=1:
                count+=1
    return count

###############################################################################

def testKerasGravityANN():
    #load in data - we have 52 million points!
    #use mode 1 = road
    TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))
    (M, N) = np.shape(TObs1)
    KGANN = KerasGravityANN()
    #KGANN.loadModel('KerasGravityANN_20181220_203918.h5')
    Oi = KGANN.calculateOi(TObs1)
    Dj = KGANN.calculateDj(TObs1)
    #now we need to make an input set which is [Oi,Dj,Cij] with a target of Tij
    print("Building training set - this might take a while...")
    count = countNonZero(TObs1) #make count N*N if you want everything
    #count=1000 #HACK!!!!
    print('Found ',count,' non-zero entries in TObs')
    inputs = np.empty([count, 3], dtype=float)
    targets = np.empty([count,1], dtype=float)
    nextpct = 0
    dataidx=0
    for i in range(0,N):
        pct = i/N*100
        if pct>=nextpct:
            print(pct," percent complete")
            nextpct+=10
        for j in range(0,N):
            if TObs1[i,j]>=1:
                inputs[dataidx,0]=Oi[i]
                inputs[dataidx,1]=Dj[j]
                inputs[dataidx,2]=Cij1[i,j]
                targets[dataidx,0]=TObs1[i,j]
                dataidx+=1
                if dataidx>=count: break #this was really to allow me to set count=1000 for debugging (also break below)
        #end for j
        if dataidx>=count: break
    #end for i
    for i in range(0,10):
        print('[',inputs[i,0],',',inputs[i,1],',',inputs[i,2],'] ---> ',targets[i,0])

    #raw inputs must be normalised for input to the ANN [0..1]
    KGANN.normaliseInputsLinear(inputs,targets)
    #input is [ [Oi, Dj, Cij], ..., ... ]
    #targets are [ TObs, ..., ... ] to match inputs
    KGANN.trainModel(inputs,targets,100) #was 1000 ~ 20 hours!
    #KGANN.loadModel('KerasGravityANN_20181218_102849.h5')

    #todo: get the beta back out by equivalence testing and plot geographically
    #TPred = KGANN.predictMatrix(TObs1,Cij1)
    for i in range(0,10):
        TPredij = KGANN.predictSingle(TObs1,Cij1,i,0)
        print('TPred [',i,',0]=',TPredij,'TObs[',i,',0]=',TObs1[i,0]) #OK, not a great test, but let's see it work
    #print('mean square error = ',meanSquareError(TObs1,TPred))

    for i in range(0,10):
        in2 = np.empty([1, 3], dtype=float)
        in2[0,0]=inputs[i,0]
        in2[0,1]=inputs[i,1]
        in2[0,2]=inputs[i,2]
        TPredij = KGANN.predict(in2)
        print('TPred2',i,'=',TPredij/KGANN.scaleTij,'Target=',targets[i,0]/KGANN.scaleTij)

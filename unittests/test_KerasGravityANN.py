#Keras Gravity ANN Tests
#Unit tests test functionality i.e. one model against another and correlations and errors between results

import os.path
import math
import time
import numpy as np

from globals import *
from utils import loadMatrix, resizeMatrix
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

"""
Testing procedure for gravity ANN. Allows for changing network size, type, matrix size, training batch and epochs
@param matrixN Allows resizing of the matrix for faster testing
@param numHiddens is a list of hiddens in each layer
@param batchSize
@param numEpochs
"""
def testKerasGravityANN(matrixN,numHiddens,batchSize,numEpochs):
    #load in data - we have 52 million points!
    #use mode 1 = road
    TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))
    (M, N) = np.shape(TObs1)
    #if the real shape of the matrix matches matrixN (i.e. 7201), then don't touch it, otherwise resize TObs1 and Cij1
    if matrixN!=N:
        TObs1=resizeMatrix(TObs1,matrixN)
        Cij1=resizeMatrix(Cij1,matrixN)
    #end matrix resize
    (M, N) = np.shape(TObs1)
    KGANN = KerasGravityANN(numHiddens)
    #KGANN.loadModel('KerasGravityANN_20181223_215731.h5')
    Oi = KGANN.calculateOi(TObs1)
    Dj = KGANN.calculateDj(TObs1)
    KGANN.targetOi = Oi #these three, targetOi/Dj/Cij are used for evaluating DjPred every epoch
    KGANN.targetDj = Dj
    KGANN.targetCij = Cij1
    #now we need to make an input set which is [Oi,Dj,Cij] with a target of Tij
    print("Building training set - this might take a while...")
    #count = countNonZero(TObs1) #make count N*N if you want everything
    #count=100 #HACK!!!!
    count=N*N #do this to include EVERY sample - including zero ones
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
            #if TObs1[i,j]>=1: #HACK!
            inputs[dataidx,0]=Oi[i] #max(Oi[i],0.001) #need to avoid log(0)
            inputs[dataidx,1]=Dj[i] #max(Dj[j],0.001)
            inputs[dataidx,2]=Cij1[i,j] #max(Cij1[i,j],0.001)
            targets[dataidx,0]=TObs1[i,j] #max(TObs1[i,j],0.001)
            dataidx+=1
            if dataidx>=count: break #this was really to allow me to set count=1000 for debugging (also break below)
        #end for j
        if dataidx>=count: break
    #end for i
    for i in range(0,10):
        print('[',inputs[i,0],',',inputs[i,1],',',inputs[i,2],'] ---> ',targets[i,0])

    #raw inputs must be normalised for input to the ANN [0..1]
    KGANN.normaliseInputsLinear(inputs,targets)
    #KGANN.normaliseInputsLog(inputs,targets)
    ###Test input normalisation
    for i in range(0,10):
        print('NORMALISED [',inputs[i,0],',',inputs[i,1],',',inputs[i,2],'] ---> ',targets[i,0])
    ###
    #input is [ [Oi, Dj, Cij], ..., ... ]
    #targets are [ TObs, ..., ... ] to match inputs
    starttime = time.time()
    KGANN.trainModel(inputs,targets,batchSize,numEpochs) #was 1000 ~ 20 hours!
    finishtime = time.time()
    print('Training time ',finishtime-starttime,' seconds')
    #KGANN.loadModel('KerasGravityANN_20181218_102849.h5')

    #todo: get the beta back out by equivalence testing and plot geographically
    #TPred = KGANN.predictMatrix(TObs1,Cij1)
    #for i in range(0,10):
    #    TPredij = KGANN.predictSingle(TObs1,Cij1,i,0)
    #    print('TPred [',i,',0]=',TPredij,'TObs[',i,',0]=',TObs1[i,0]) #OK, not a great test, but let's see it work
    #print('mean square error = ',meanSquareError(TObs1,TPred))

    for i in range(0,10):
        in2 = np.empty([1, 3], dtype=float)
        in2[0,0]=inputs[i,0]
        in2[0,1]=inputs[i,1]
        in2[0,2]=inputs[i,2]
        TPredij = KGANN.predict(in2)
        print('TPred2',i,'=',KGANN.unconvertTij(TPredij),'Target=',KGANN.unconvertTij(targets[i,0]))

    #this is computationally intensive - compute the mean trips error to see whether the training is aacceptable
    KGANN.calculateCBarError(TObs1,Cij1)
    
    #time inference time
    starttime = time.time()
    KGANN.predict(inputs)
    finishtime = time.time()
    print('Inference time: ',finishtime-starttime, ' seconds')


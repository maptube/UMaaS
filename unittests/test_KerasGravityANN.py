#Keras Gravity ANN Tests
#Unit tests test functionality i.e. one model against another and correlations and errors between results

import os.path
import math
import time
import numpy as np
import random
from tensorflow.keras import optimizers

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

def filterValidData(i,j,T,C):
    #returns true if data is valid for i,j,T,C
    #note i,j are zone indices, T is the nimber of trips between i and j and C is the cost
    #return i!=j and T>=10 and C>=30 #this is what the initial training used
    #return i!=j and T>=10 and C>=25
    return T>=5 and C>30

"""
Return a count of all the non-zero elements in TObs
"""
def countNonZero(TObs,Cij):
    (M, N) = np.shape(TObs)
    count=0
    for i in range(0,N):
        for j in range(0,N):
            if not filterValidData(i,j,TObs[i,j],Cij[i,j]):
                continue #HACK!
            count+=1
    return count

###############################################################################

"""
Write the training set out to disk
"""
def writeTrainingSet(filename,inputs,targets):
    with open(filename,"w") as f:
        f.write('Oi,Dj,Cij,Tij\n')
        for i in range(0,len(inputs)):
            f.write(str(inputs[i,0]) + ',' + str(inputs[i,1]) + ',' + str(inputs[i,2]) + ',' + str(targets[i,0]) + '\n')

###############################################################################

"""
Testing procedure for gravity ANN. Allows for changing network size, type, matrix size, training batch and epochs
@param modelFilename null, or filename of an h5 model and weights to load for continuation training. If null, then train a new model from scratch.
        NOTE: the file loaded doesn't necessarily match the numHiddens, but will override numHiddens with the model structure that was saved. numHiddens
        is redundant if modelFilename is used.
@param matrixN Allows resizing of the matrix for faster testing
@param numHiddens is a list of hiddens in each layer
@param batchSize
@param numEpochs
"""
def testKerasGravityANN(modelFilename,matrixN,numHiddens,batchSize,numEpochs):
    #load in data - we have 52 million points!
    #use mode 1 = road
    #TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    #Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))
    TObs1 = loadMatrix("data/fromQUANT/Py_TObs_road.bin") #alternate matrices built directly from the QUANT training data
    Cij1 = loadMatrix("data/fromQUANT/Py_Cij_road.bin")
    (M, N) = np.shape(TObs1)
    #if the real shape of the matrix matches matrixN (i.e. 7201), then don't touch it, otherwise resize TObs1 and Cij1
    if matrixN!=N:
        TObs1=resizeMatrix(TObs1,matrixN)
        Cij1=resizeMatrix(Cij1,matrixN)
    #end matrix resize
    (M, N) = np.shape(TObs1)
    KGANN = KerasGravityANN(numHiddens)
    if modelFilename!='': #NOTE: this OVERWRITES the model that was just created, so numHiddens is not used - could be a different architecture
        KGANN.loadModel(modelFilename)
    #override loaded model
    #sgd = optimizers.SGD(lr=0.1, momentum=0, decay=0)
    #KGANN.model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse','mae'])
    #KGANN.model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['mse','mae'])
    #end
    Oi = KGANN.calculateOi(TObs1)
    Dj = KGANN.calculateDj(TObs1)
    #for i in range(0,1000):
    #    print("testGravityANN: Oi,Dj: ",Oi[i],Dj[i])
    KGANN.targetOi = Oi #these three, targetOi/Dj/Cij are used for evaluating DjPred every epoch
    KGANN.targetDj = Dj
    KGANN.targetCij = Cij1
    #now we need to make an input set which is [Oi,Dj,Cij] with a target of Tij
    print("Building training set - this might take a while...")
    count = countNonZero(TObs1,Cij1) #make count N*N if you want everything
    #count=100 #HACK!!!!
    #count=N*N #do this to include EVERY sample - including zero ones
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
            if not filterValidData(i,j,TObs1[i,j],Cij1[i,j]):
                continue #HACK!
            inputs[dataidx,0]=Oi[i] #max(Oi[i],0.001) #need to avoid log(0)
            inputs[dataidx,1]=Dj[j] #max(Dj[j],0.001)
            inputs[dataidx,2]=Cij1[i,j] #max(Cij1[i,j],0.001)
            targets[dataidx,0]=TObs1[i,j] #max(TObs1[i,j],0.001)
            dataidx+=1
            if dataidx>=count: break #this was really to allow me to set count=1000 for debugging (also break below)
        #end for j
        if dataidx>=count: break
    #end for i
    #thin out the training data if required
    #count=20000
    #inputs, targets = KGANN.reduceInputData(count,inputs,targets)
    #inputs, targets = KGANN.clusterInputData(inputs,targets)
    #end thinning out
    for i in range(0,10):
        print('[',inputs[i,0],',',inputs[i,1],',',inputs[i,2],'] ---> ',targets[i,0])
    writeTrainingSet("training_data.csv",inputs,targets) #write training data out to disk

    #raw inputs must be normalised for input to the ANN [0..1] - three forms of normalisation here
    #KGANN.normaliseInputsLinear(inputs,targets)
    #KGANN.normaliseInputsLog(inputs,targets)
    KGANN.normaliseInputsMeanSD(inputs,targets)
    KGANN.randomiseInputOrder(inputs,targets) #randomise the ordering of the inputs and targets
    writeTrainingSet("training_data_norm.csv",inputs,targets) #write out normalised training data to disk for comparison
    ###Test input normalisation
    #for i in range(0,10):
    #    print('NORMALISED [',inputs[i,0],',',inputs[i,1],',',inputs[i,2],'] ---> ',targets[i,0])
    ###
    #input is [ [Oi, Dj, Cij], ..., ... ]
    #targets are [ TObs, ..., ... ] to match inputs
    #KGANN.setLearningRate(0.02)  #HACK! Override learning rate
    starttime = time.time()
    KGANN.trainModel(inputs,targets,batchSize,numEpochs) #was 1000 ~ 20 hours!
    finishtime = time.time()
    print('Training time ',finishtime-starttime,' seconds')
    #KGANN.loadModel('KerasGravityANN_20181218_102849.h5')
    score = KGANN.model.evaluate(inputs,targets,verbose=0)
    print("%s: %.2f%%" % (KGANN.model.metrics_names[1], score[1]*100))

    #todo: get the beta back out by equivalence testing and plot geographically
    #TPred = KGANN.predictMatrix(TObs1,Cij1)
    #for i in range(0,10):
    #    TPredij = KGANN.predictSingle(TObs1,Cij1,i,0)
    #    print('TPred [',i,',0]=',TPredij,'TObs[',i,',0]=',TObs1[i,0]) #OK, not a great test, but let's see it work
    #print('mean square error = ',meanSquareError(TObs1,TPred))

    for i in range(0,100):
        in2 = np.empty([1, 3], dtype=float)
        in2[0,0]=inputs[i,0]
        in2[0,1]=inputs[i,1]
        in2[0,2]=inputs[i,2]
        TPredij = KGANN.predict(in2)
        print('TPred2 RAW',i,'=',in2[0,0],in2[0,1],in2[0,2],TPredij,'Target=',targets[i,0])
        #print('TPred2',i,'=',in2[0,0],in2[0,1],in2[0,2],KGANN.unconvertTij(TPredij),'Target=',KGANN.unconvertTij(targets[i,0]))

    #this is computationally intensive - compute the mean trips error to see whether the training is aacceptable
    KGANN.calculateCBarError(TObs1,Cij1)
    
    #time inference time
    starttime = time.time()
    KGANN.predict(inputs)
    finishtime = time.time()
    print('Inference time: ',finishtime-starttime, ' seconds')

###############################################################################

def testKerasGravityANNInference(matrixN,numHiddens):
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
    Oi = KGANN.calculateOi(TObs1)
    Dj = KGANN.calculateDj(TObs1)
    KGANN.targetOi = Oi #these three, targetOi/Dj/Cij are used for evaluating DjPred every epoch
    KGANN.targetDj = Dj
    KGANN.targetCij = Cij1
    #now we need to make an input set which is [Oi,Dj,Cij] with a target of Tij
    print("Building training set - this might take a while...")
    count = countNonZero(TObs1,Cij1) #make count N*N if you want everything
    #count=100 #HACK!!!!
    #count=N*N #do this to include EVERY sample - including zero ones
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
            inputs[dataidx,1]=Dj[j] #max(Dj[j],0.001)
            inputs[dataidx,2]=Cij1[i,j] #max(Cij1[i,j],0.001)
            targets[dataidx,0]=TObs1[i,j] #max(TObs1[i,j],0.001)
            dataidx+=1
            if dataidx>=count: break #this was really to allow me to set count=1000 for debugging (also break below)
        #end for j
        if dataidx>=count: break
    #end for i

    #raw inputs must be normalised for input to the ANN [0..1]
    KGANN.normaliseInputsLinear(inputs,targets)

    #time inference time
    starttime = time.time()
    KGANN.predict(inputs)
    finishtime = time.time()
    print('Inference time: ',finishtime-starttime, ' seconds')

###############################################################################

"""
Take a trained model file (h5) and run the CBar test on it to determine goodness of fit against
the proper matrix
"""
def testKerasCBarError(filename):
    print("test_KerasGravityANN.py::testKerasCBarError\n")
    TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))

    KGANN = KerasGravityANN([256,256,256]) #doesn't matter how many hiddens we pass as we're loading a new one anyway
    #KGANN.loadModel(filename)
    KGANN.model.load_weights(filename)
    KGANN.isMeanSDScale=True #need this as the model was trained on sd mean normalisation
    #and need to put these numbers back in for all the normalisation to work
    KGANN.meanOi= 7741.307142857143
    KGANN.meanDj= 2406.9841013824885
    KGANN.meanCij= 41.98041474654378
    KGANN.meanTij= 16.243087557603687
    KGANN.sdOi= 7106.816265087332
    KGANN.sdDj= 728.4594067969223
    KGANN.sdCij= 24.522797327549736
    KGANN.sdTij= 9.358829127240579
    TijPred = KGANN.calculateCBarError(TObs1,Cij1)
    print("MSE=",meanSquareError(TObs1,TijPred))

###############################################################################



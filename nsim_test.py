#isolated test of neural spatial interaction model

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
K.set_floatx('float64')
from tensorflow.keras.callbacks import LambdaCallback, CSVLogger, Callback
from sklearn.preprocessing import StandardScaler

#deprecated from tensorflow.keras.backend import set_session
#doesn't work: from tensorflow.keras.backend import manual_variable_initialization manual_variable_initialization(True)

import numpy as np
from math import exp, log, fabs
import time
import os
import random
import pickle

meanOi = 0.0
meanDj = 0.0
meanCij = 0.0
meanTij = 0.0
sdOi = 0.0
sdDj = 0.0
sdCij = 0.0
sdTij = 0.0


###############################################################################

"""
Load a numpy matrix from a file
"""
def loadMatrix(filename):
    with open(filename,'rb') as f:
        matrix = pickle.load(f)
    return matrix

###############################################################################

"""
Calculate Oi for a trips matrix.
This is the fast method taken from SingleOrigin.py
Needed for training.
"""
def calculateOi(Tij):
    (M, N) = np.shape(Tij)
    Oi = np.zeros(N)
    Oi=Tij.sum(axis=1)
    return Oi

###############################################################################

"""
Calculate Dj for a trips matrix.
This is the fast method taken from SingleOrigin.py
Needed for training.
"""
def calculateDj(Tij):
    (M, N) = np.shape(Tij)
    Dj = np.zeros(N)
    Dj=Tij.sum(axis=0)
    return Dj

###############################################################################

def normaliseInputsMeanSD(inputs,targets):
    #find max values on each column
    #maxCols = np.amax(inputs,axis=0) # [maxOi,maxDj,maxCij]
    meanCols = np.mean(inputs,axis=0)
    sdCols = np.std(inputs,axis=0)
    #clacualte and save all the means and stddev as we're normalising by (x-mean)/sd
    meanOi = meanCols[0]
    meanDj = meanCols[1]
    meanCij = meanCols[2]
    meanTij = np.mean(targets)
    sdOi = sdCols[0]
    sdDj = sdCols[1]
    sdCij = sdCols[2]
    sdTij = np.std(targets)
    print('meanOi=',meanOi,'meanDj=',meanDj,'meanCij=',meanCij,'meanTij=',meanTij)
    print('sdOi=',sdOi,'sdDj=',sdDj,'sdCij=',sdCij,'sdTij=',sdTij)
    #now linearise and scale the data
    inputs[:,0]=(inputs[:,0]-meanOi)/sdOi
    inputs[:,1]=(inputs[:,1]-meanDj)/sdDj
    inputs[:,2]=(inputs[:,2]-meanCij)/sdCij
    targets[:]=(targets[:]-meanTij)/sdTij

###############################################################################

"""
Evaluate from the training data to get the TPred matrix that we can compare to the
other methods.
@param TObs Input matrix, used to calculate Oi and Dj
@param Cij Cost matrix
@returns TPred predicted matrix which can be compared to TObs for accuracy
"""
def predictMatrix(TObs,Cij):
    (M, N) = np.shape(TObs)
    Oi = calculateOi(TObs)
    Dj = calculateDj(TObs)
    Tij = np.empty([N, N], dtype=float)
    inputs = np.empty([N*N,3], dtype=float)
    for i in range(0,N):
        if i%100==0:
            print('i=',i)
        for j in range(0,N):
            inputs[i*N+j,0]=log(Oi[i]+0.001)
            inputs[i*N+j,1]=log(Dj[j]+0.001)
            inputs[i*N+j,2]=Cij[i,j]
        #end for j
    #end for i
    Tij=model.predict(inputs,batch_size=10240).reshape([N,N])
    #unconvert - slow! could make this a vector!
    for i in range(0,N):
        for j in range(0,N):
            Tij[i,j]=Tij[i,j]*sdTij+meanTij #REMEMBER to unnormalise!!!
            #check range of data about to go to exp, otherwise you get a numeric range error
            if Tij[i,j]>5:
                Tij[i,j]=5
            if Tij[i,j]<0:
                Tij[i,j]=0.001
            Tij[i,j]=exp(Tij[i,j])-0.001 #unconvert as it's actually predicting log(Tij)
    #unconvert
    return Tij

###############################################################################

def filterValidData(i,j,T,C):
    #returns true if data is valid for i,j,T,C
    #note i,j are zone indices, T is the nimber of trips between i and j and C is the cost
    #return i!=j and T>=10 and C>=30 #this is what the initial training used
    #return i!=j and T>=10 and C>=25
    #return T>=5 and C>10
    #if T<1:
    #    return random.random()>0.99 #take 1% of the zero data
    #else:
    #    return random.random()>0.95 #and 5% of the non zero data
    #return True
    #return i<100
    return T>=0

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

#create network
model=Sequential()
#model.add(Dense(4, input_dim=3, dtype='float64', activation='sigmoid', kernel_initializer='normal', use_bias=True)) #relu=f(x)=max(0,x)
#model.add(BatchNormalization())
#model.add(Dense(32, dtype='float64', activation='sigmoid'))
#model.add(Dense(4, dtype='float64', activation='sigmoid', use_bias=True))
#model.add(Dense(4, dtype='float64', activation='sigmoid', use_bias=True))
#model.add(Dense(1, dtype='float64', activation='sigmoid', kernel_initializer='normal')) #sigmoid=S(x)=1/(1+exp(-x))

#new code - build from the numHiddens list...
#relu or sigmoid or linear?
#initialisers: normal, random_uniform, truncated_normal, lecun_uniform, lecun_normal, glorot_normal, glorot_uniform, he_normal, he_uniform
#dtype=float64
model.add(Dense(256, input_dim=3, kernel_initializer='he_uniform', use_bias=True))
#model.add(layers.BatchNormalization())
model.add(Activation("relu"))
#model.add(Dropout(0.2))
#model.add(Dense(numHiddens[h], activation='relu', kernel_initializer='he_uniform', use_bias=True))
model.add(Dropout(0.2))
model.add(Dense(1, activation='linear'))
#model.add(Dense(1, activation='relu', kernel_initializer='random_uniform'))

##logarithmic loss??? https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
opt = optimizers.SGD(lr=0.01, momentum=0.9)
#model.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['mse'])
model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mse'])

#load model and previous weights here if needed
#model = tf.keras.models.load_model("KerasGravityANN_20191005_195657.h5")
#K.set_value(model.optimizer.lr,0.001)
print('setLearningRate: lr=',K.get_value(model.optimizer.lr))
###end loading

##loading data
TObs1 = loadMatrix("data/fromQUANT/Py_TObs_road.bin") #alternate matrices built directly from the QUANT training data
Cij1 = loadMatrix("data/fromQUANT/Py_Cij_road.bin")
(M, N) = np.shape(TObs1)
Oi = calculateOi(TObs1)
Dj = calculateDj(TObs1)
#now build into a training set
print("Building training set - this might take a while...")
count = countNonZero(TObs1,Cij1) #make count N*N if you want everything
print('Found ',count,' non-zero entries in TObs')
inputs = np.empty([count, 3], dtype=float)
targets = np.empty([count,1], dtype=float)
nextpct = 0
dataidx=0
mseTObsTPred = 0.0
for i in range(0,N):
    pct = i/N*100
    if pct>=nextpct:
        print(pct," percent complete")
        nextpct+=10
    #balance
    beta = 0.137
    Ai=0.0
    for j in range(0,N):
        Ai+=Dj[j]*exp(-beta * Cij1[i,j])
    Ai=1.0/Ai
    #end balance
    for j in range(0,N):
        if not filterValidData(i,j,TObs1[i,j],Cij1[i,j]):
            continue #HACK!
        inputs[dataidx,0]=log(Oi[i]+0.001)
        inputs[dataidx,1]=log(Dj[j]+0.001)
        inputs[dataidx,2]=-Cij1[i,j]
        targets[dataidx,0]=log(TObs1[i,j]+0.001)
        #targets[dataidx,0]=log(Ai*Oi[i]*Dj[j]*exp(-beta*Cij1[i,j]))
        #this is a mean square error calculation for checking TOvs against TPred
        deltaTij = TObs1[i,j] - Ai*Oi[i]*Dj[j]*exp(-beta*Cij1[i,j])
        mseTObsTPred+=deltaTij*deltaTij
        ##
        dataidx+=1
        if dataidx>=count: break #this was really to allow me to set count=1000 for debugging (also break below)
    #end for j
    if dataidx>=count: break
#end for i

mseTObsTPred/=count
print("mseTObsTPred=",mseTObsTPred)

#normalise data
normaliseInputsMeanSD(inputs,targets)

##training
numEpochs = 10 #400
batchSize = 102400 #10240

trainLogFilename='KerasGravityANN_'
trainTimestamp = time.strftime('%Y%m%d_%H%M%S')
csv_logger = CSVLogger(trainLogFilename+trainTimestamp+'.csv', separator=',', append=False)
model.fit(inputs, targets, epochs=numEpochs, shuffle=True, batch_size=batchSize, verbose=1, validation_split=0.2, callbacks=[csv_logger])
        
#save the model for later
model.save(trainLogFilename+trainTimestamp+'.h5')

#do my own evaluation this is mean squared error between TObs and the ANN TPred matrix we just trained
TijANNPred = predictMatrix(TObs1,Cij1)
mseANN = 0
for i in range(0,N):
    for j in range(0,N):
        delta = TObs1[i,j]-TijANNPred[i,j]
        mseANN+=delta*delta
mseANN/=(N*N)
print("mseANN=",mseANN)
###

#evaluate the model to see how well it did
score = model.evaluate(inputs,targets,verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
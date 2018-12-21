from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model
from keras import optimizers
from keras import backend as K
K.set_floatx('float64')

import numpy as np
from math import exp, fabs
import time
import os

"""
Artificial Neural Network model using Keras for gravity model
This is the example to read: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
ADAM gradient descent algorithm: http://arxiv.org/abs/1412.6980
"""

class KerasGravityANN:
    ###############################################################################

    def __init__(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #RUN ON CPU!!!!!!!!!!!!!!!!!

        np.random.seed(42) #set random seed for repeatability
        #gravity model
        self.numModes=3
        #ANN
        self.scaleOiDj=1 #scaling for Oi and Dj inputs to ANN to bring inputs into [0..1] range
        self.scaleCij=1 #scaling for Cij inputs to ANN to bring inputs into [0..1] range
        self.scaleTij=1 #scaling for Tij targets for ANN to bring targets into [0..1] range
        #self.numLayers???
        #self.batchSize=10 #training batch size
        #dropout?
        self.model=self.createNetwork()

    ###############################################################################

    """
    Calculate Oi for a trips matrix.
    This is the fast method taken from SingleOrigin.py
    Needed for training.
    """
    def calculateOi(self,Tij):
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
    def calculateDj(self,Tij):
        (M, N) = np.shape(Tij)
        Dj = np.zeros(N)
        Dj=Tij.sum(axis=0)
        return Dj

    ###############################################################################

    """
    Create the neural network model and compile it. This creates the whole network for
    training, including the loss metrics.
    @returns the model
    """
    def createNetwork(self):
        model=Sequential()
        model.add(Dense(128, input_dim=3, activation='sigmoid')) #relu=f(x)=max(0,x)
        #model.add(Dense(64, activation='sigmoid'))
        #model.add(Dense(64, activation='sigmoid'))
        model.add(Dense(1, activation='sigmoid')) #sigmoid=S(x)=1/(1+exp(-x))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae','accuracy'])
        #model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae','accuracy'])
        #sgd = optimizers.SGD(lr=0.1, decay=0.0, momentum=0.0, nesterov=False)
        #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae','accuracy']) #use sgd with custom params

        return model
    
    ###############################################################################

    def loadModel(self,filename):
        self.model=load_model(filename)

    ###############################################################################


    """
    Normalise the inputs and outputs to fit between +-1. Uses linear normalisation.
    @param inputs [ [Oi,Dj,Cij], ... ]
    @param targets [ Tij, Tij, Tij... ] #TObs values to match the input triples
    POST: Modifies inputs and trainingSet in place to normalise to values suitable for ANN input 
    """
    def normaliseInputsLinear(self,inputs,targets):
        #OK, I'm going to do a simple normalisation on [Oi,Dj] together between [0..1] [0..max]
        #and Tij and Cij separately [0..1] [0..max] which gives me 3 linear constants that
        #I'm going to need for conversion.
        #maxOi = max(inputs,key=itemgetter(0))[0]
        #maxDj = max(inputs,key=itemgetter(1))[1]
        #maxCij = max(inputs,key=itemgetter(2))[2]

        maxOi=0
        maxDj=0
        maxCij=0
        maxTij=0
        for i in range(0,len(inputs)):
            if inputs[i,0]>maxOi:
                maxOi = inputs[i,0]
            if inputs[i,1]>maxDj:
                maxDj = inputs[i,1]
            if inputs[i,2]>maxCij:
                maxCij = inputs[i,2]
            if targets[i]>maxTij:
                maxTij = targets[i,0]
        print('first max Oi = ',maxOi,' first max Dj = ',maxDj,' first maxCij=',maxCij,' first maxTij=',maxTij)

        #this is rubbish - doesn't work properly
        #maxCols = np.amax(inputs,axis=1) # [maxOi,maxDj,maxCij]
        #maxOi = maxCols[0]
        #maxDj = maxCols[1]
        #maxCij = maxCols[2]
        #maxTij = np.amax(targets)
        #end
        maxOiDj = max(maxOi,maxDj) #composite max of origins and destinations - should be approx same magnitude
        #calculate and save the linear scale factors as we will need it for inference
        self.scaleOiDj=1/maxOiDj
        self.scaleCij=1/maxCij
        self.scaleTij=1/maxTij
        print('maxOi=',maxOi,'maxDj=',maxDj,'maxCij=',maxCij,'maxTij=',maxTij)
        print('scaleOiDj=',self.scaleOiDj,'scaleCij=',self.scaleCij,'scaleTij=',self.scaleTij)
        #now scale the data
        inputs[:,0]*=self.scaleOiDj
        inputs[:,1]*=self.scaleOiDj
        inputs[:,2]*=self.scaleCij
        targets*=self.scaleTij

        #results returned in inputs and targets arrays which are now modified and scaled for ANN input


    ###############################################################################

    """
    Train model to fit data.
    PRE: inputs and targets are NORMALISED [0..1] as suitable for input to the ANN
    @param inputs NDArray of (N*N,3) [[Oi,Dj,Cij], ..., ...]
    @param targets NDArray of (N*N) [Tij, ..., ...] to match inputs
    @numEpochs number of epochs to train for
    TODO: need to save the resulting model for later
    """
    def trainModel(self,inputs,targets,numEpochs):
        #TODO: the inputs and outputs stil
        self.model.fit(inputs, targets, epochs=numEpochs, batch_size=1000) #batch was 10 originally
        #save the model for later
        self.model.save('KerasGravityANN_'+time.strftime('%Y%m%d_%H%M%S')+'.h5')
        # evaluate the model - takes ages on fermi, very quick on xmesh though....?
        #scores = self.model.evaluate(inputs, targets)
        #print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

    ###############################################################################

    """
    Evaluation function for new data.
    @param inputs Single input line containing Oi, Dj, Cij
    """
    def predict(self,inputs):
        # calculate predictions
        predictions = self.model.predict(inputs)
        # round predictions
        #rounded = [round(x[0]) for x in predictions]
        #print(rounded)
        #print('KerasGravityANN::predictions ',predictions)
        return predictions

    ###############################################################################

    """
    Evaluate from the training data to get the TPred matrix that we can compare to the
    other methods.
    @param TObs Input matrix, used to calculate Oi and Dj
    @param Cij Cost matrix
    @returns TPred predicted matrix which can be compared to TObs for accuracy
    """
    def predictMatrix(self,TObs,Cij):
        (M, N) = np.shape(TObs)
        Oi = self.calculateOi(TObs)
        Dj = self.calculateDj(TObs)
        Tij = np.empty([N, N], dtype=float)
        inputs = np.empty([N*N,3], dtype=float)
        for i in range(0,N):
            if i%100==0:
                print('i=',i)
            for j in range(0,N):
                inputs[i*N+j,0]=Oi[i]*self.scaleOiDj
                inputs[i*N+j,1]=Dj[j]*self.scaleOiDj
                inputs[i*N+j,2]=Cij[i,j]*self.scaleTij
            #end for j
        #end for i
        Tij=self.predict(inputs).reshape([N,N])
        Tij=Tij/self.scaleTij
        return Tij

    def predictSingle(self,TObs,Cij,i,j):
        (M, N) = np.shape(TObs)
        Oi = self.calculateOi(TObs)
        Dj = self.calculateDj(TObs)
        inputs = np.empty([1,3], dtype=float)
        inputs[0,0]=Oi[i]*self.scaleOiDj
        inputs[0,1]=Dj[j]*self.scaleOiDj
        inputs[0,2]=Cij[i,j]*self.scaleCij
        Tij=self.model.predict(inputs)
        Tij=Tij/self.scaleTij
        return Tij






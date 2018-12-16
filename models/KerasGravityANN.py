from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from math import exp, fabs
import time

"""
Artificial Neural Network model using Keras for gravity model
This is the example to read: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
ADAM gradient descent algorithm: http://arxiv.org/abs/1412.6980
"""

class KerasGravityANN:
    ###############################################################################

    def __init__(self):
        np.random.seed(42) #set random seed for repeatability
        #gravity model
        self.numModes=3
        #ANN
        self.scaleOiDj=1 #scaling for Oi and Dj inputs to ANN to bring inputs into [0..1] range
        self.scaleCij=1 #scaling for Cij inputs to ANN to bring inputs into [0..1] range
        self.scaleTij=1 #scaling for Tij targets for ANN to bring targets into [0..1] range
        #self.numLayers???
        self.batchSize=10 #training batch size
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
        model.add(Dense(8, input_dim=3, activation='relu')) #relu=f(x)=max(0,x)
        model.add(Dense(8, activation='relu'))
        model.add(Dense(8, activation='sigmoid')) #sigmoid=S(x)=1/(1+exp(-x))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model
    
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
        maxCols = np.amax(inputs,axis=1) # [maxOi,maxDj,maxCij]
        maxOi = maxCols[0]
        maxDj = maxCols[1]
        maxCij = maxCols[2]
        maxTij = np.amax(targets)
        maxOiDj = max(maxOi,maxDj) #composite max of origins and destinations - should be approx same magnitude
        #calculate and save the linear scale factors as we will need it for inference
        self.scaleOiDj=1/maxOiDj
        self.scaleCij=1/maxCij
        self.scaleTij=1/maxTij
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
        # evaluate the model
        scores = self.model.evaluate(inputs, targets)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

    ###############################################################################

    def predict(self,inputs):
        # calculate predictions
        predictions = self.model.predict(inputs)
        # round predictions
        rounded = [round(x[0]) for x in predictions]
        print(rounded)

    ###############################################################################


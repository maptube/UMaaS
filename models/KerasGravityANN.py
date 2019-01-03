from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.models import load_model
from keras import optimizers
from keras import backend as K
K.set_floatx('float64')
from keras.callbacks import LambdaCallback, CSVLogger, Callback

import numpy as np
from math import exp, fabs
import time
import os
import random

"""
Artificial Neural Network model using Keras for gravity model
This is the example to read: https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
ADAM gradient descent algorithm: http://arxiv.org/abs/1412.6980
TODO:
keras.callbacks.CSVLogger(filename, separator=',', append=False)
keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
"""

class KerasGravityANN:
    ###############################################################################

    def __init__(self,numHiddens):
        #NOTE: numHiddens is a list, so [16] is 16 hiddens in layer 1, [16,16] is two hidden layers of 16
        #os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #RUN ON CPU!!!!!!!!!!!!!!!!!

        np.random.seed(42) #set random seed for repeatability
        #gravity model
        self.numModes=3
        #ANN
        self.isLogScale = False #set when you do either normaliseInputsLinear or normaliseInputsLog to enable you to get the values back
        self.scaleOiDj=1 #scaling for Oi and Dj inputs to ANN to bring inputs into [0..1] range
        self.scaleCij=1 #scaling for Cij inputs to ANN to bring inputs into [0..1] range
        self.scaleTij=1 #scaling for Tij targets for ANN to bring targets into [0..1] range
        
        #logging
        self.trainLogFilename='KerasGravityANN_'
        self.trainTimestamp=''
        #dropout?
        self.model=self.createNetwork(numHiddens)

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
    calculateCBar
    Mean trips calculation
    @param name="Tij" NDArray
    @param name="cij" NDArray
    @returns float
    """
    def calculateCBar(self,Tij,cij):
        CNumerator = np.sum(Tij*cij)
        CDenominator = np.sum(Tij)
        CBar=CNumerator/CDenominator
        #print("CBar=",CBar)
        return CBar

    ###############################################################################


    """
    Create the neural network model and compile it. This creates the whole network for
    training, including the loss metrics.
    Accuracy: mse or mae for regression, DO NOT use softmax (not probs?) use tanh instead (TODO: check sigmoid?):
    model.compile(loss='mean_absolute_error', optimizer='rmsprop')
    @param humHiddens list of numbers of neurons in hidden layer
    @returns the model
    """
    def createNetwork(self,numHiddens):
        model=Sequential()
        #model.add(Dense(4, input_dim=3, dtype='float64', activation='sigmoid', kernel_initializer='normal', use_bias=True)) #relu=f(x)=max(0,x)
        #model.add(BatchNormalization())
        #model.add(Dense(32, dtype='float64', activation='sigmoid'))
        #model.add(Dense(4, dtype='float64', activation='sigmoid', use_bias=True))
        #model.add(Dense(4, dtype='float64', activation='sigmoid', use_bias=True))
        #model.add(Dense(1, dtype='float64', activation='sigmoid', kernel_initializer='normal')) #sigmoid=S(x)=1/(1+exp(-x))

        #new code - build from the numHiddens list...
        model.add(Dense(numHiddens[0], input_dim=3, dtype='float64', activation='sigmoid', kernel_initializer='normal', use_bias=True))
        for h in range(1,len(numHiddens)):
            model.add(Dense(numHiddens[h], dtype='float64', activation='sigmoid', kernel_initializer='normal', use_bias=True))
        model.add(Dense(1, dtype='float64', activation='sigmoid', kernel_initializer='normal'))

        # Compile model
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse','mae'])
        #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae','accuracy'])
        #model.compile(loss='mean_squared_error', optimizer='sgd', metrics=['mae','accuracy'])
        #sgd = optimizers.SGD(lr=0.9, decay=0.01, momentum=0.1, nesterov=True)
        #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mae','accuracy']) #use sgd with custom params
        #V2
        #model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mae'])
        model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse','mae'])
        #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])

        return model
    
    ###############################################################################

    def loadModel(self,filename):
        self.model=load_model(filename)

    ###############################################################################


    """
    Three (un) convert functions which enable the raw Oi, Dj, Cij and Tij values
    to be retrieved from the input data after calling either normaliseInputsLinear
    or normaliseInputsLog.
    NOTE: the two normalise functions don't use the functions here due to speed - they
    operate on large sets of vector data and these individual conversions would be too
    slow. The code here is used mainly for debugging small sets of output data.
    """
    def convertOiDj(self,OiDj):
        if self.isLogScale:
            return np.log(OiDj)*self.scaleOiDj
        else:
            return OiDj * self.scaleOiDj + 0.1

    def convertCij(self,Cij):
        if self.isLogScale:
            return np.log(Cij)*self.scaleCij
        else:
            return Cij * self.scaleCij + 0.1

    def convertTij(self,Tij):
        if self.isLogScale:
            return np.log(Tij)*self.scaleTij
        else:
            return Tij * self.scaleTij + 0.1
    ##

    def unconvertOiDj(self,OiDj):
        if self.isLogScale:
            return np.exp(OiDj/self.scaleOiDj)
        else:
            return (OiDj-0.1)/self.scaleOiDj

    def unconvertCij(self,Cij):
        if self.isLogScale:
            return np.exp(Cij/self.scaleCij)
        else:
            return (Cij-0.1)/self.scaleCij

    def unconvertTij(self,Tij):
        if self.isLogScale:
            return np.exp(Tij/self.scaleTij)
        else:
            return (Tij-0.1)/self.scaleTij

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
        self.isLogScale = False

        maxOi=0
        maxDj=0
        maxCij=0
        maxTij=0
        #slow method of finding max of Oi, Dj, Cij and Tij inputs - verified as identical to fast method below
        #the slow method is fine for ~200,000 inputs, but at 54 million, you really need the fast one
        #for i in range(0,len(inputs)):
        #    if inputs[i,0]>maxOi:
        #        maxOi = inputs[i,0]
        #    if inputs[i,1]>maxDj:
        #        maxDj = inputs[i,1]
        #    if inputs[i,2]>maxCij:
        #        maxCij = inputs[i,2]
        #    if targets[i]>maxTij:
        #        maxTij = targets[i,0]
        #print('first max Oi = ',maxOi,' first max Dj = ',maxDj,' first maxCij=',maxCij,' first maxTij=',maxTij)

        #this is the fast method - you can compare it to the above as a test, but they're verified as identical
        maxCols = np.amax(inputs,axis=0) # [maxOi,maxDj,maxCij]
        maxOi = maxCols[0]
        maxDj = maxCols[1]
        maxCij = maxCols[2]
        maxTij = np.amax(targets)
        #end fast method
        maxOiDj = max(maxOi,maxDj) #composite max of origins and destinations - should be approx same magnitude
        #calculate and save the linear scale factors as we will need it for inference
        self.scaleOiDj=0.8/maxOiDj #scale to 0.1..0.9
        self.scaleCij=0.8/maxCij
        self.scaleTij=0.8/maxTij
        print('maxOi=',maxOi,'maxDj=',maxDj,'maxCij=',maxCij,'maxTij=',maxTij)
        print('scaleOiDj=',self.scaleOiDj,'scaleCij=',self.scaleCij,'scaleTij=',self.scaleTij)
        #now scale the data
        inputs[:,0]=inputs[:,0]*self.scaleOiDj+0.1
        inputs[:,1]=inputs[:,1]*self.scaleOiDj+0.1
        inputs[:,2]=inputs[:,2]*self.scaleCij+0.1
        targets[:]=targets[:]*self.scaleTij+0.1

        #results returned in inputs and targets arrays which are now modified and scaled for ANN input


    ###############################################################################

    """
    Same as normaliseInputsLinear, except this one normalises by the ln() value of the column maximum value.
    @param inputs [ [Oi,Dj,Cij], ... ]
    @param targets [ Tij, Tij, Tij... ] #TObs values to match the input triples
    POST: Modifies inputs and trainingSet in place to normalise to values suitable for ANN input 
    """
    def normaliseInputsLog(self,inputs,targets):
        #OK, I'm going to do a simple logarithmic normalisation on [Oi,Dj] together between [0..1] [0..max]
        #and Tij and Cij separately [0..1] [0..max] which gives me 3 constants that I'm going to need for conversion.
        #Normalisation function: newX = ln(X)*scaleX
        #Reverse function: X = exp(newX/scaleX)

        self.isLogScale=True

        #find max values on each column
        maxCols = np.amax(inputs,axis=0) # [maxOi,maxDj,maxCij]
        maxOi = np.log(maxCols[0])
        maxDj = np.log(maxCols[1])
        maxCij = np.log(maxCols[2])
        maxTij = np.log(np.amax(targets))
        maxOiDj = max(maxOi,maxDj) #composite max of origins and destinations - should be approx same magnitude
        #calculate and save the linear scale factors as we will need it for inference
        self.scaleOiDj=1/maxOiDj
        self.scaleCij=1/maxCij
        self.scaleTij=1/maxTij
        print('lnMaxOi=',maxOi,'lnMaxLnDj=',maxDj,'lnMaxCij=',maxCij,'lnMaxTij=',maxTij)
        print('scaleOiDj=',self.scaleOiDj,'scaleCij=',self.scaleCij,'scaleTij=',self.scaleTij)
        #now linearlise and scale the data to fit in 0..1
        inputs[:,0]=np.log(inputs[:,0])*self.scaleOiDj
        inputs[:,1]=np.log(inputs[:,1])*self.scaleOiDj
        inputs[:,2]=np.log(inputs[:,2])*self.scaleCij
        targets[:]=np.log(targets[:])*self.scaleTij

        #results returned in inputs and targets arrays which are now modified and scaled for ANN input


    """
    Train model to fit data.
    PRE: inputs and targets are NORMALISED [0..1] as suitable for input to the ANN
    @param inputs NDArray of (N*N,3) [[Oi,Dj,Cij], ..., ...]
    @param targets NDArray of (N*N) [Tij, ..., ...] to match inputs
    @batchSize size of training batch - error updates are made over the batch until all training data seen, which is an epoch
    @numEpochs number of epochs to train for
    TODO: need to save the resulting model for later
    """
    def trainModel(self,inputs,targets,batchSize,numEpochs):
        self.trainTimestamp = time.strftime('%Y%m%d_%H%M%S')
        csv_logger = CSVLogger(self.trainLogFilename+self.trainTimestamp+'.csv', separator=',', append=False)

        #First version - train on batch data
        #self.model.fit(inputs, targets, epochs=numEpochs, shuffle=True, batch_size=batchSize, verbose=1, callbacks=[csv_logger])
        #Second version - train using a generator
        self.model.fit_generator(self.generator(inputs, targets, batchSize), steps_per_epoch=1, epochs=numEpochs, verbose=1, callbacks=[csv_logger])
        
        #save the model for later
        #self.model.save('KerasGravityANN_'+time.strftime('%Y%m%d_%H%M%S')+'.h5')
        self.model.save(self.trainLogFilename+self.trainTimestamp+'.h5')
        # evaluate the model - takes ages on fermi, very quick on xmesh though....?
        #scores = self.model.evaluate(inputs, targets)
        #print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

    ###############################################################################

    def generator(self, inputs, targets, batchSize):
        # Create empty arrays to contain batch of features and labels#
        batchInputs = np.zeros((batchSize, 3))
        batchTargets = np.zeros((batchSize,1))
        while True:
            for i in range(batchSize):
                # choose random index in features
                #index= random.choice(len(inputs),1)
                index = random.randrange(0,len(inputs))
                #batchInputs[i] = some_processing(inputs[index]) what????
                batchInputs[i,:] = inputs[index,:]
                batchTargets[i] = targets[index]
            yield batchInputs, batchTargets

    ###############################################################################

    def predgenerator(self, inputs, targets, batchSize):
        #TODO!!!!
        # Create empty arrays to contain batch of features and labels#
        batchInputs = np.zeros((batchSize, 3))
        batchTargets = np.zeros((batchSize,1))
        while True:
            for i in range(batchSize):
                # choose random index in features
                #index= random.choice(len(inputs),1)
                index = random.randrange(0,len(inputs))
                #batchInputs[i] = some_processing(inputs[index]) what????
                batchInputs[i,:] = inputs[index,:]
                batchTargets[i] = targets[index]
            yield batchInputs, batchTargets

    ###############################################################################

    """
    Evaluation function for new data.
    @param inputs Single input line containing Oi, Dj, Cij
    """
    def predict(self,inputs):
        # calculate predictions
        #predictions = self.model.predict(inputs)
        predictions = self.model.predict_on_batch(inputs)
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
                inputs[i*N+j,0]=Oi[i]
                inputs[i*N+j,1]=Dj[j]
                inputs[i*N+j,2]=Cij[i,j]
            #end for j
        #end for i
        inputs[:,0]=self.convertOiDj(inputs[:,0])
        inputs[:,1]=self.convertOiDj(inputs[:,1])
        inputs[:,2]=self.convertCij(inputs[:,2])
        #inputs[:,0]=inputs[:,0]*self.scaleOiDj+0.1
        #inputs[:,1]=inputs[:,1]*self.scaleOiDj+0.1
        #inputs[:,2]=inputs[:,2]*self.scaleCij+0.1
        Tij=self.predict(inputs).reshape([N,N])
        Tij=self.unconvertTij(Tij)
        return Tij

    #def predictMatrixFromOiDjCij(self,Oi,Dj,Cij):
    #    #todo
    #    N = len(Oi)
    #    inputs = np.empty([N*N,3], dtype=float)
    #    for i in range(0,N):
    #        for j in range(0,N):
    #            inputs[i*N+j,0]=self.convertOiDj(Oi[i])
    #            inputs[i*N+j,1]=self.convertOiDj(Dj[j])
    #            inputs[i*N+j,2]=self.convertCij(Cij[i,j])
    #        #end for j
    #    #end for i
    #    Tij=self.predict(inputs).reshape([N,N])
    #    Tij=self.convertTij(Tij)
    #    return Tij

    def predictSingle(self,TObs,Cij,i,j):
        (M, N) = np.shape(TObs)
        Oi = self.calculateOi(TObs)
        Dj = self.calculateDj(TObs)
        inputs = np.empty([1,3], dtype=float)
        inputs[0,0]=self.convertOiDj(Oi[i])
        inputs[0,1]=self.convertOiDj(Dj[j])
        inputs[0,2]=self.convertCij(Cij[i,j])
        Tij=self.model.predict(inputs)
        Tij=self.unconvertTij(Tij)
        return Tij

    def calculateCBarError(self,TObs,Cij):
        #calculate the difference in the mean trip distances between the observed CBar and predicted CBar
        print('calculateCBarError...')
        CBarObs = self.calculateCBar(TObs,Cij)
        TijPred = self.predictMatrix(TObs,Cij)
        for i in range(0,500):
            print('CBAR TObs, TPred: ',TObs[i,0],' ',TijPred[i,0])
        CBarPred = self.calculateCBar(TijPred,Cij)
        #then work out error between self.Dj and DjPred - you subtract the sums
        error = CBarPred-CBarObs
        print('CBar error = ',error,' CBarObs=',CBarObs,' CBarPred=',CBarPred)

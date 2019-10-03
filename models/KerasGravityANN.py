import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
K.set_floatx('float64')
from tensorflow.keras.callbacks import LambdaCallback, CSVLogger, Callback
#from sklearn.preprocessing import StandardScaler

#deprecated from tensorflow.keras.backend import set_session
#doesn't work: from tensorflow.keras.backend import manual_variable_initialization manual_variable_initialization(True)

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

        #np.random.seed(42) #set random seed for repeatability
        #gravity model
        self.numModes=3
        #ANN
        self.isLogScale = False #set when you do either normaliseInputsLinear or normaliseInputsLog to enable you to get the values back
        self.isMeanSDScale = False #set to true when meanSDNormalisation is used - mean[OiDjCijTid] and sd[OiDjCijTid] used in this case
        self.scaleOiDj=1 #scaling for Oi and Dj inputs to ANN to bring inputs into [0..1] range
        self.scaleCij=1 #scaling for Cij inputs to ANN to bring inputs into [0..1] range
        self.scaleTij=1 #scaling for Tij targets for ANN to bring targets into [0..1] range
        self.meanOi = 0 #these are the means for the Oi, Dj, Cij and Tij values, used for meand stddev normalisation
        self.meanDj = 0
        self.meanCij = 0
        self.meanTij = 0
        self.sdOi = 1.0 #these are the stddevs for the Oi, Dj, Cij and Tij values, used for meand stddev normalisation
        self.sdDj = 1.0
        self.sdCij = 1.0
        self.sdTij = 1.0
        
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
        #relu or sigmoid or linear?
        #initialisers: normal, random_uniform, truncated_normal, lecun_uniform, lecun_normal, glorot_normal, glorot_uniform, he_normal, he_uniform
        #dtype=float64
        model.add(Dense(numHiddens[0], input_dim=3, kernel_initializer='he_uniform', use_bias=True))
        #model.add(layers.BatchNormalization())
        model.add(Activation("relu"))
        #model.add(Dropout(0.2))
        for h in range(1,len(numHiddens)):
            model.add(Dense(numHiddens[h], activation='relu', kernel_initializer='he_uniform', use_bias=True))
            model.add(Dropout(0.2))
        model.add(Dense(1, activation='linear'))
        #model.add(Dense(1, activation='relu', kernel_initializer='random_uniform'))

        # Compile model
        # shuffle=True to shuffle batches?
        #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mse','mae'])
        #model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae','accuracy']) #this for 256,256,256
        #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae','accuracy'])
        #model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=['mae','accuracy'])
        #sgd = optimizers.SGD(lr=0.1, decay=0.0, momentum=0.0, nesterov=False)
        #model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['mse', 'mae','accuracy']) #use sgd with custom params
        #V2
        #model.compile(loss='mean_absolute_error', optimizer='rmsprop', metrics=['mae'])
        #model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['mse','mae']) #<<this one
        #model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse','mae'])

        #rmsprop = optimizers.rmsprop(lr=0.0001, rho=0.9, epsilon=None, decay=0.000001)
        #model.compile(loss='mean_squared_error', optimizer=rmsprop, metrics=['mse','mae'])

        #???
        #sgd = optimizers.SGD(lr=0.01, momentum=0, decay=0)
        #model.compile(loss='mean_squared_error', optimizer=sgd, metrics=['mse','mae'])

        ##
        #256-256-256 model relu lin
        #model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mae','accuracy'])

        ##logarithmic loss??? https://machinelearningmastery.com/how-to-choose-loss-functions-when-training-deep-learning-neural-networks/
        opt = optimizers.SGD(lr=0.00001, momentum=0.9)
        model.compile(loss='mean_squared_logarithmic_error', optimizer=opt, metrics=['mse'])

        #adagrad, adadelta? adamax nadam
        #model.compile(loss='mean_absolute_error', optimizer='adagrad', metrics=['mae','accuracy'])
        #model.compile(loss='mean_squared_error', optimizer='adadelta', metrics=['mse','mae'])

        print('Learning rate at creation: ',K.get_value(model.optimizer.lr))

        return model
    
    ###############################################################################

    def loadModel(self,filename):
        #this hopefully makes the loading work
        #sess = tf.Session()
        #set_session(sess)
        #sess.run(tf.global_variables_initializer())
        ###
        #self.model=load_model(filename)
        self.model = tf.keras.models.load_model(filename)
        #self.model.load_weights(filename)

    def setLearningRate(self,lr):
        K.set_value(self.model.optimizer.lr,lr)
        print('setLearningRate: lr=',K.get_value(self.model.optimizer.lr))

    ###############################################################################

    def fudgeNormalisationScalingData(self):
        #this is a kludge to fit the sd and mean data for any target and input dataset
        #to match the data that the model was trained with i.e. the small Tij>10 Cij>30 dataset
        #self.meanOi= 7741.307142857143
        #self.meanDj= 2406.9841013824885
        #self.meanCij= 41.98041474654378
        #self.meanTij= 16.243087557603687
        #self.sdOi= 7106.816265087332
        #self.sdDj= 728.4594067969223
        #self.sdCij= 24.522797327549736
        #self.sdTij= 9.358829127240579
        #this is for the full 1951743 data Tij>=1, Cij>=1
        #self.meanOi= 3368.98435
        #self.meanDj= 2093.4135
        #self.meanCij= 35.34465
        #self.meanTij= 6.90615
        #self.sdOi= 3797.9759197110707
        #self.sdDj= 756.3374526742354
        #self.sdCij= 42.58190421267552
        #self.sdTij= 22.40629023683974
        #
        #this is for the full 52 million matrix data with zeros
        self.meanOi= 2009.6840716567144
        self.meanDj= 2009.6840716567144
        self.meanCij= 130.98284110928213
        self.meanTij= 0.27908402605981314
        self.sdOi= 2266.1299971323974
        self.sdDj= 730.7077889507163
        self.sdCij= 70.66869631022215
        self.sdTij= 5.138515211737166

    """
    Three (un) convert functions which enable the raw Oi, Dj, Cij and Tij values
    to be retrieved from the input data after calling either normaliseInputsLinear
    or normaliseInputsLog.
    NOTE: the two normalise functions don't use the functions here due to speed - they
    operate on large sets of vector data and these individual conversions would be too
    slow. The code here is used mainly for debugging small sets of output data.
    """
    def convertOi(self,Oi):
        if self.isMeanSDScale:
            return (Oi-self.meanOi)/self.sdOi
        elif self.isLogScale:
            return np.log(Oi)*self.scaleOiDj
        else:
            return Oi * self.scaleOiDj + 0.1

    def convertDj(self,Dj):
        if self.isMeanSDScale:
            return (Dj-self.meanDj)/self.sdDj
        elif self.isLogScale:
            return np.log(Dj)*self.scaleOiDj
        else:
            return Dj * self.scaleOiDj + 0.1

    def convertCij(self,Cij):
        if self.isMeanSDScale:
            return (Cij-self.meanCij)/self.sdCij
        elif self.isLogScale:
            return np.log(Cij)*self.scaleCij
        else:
            return Cij * self.scaleCij + 0.1

    def convertTij(self,Tij):
        if self.isMeanSDScale:
            return (Tij-self.meanTij)/self.sdTij
        elif self.isLogScale:
            return np.log(Tij)*self.scaleTij
        else:
            return Tij * self.scaleTij + 0.1
    ##

    def unconvertOi(self,Oi):
        if self.isMeanSDScale:
            return (Oi * self.sdOi) + self.meanOi
        elif self.isLogScale:
            return np.exp(Oi/self.scaleOiDj)
        else:
            return (Oi-0.1)/self.scaleOiDj

    def unconvertDj(self,Dj):
        if self.isMeanSDScale:
            return (Dj * self.sdDj) + self.meanDj
        elif self.isLogScale:
            return np.exp(Dj/self.scaleOiDj)
        else:
            return (Dj-0.1)/self.scaleOiDj

    def unconvertCij(self,Cij):
        if self.isMeanSDScale:
            return (Cij * self.sdCij) + self.meanCij
        elif self.isLogScale:
            return np.exp(Cij/self.scaleCij)
        else:
            return (Cij-0.1)/self.scaleCij

    def unconvertTij(self,Tij):
        if self.isMeanSDScale:
            return (Tij * self.sdTij) + self.meanTij
        elif self.isLogScale:
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
        self.isMeanSDScale = False

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
        self.isMeanSDScale=False

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

    ###############################################################################

    def normaliseInputsMeanSD(self,inputs,targets):
        self.isLogScale=False
        self.isMeanSDScale=True

        #find max values on each column
        #maxCols = np.amax(inputs,axis=0) # [maxOi,maxDj,maxCij]
        meanCols = np.mean(inputs,axis=0)
        sdCols = np.std(inputs,axis=0)
        #clacualte and save all the means and stddev as we're normalising by (x-mean)/sd
        self.meanOi = meanCols[0]
        self.meanDj = meanCols[1]
        self.meanCij = meanCols[2]
        self.meanTij = np.mean(targets)
        self.sdOi = sdCols[0]
        self.sdDj = sdCols[1]
        self.sdCij = sdCols[2]
        self.sdTij = np.std(targets)
        #HACK! this fudges the data above to match the data that the big network was trained with
        self.fudgeNormalisationScalingData()
        print('meanOi=',self.meanOi,'meanDj=',self.meanDj,'meanCij=',self.meanCij,'meanTij=',self.meanTij)
        print('sdOi=',self.sdOi,'sdDj=',self.sdDj,'sdCij=',self.sdCij,'sdTij=',self.sdTij)
        #now linearise and scale the data
        inputs[:,0]=(inputs[:,0]-self.meanOi)/self.sdOi
        inputs[:,1]=(inputs[:,1]-self.meanDj)/self.sdDj
        inputs[:,2]=(inputs[:,2]-self.meanCij)/self.sdCij
        targets[:]=(targets[:]-self.meanTij)/self.sdTij


    ###############################################################################

    def randomiseInputOrder(self,inputs,targets):
        #this is basically just a quick hack to mix up the ordering a bit
        for i in range(0,len(inputs)):
            Oi1=inputs[i,0]
            Dj1=inputs[i,1]
            Cij1=inputs[i,2]
            Tij1=targets[i]
            j = random.randrange(0,len(inputs))
            Oi2=inputs[j,0]
            Dj2=inputs[j,1]
            Cij2=inputs[j,2]
            Tij2=targets[j]
            #now swap them over
            inputs[i,0]=Oi2
            inputs[i,1]=Dj2
            inputs[i,2]=Cij2
            targets[i]=Tij2
            inputs[j,0]=Oi1
            inputs[j,1]=Dj1
            inputs[j,2]=Cij1
            targets[j]=Tij1

    ###############################################################################

    def reduceInputData(self,newN,inputs,targets):
        #thin out the input data to a new size NOTE: this is RANDOM!
        #ASSERT newN<=N, otherwise it's going to hang!!!
        print("KerasGravityANN::reduceInputData original count=",len(inputs)," new count=",newN)
        newInputs = np.empty([newN,3], dtype=float)
        newTargets = np.empty([newN,1], dtype=float)
        for i in range(0,newN):
            i2 = random.randrange(0,len(inputs))
            while (inputs[i2,0]==-999999): i2=(i2+1)%len(inputs) #standard optimal hash search
            newInputs[i,0]=inputs[i2,0]
            newInputs[i,1]=inputs[i2,1]
            newInputs[i,2]=inputs[i2,2]
            newTargets[i]=targets[i2]
            inputs[i2,0]=-999999 #OK, that's a bit of a hack, but it prevents the same value being used twice

        return newInputs, newTargets

    ###############################################################################

    def clusterInputData(self,inputs,targets):
        #thin out the input data, but by clustering each individual Tij as a target cluster
        #return each Tij as an average of all the targets that made it
        #TODO: I'm not sure this works - you might need to do a test on how close the Oi, Dj, Cij are for each Tij and have multiple Tij
        print("KerasGravityANN::clusterInputData original count=",len(inputs))
        repeatN = 5 #this is the number from each Tij cluster to randomly include
        maxCols = np.amax(inputs,axis=0) # [maxOi,maxDj,maxCij]
        maxOi = maxCols[0]
        maxDj = maxCols[1]
        maxCij = maxCols[2]
        maxTij = np.amax(targets)
        print("KerasGravityANN::clusterInputData maxOi=",maxOi," maxDj=",maxDj," maxCij=",maxCij," maxTij=",maxTij)
        #there are 52 million rows in the training set, so let's do this with one walk over the data...
        N = len(inputs)
        cluster = {}
        for i in range(0,N):
            #if i % 1000 == 0:
            #    print(i)
            Tij = int(targets[i,0])
            if not Tij in cluster:
                cluster[Tij] = []
            cluster[Tij].append([inputs[i,0],inputs[i,1],inputs[i,2]])
        #OK, so we have cluster which contains ever Tij as a key, containing all the [Oi,Dj,Cij] for that Tij as a list
        #Now we need to figure out how to turn those data points into representative data
        newN=len(cluster)*repeatN
        print("KerasGravityANN::clusterInputData newN=",newN)
        newInputs = np.empty([newN,3], dtype=float)
        newTargets = np.empty([newN,1], dtype=float)
        idx=0
        for Tij in cluster: #pick a random data point from each individual Tij
            data = cluster[Tij]
            for c in range(0,repeatN): #pick repeatN samples randomly from data - yes, this can repeat samples
                i = random.randrange(0,len(data))
                newInputs[idx,0]=data[i][0]
                newInputs[idx,1]=data[i][1]
                newInputs[idx,2]=data[i][2]
                newTargets[idx,0]=Tij
                print(newInputs[idx,0],",",newInputs[idx,1],",",newInputs[idx,2],",",newTargets[idx,0])
                idx+=1
        return newInputs, newTargets

    ###############################################################################

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
        self.model.fit(inputs, targets, epochs=numEpochs, shuffle=True, batch_size=batchSize, verbose=1, validation_split=0.2, callbacks=[csv_logger])
        #Second version - train using a generator
        #self.model.fit_generator(self.generator(inputs, targets, batchSize), steps_per_epoch=1, epochs=numEpochs, verbose=1, callbacks=[csv_logger])
        
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
        #print("KerasGravityANN::predict inputs=",inputs)
        predictions = self.model.predict(inputs,batch_size=10240) #this is VERY slow, but guaranteed to work - batch_size makes it faster
        #predictions = self.model.predict_on_batch(inputs) #this is fast, but can exceed memory
        # round predictions
        #rounded = [round(x[0]) for x in predictions]
        #print(rounded)
        #print('KerasGravityANN::predict predictions=',predictions)
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
        inputs[:,0]=self.convertOi(inputs[:,0])
        inputs[:,1]=self.convertDj(inputs[:,1])
        inputs[:,2]=self.convertCij(inputs[:,2])
        Tij=self.predict(inputs).reshape([N,N])
        #for i in range(0,100):
        #    print("KGANN Oi Dj: ",Oi[i],Dj[i])
        #    print("KGANN Predict inputs: ",inputs[i])
        #    print("KGANN Predict outputs: ",Tij[i])
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
        inputs[0,0]=self.convertOi(Oi[i])
        inputs[0,1]=self.convertDj(Dj[j])
        inputs[0,2]=self.convertCij(Cij[i,j])
        Tij=self.model.predict(inputs)
        Tij=self.unconvertTij(Tij)
        return Tij

    def calculateCBarError(self,TObs,Cij):
        #calculate the difference in the mean trip distances between the observed CBar and predicted CBar
        print('calculateCBarError...')
        CBarObs = self.calculateCBar(TObs,Cij)
        TijPred = self.predictMatrix(TObs,Cij)
        for i in range(0,100):
            print('CBAR TObs, TPred: ',TObs[i,0],' ',TijPred[i,0])
        for j in range(0,100):
            print('CBAR TObs, TPred: ',TObs[0,j],' ',TijPred[0,j])
        CBarPred = self.calculateCBar(TijPred,Cij)
        #then work out error between self.Dj and DjPred - you subtract the sums
        error = CBarPred-CBarObs
        print('CBar error = ',error,' CBarObs=',CBarObs,' CBarPred=',CBarPred)
        return TijPred

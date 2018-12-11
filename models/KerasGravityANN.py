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
        #self.numLayers???
        self.batchSize=10 #training batch size
        #dropout?
        self.model=self.createNetwork()

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
        model.add(Dense(1, activation='sigmoid')) #sigmoid=S(x)=1/(1+exp(-x))

        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return model

    ###############################################################################

    def trainModel(self,inputs,trainingSet,numEpochs):
        self.model.fit(inputs, trainingSet, epochs=numEpochs, batch_size=10)
        # evaluate the model
        scores = self.model.evaluate(inputs, trainingSet)
        print("\n%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100))

    ###############################################################################

    def predict(self,inputs):
        # calculate predictions
        predictions = self.model.predict(inputs)
        # round predictions
        rounded = [round(x[0]) for x in predictions]
        print(rounded)

    ###############################################################################


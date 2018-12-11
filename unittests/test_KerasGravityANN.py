#Keras Gravity ANN Tests
#Unit tests test functionality i.e. one model against another and correlations and errors between results

from models.KerasGravityANN import KerasGravityANN

###############################################################################

def testKerasGravityANN():
    #load in data - we have 52 million points!

    KGANN = KerasGravityANN()
    #input is a triple of Oi, Dj, Cij
    KGANN.trainModel(inputs,trainingSet,1000)

    #todo: get the beta back out by equivalence testing and plot geographically

import tensorflow as tf
import numpy as np
from math import exp, fabs
import time

"""
Tensorflow implementation of single destination constrained gravity model
How about PyTorch or Keras?
"""
class TFSingleOrigin:
    ###############################################################################

    def __init__(self,N):
        self.N=N #7201 #number of zones - needed to be preset for the TF code (i.e. before matrix was loaded to give us N) - TODO: use TF adaptive sizes
        self.numModes=3
        self.TObs=[] #Data input to model list of NDArray
        self.Cij=[] #cost matrix for zones in TObs

        self.isUsingConstraints = False
        self.constraints = [] #1 or 0 to indicate constraints for zones matching TObs - this applies to all modes

        self.TPred=[] #this is the output
        self.B=[] #this is the constraints output vector - this applies to all modes
        self.Beta=[] #Beta values for three modes - this is also output

        #create a graph for TensorFlow to calculate the CBar value
        self.tfTij = tf.placeholder(tf.float32, shape=(self.N,self.N), name='Tij') #input tensor 1 #hack! need to have the matrix dimension here!
        self.tfCij = tf.placeholder(tf.float32, shape=(self.N,self.N), name='Cij') #input tensor 2
        self.tfCBar = self.buildTFGraphCBar()
        #create Oi graph
        self.tfOi = self.buildTFGraphOi()
        #create Dj graph
        self.tfDj = self.buildTFGraphDj()

        #create other operations here...
        self.tfBeta = tf.Variable(1.0, name='beta')
        self.tfRunModel = self.buildTFGraphRunModel()

    ###############################################################################

    """
    Build TensorFlow graph to calcualate CBar
    @param N order of matrix i.e. 7201
    """
    def buildTFGraphCBar(self):
        #TensorFlow compute graph creation here

        #define tensors
        #tfTij = tf.placeholder(tf.float32, shape=(N,N), name='Tij')
        #tfCij = tf.placeholder(tf.float32, shape=(N,N), name='Cij')
        #build graph
        CNumerator = tf.reduce_sum(tf.multiply(self.tfTij,self.tfCij))
        CDenominator = tf.reduce_sum(self.tfTij)
        tfCBar = tf.divide(CNumerator,CDenominator,name='CBar')
        #tf.math.multiply
        #tf.math.exp

        #this is how you would run it
        #with tf.Session() as sess:
        #    sess.run(tf.global_variables_initializer())
        #    print(sess.run(tfCBar, {tfTij: Tij, tfCij: Cij}))

        return tfCBar #note returning operation graph, NOT value

    ###############################################################################

    """
    Build TensorFlow graph to calculate Oi
    """
    def buildTFGraphOi(self):
        tfOi = tf.reduce_sum(self.tfTij,axis=1)
        return tfOi

    ###############################################################################

    """
    Build TensorFlow graph to calculate Dj
    """
    def buildTFGraphDj(self):
        tfDj = tf.reduce_sum(self.tfTij,axis=0)
        return tfDj

    ###############################################################################

    #todo: Build TensorFlow graph to calculate main model equation: B*Oi*Dj*exp(beta*Cij)/sigma etc
    """
    """
    def buildTFGraphRunModel(self):
        #TODO: here!!!!
        #[Oi 1 x n] [Dj n x 1]
        #formula: Tij=Oi * Dj * exp(-beta * Cij)/(sumj Dj * exp(-beta * Cij))
        tfBalance = tf.reciprocal(tf.matmul(tf.reshape(self.tfDj, shape=(1,self.N)), tf.exp(tf.negative(self.tfBeta) * self.tfCij)))
        #this is the real model
        tfRunModel = tf.multiply(
            tfBalance,
            tf.matmul(tf.reshape(self.tfOi, shape=(self.N,1)),tf.reshape(self.tfDj,shape=(1,self.N))) * tf.exp(tf.negative(self.tfBeta) * self.tfCij),
            name='result'
        )
        #this is testing
        #tfRunModel = tf.matmul(tf.reshape(self.tfOi, shape=(7201,1)),tf.reshape(self.tfDj,shape=(1,7201))) * tf.exp(tf.negative(self.tfBeta) * self.tfCij)
        return tfRunModel

    ###############################################################################

    """
    calculateCBar
    Mean trips calculation
    @param name="Tij" NDArray
    @param name="cij" NDArray
    @returns float
    """
    def calculateCBar(self,Tij,Cij):
        #(M, N) = np.shape(Tij)
        #CNumerator = 0.0
        #CDenominator = 0.0
        #for i in range(0,N):
        #    for j in range(0,N):
        #        CNumerator += Tij[i, j] * cij[i, j]
        #        CDenominator += Tij[i, j]
        #CBar = CNumerator / CDenominator
        #print("CBar=",CBar)
        #faster
        #CNumerator2 = np.sum(Tij*Cij)
        #CDenominator2 = np.sum(Tij)
        #CBar2=CNumerator2/CDenominator2
        #print("CBar2=",CBar2)

        #TensorFlow
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            #print(sess.run(self.tfCBar, {tfTij: Tij, tfCij: Cij}))
            CBar = sess.run(self.tfCBar, {self.tfTij: Tij, self.tfCij: Cij})

        return CBar

    ###############################################################################

    def calculateOi(self,Tij):
        #TensorFlow
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            #print(sess.run(self.tfCBar, {tfTij: Tij, tfCij: Cij}))
            Oi = sess.run(self.tfOi, {self.tfTij: Tij})

        return Oi

    ###############################################################################

    def calculateDj(self,Tij):
        #TensorFlow
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            #print(sess.run(self.tfCBar, {tfTij: Tij, tfCij: Cij}))
            Dj = sess.run(self.tfDj, {self.tfTij: Tij})

        return Dj

    ###############################################################################

    def runModel(self,Tij,Cij,Beta):
        #TensorFlow
        #run Tij = Ai * Oi * Dj * exp(-Beta * Cij)   where Ai = 1/sumj Dj*exp(-Beta * Cij)
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            writer = tf.summary.FileWriter("log/TFSingleDest", sess.graph)
            sess.run(tf.global_variables_initializer())
            starttime = time.time()
            for i in range(0,1000):
                Tij = sess.run(self.tfRunModel, {self.tfTij: Tij, self.tfCij: Cij, self.tfBeta: Beta})
            finishtime = time.time()
            writer.close()
        print("TFSingleDest: runModel ",finishtime-starttime," seconds")
        return Tij

    ###############################################################################

    def debugWriteModelGraph(self,Tij,Cij,Beta):
        with tf.Session() as sess:
            writer = tf.summary.FileWriter("log/TFSingleOrigin", sess.graph)
            sess.run(tf.global_variables_initializer())
            Tij = sess.run(self.tfRunModel, {self.tfTij: Tij, self.tfCij: Cij, self.tfBeta: Beta})
    
            writer.close()
            g = tf.get_default_graph()
            print(g.get_operations())
        

    ###############################################################################


    """
    Test run of equation Tij=OiDje(-BetaCij)/denom
    This is a slow implementation as a test of correct functionality. This allows
    breaking up of the TensorFlow code for verification.
    """
    def debugRunModel(self,Oi,Dj,Tij,Cij,Beta):
        (M,N) = np.shape(Tij)
        TPred = np.zeros(N*N).reshape(N, N)

        for i in range(0,N):
            #denominator calculation which is sum of all modes
            denom = 0.0  #double
            for j in range(0,N):
                denom += Dj[j] * exp(-Beta * Cij[i, j])
            #end for j
            #print("denom=",denom)

            #numerator calculation
            for j in range(0,N):
                TPred[i, j] = Oi[i] * Dj[j] * exp(-Beta * Cij[i, j]) / denom
            #print("Tijk[0,0]=",Tij[i,0])
        #end for i

        return TPred

    ###############################################################################

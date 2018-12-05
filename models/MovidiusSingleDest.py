import os.path

import tensorflow as tf
from tensorflow.python.framework import graph_io
from mvnc import mvncapi as mvnc

from globals import *
from utils import loadMatrix
from models.TFSingleDest import TFSingleDest

"""
Movidius shade processor model
To compile TF models: OpenVino Model Optimizer https://software.intel.com/en-us/articles/OpenVINO-Using-TensorFlow
mo_tf.py --input_model <INFERENCE_GRAPH>.pb --input_checkpoint <INPUT_CHECKPOINT>
mo_tf.py --input_model <INFERENCE_GRAPH>.pbtxt --input_checkpoint <INPUT_CHECKPOINT> --input_model_is_text

"""

class MovidiusSingleDest:
    ###############################################################################

    def __init__(self):
        self.numModes=3
        #self.TObs=[] #Data input to model list of NDArray
        #self.Cij=[] #cost matrix for zones in TObs

        #self.isUsingConstraints = False
        #self.constraints = [] #1 or 0 to indicate constraints for zones matching TObs - this applies to all modes

        #self.TPred=[] #this is the output
        #self.B=[] #this is the constraints output vector - this applies to all modes
        #self.Beta=[] #Beta values for three modes - this is also output

        #create a graph for TensorFlow to calculate the CBar value
        #self.tfTij = tf.placeholder(tf.float32, shape=(7201,7201), name='Tij') #input tensor 1 #hack! need to have the matrix dimension here!
        #self.tfCij = tf.placeholder(tf.float32, shape=(7201,7201), name='Cij') #input tensor 2
        #self.tfCBar = self.buildTFGraphCBar()
        #create Oi graph
        #self.tfOi = self.buildTFGraphOi()
        #create Dj graph
        #self.tfDj = self.buildTFGraphDj()

        #create other operations here...
        #self.tfBeta = tf.Variable(1.0, name='beta')
        #self.tfRunModel = self.buildTFGraphRunModel()

    ###############################################################################

    def writeGraph(self,filename):
        #this is slightly strange, but we need a graph to push to the Movidius device,
        #so why not use the TensorFlow graph from the TFSingleDest model?
        #This sets up the graph model (which we know works) and that can be saved to
        #file for use by the run function.
        TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
        Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))
        #tfModel = TFSingleDest()
        #export_dir = ""
        #builder = tf.saved_model.builder.SavedModelBuilder(export_dir)
        #with tf.Session(graph=tf.Graph()) as sess:
        #    builder.add_meta_graph_and_variables(sess,
        #                               [tag_constants.TRAINING],
        #                               signature_def_map=foo_signatures,
        #                               assets_collection=foo_assets,
        #                               strip_default_attrs=True)
        # Add a second MetaGraphDef for inference.
        #with tf.Session(graph=tf.Graph()) as sess:
        #    builder.add_meta_graph([tag_constants.SERVING], strip_default_attrs=True)
        #builder.save()
        #sess = tf.Session()
        #tfModel = TFSingleDest()
        #tf.train.write_graph(sess.graph,'logs','graphRunModel.pbtxt',as_text=True)
        #tf.train.write_graph(sess.graph,'logs','graphRunModel.pb',as_text=False)
        with tf.Session(graph=tf.Graph()) as sess:
            tfModel = TFSingleDest()
            saver = tf.train.Saver(tf.global_variables())
            #saver = tf.train.Saver({"tfRunModel": tfModel.tfRunModel})
            tfModel.tfBeta.initializer.run()
            Tij = sess.run(tfModel.tfRunModel, {tfModel.tfTij: TObs1, tfModel.tfCij: Cij1, tfModel.tfBeta: 1.0})
            saver.save(sess, "logs/graphRunModel2.pb")

        #from the model optimizer docs:
        with tf.Session() as sess:
            tfModel = TFSingleDest()
            sess.run(tf.global_variables_initializer())
            frozen = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["result"])
            graph_io.write_graph(frozen, './', 'inference_graph.pb', as_text=False)
        #python import_pb_to_tensorboard.py --model_dir="inference_graph.pb" --log_dir="logs"


    ###############################################################################

    def run(self):
        #mvnc.SetGlobalOption(mvnc.GlobalOption.LOGLEVEL, 2)
        devices = mvnc.EnumerateDevices()
        if len(devices) == 0:
            print('No devices found')
            quit()
            
        device = mvnc.Device(devices[0])
        device.OpenDevice()

        #Load graph
        with open('logs/graphRunModel.pb', mode='rb') as f:
            graphfile = f.read()

        graph = device.AllocateGraph(graphfile)

        print('Start download to NCS...')
        #graph.LoadTensor(img.astype(numpy.float16), 'user object')
        #output, userobj = graph.GetResult()

        #graph.DeallocateGraph()
        device.CloseDevice()
        print('Finished')


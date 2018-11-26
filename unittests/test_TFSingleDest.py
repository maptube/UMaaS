"""
This is a test for working functionality of TFSingleDest.
The TensorFlow version is compared against the regular python version to verify that CBar, Oi, Dj etc
are all equal when calculated using the different platforms.
SingleDest.py is taken as the gold standard.
"""

import os.path
import math

from globals import *
from utils import loadMatrix
from models.SingleDest import SingleDest
from models.TFSingleDest import TFSingleDest

#define epsilon difference limit for assertion tests
epsilon = 0.1

###############################################################################

def assertEqualFloatsMsg(val1,val2,msg):
    diff = abs(val1-val2)
    status = 'FAILED'
    if diff<epsilon:
        status='OK'
    text = msg.format(status=status,val1=val1,val2=val2,diff=diff)
    return text

###############################################################################

def assertEqualVectorsMsg(v1,v2,msg):
    #lengths must be equal and every v1 value must be equal to its corresponding v2 value withing epsilon
    if len(v1)!=len(v2):
        return msg.format(status="FAILED LENGTH")
    for i in range(0,len(v1)):
        diff = abs(v1[i]-v2[i])
        if (diff>=epsilon):
            text = msg.format(status='FAILED i='+str(i),val1=v1[i],val2=v2[i],diff=diff)
            return text
            ###########
    text = msg.format(status='OK',val1=v1[0],val2=v2[0],diff=abs(v1[0]-v2[0])) #what should you return here? OK should be enough
    return text

###############################################################################

def assertEqualMatricesMsg(mat1,mat2,msg):
    #dimensions must be equal and every v1 value must be equal to its corresponding v2 value withing epsilon
    (m1,n1) = mat1.size()
    (m2,n2) = mat2.size()
    if len(m1)!=len(m2) or (n1)!=(n2):
        return msg.format(status="FAILED DIMENSIONS")
    for i in range(0,n1):
        for j in range(0,n1):
            diff = abs(mat1[i,j]-mat2[i,j])
            if (diff>=epsilon):
                text = msg.format(status='FAILED i='+str(i)+' j='+str(j),val1=mat1[i,j],val2=mat2[i,j],diff=diff)
                return text
                ###########
    text = msg.format(status='OK',val1=mat1[0,0],val2=mat2[0],diff=abs(mat1[0]-mat2[0])) #what should you return here? OK should be enough
    return text

###############################################################################

def testTFSingleDest():
    #TensorFlow tests - load testing matrices
    TObs1 = loadMatrix(os.path.join(modelRunsDir,TObs31Filename))
    TObs2 = loadMatrix(os.path.join(modelRunsDir,TObs32Filename))
    TObs3 = loadMatrix(os.path.join(modelRunsDir,TObs33Filename))
    Cij1 = loadMatrix(os.path.join(modelRunsDir,CijRoadMinFilename))
    Cij2 = loadMatrix(os.path.join(modelRunsDir,CijBusMinFilename))
    Cij3 = loadMatrix(os.path.join(modelRunsDir,CijRailMinFilename))
    TObs = [TObs1,TObs2,TObs3]
    Cij = [Cij1,Cij2,Cij3]
    
    #now set up the two models for comparison
    testModel = SingleDest()
    testTFModel = TFSingleDest()
    
    #CBar Test
    CBar = testModel.calculateCBar(TObs1,Cij1)
    TFCBar = testTFModel.calculateCBar(TObs1,Cij1)
    print(assertEqualFloatsMsg(CBar,TFCBar,'{status} CBar test: CBar={val1} TFCBar={val2} diff={diff}'))

    #Oi Test
    Oi = testModel.calculateOi(TObs1)
    TFOi = testTFModel.calculateOi(TObs1)
    print(assertEqualVectorsMsg(Oi,TFOi,'{status} Oi test: Oi={val1} TFOi={val2} diff={diff}'))

    #Dj Test
    Dj = testModel.calculateDj(TObs1)
    TFDj = testTFModel.calculateDj(TObs1)
    print(assertEqualVectorsMsg(Dj,TFDj,'{status} Dj test: Dj={val1} TFDj={val2} diff={diff}'))

    #Calibrate test - this gives us three predicted matrices and three beta values
    testModel.TObs=TObs
    testModel.Cij=Cij
    testModel.isUsingConstraints=False
    testModel.run()
    TPred=testModel.TPred
    Beta=testModel.Beta
    #
    testTFModel.TObs=TObs
    testTFModel.Cij=Cij
    testTFModel.isUsingConstraints=False
    testTFModel.run()
    TFTPred=testTFModel.TPred
    TFBeta=testTFModel.Beta
    #and compare them...
    print(assertEqualFloatsMsg(Beta[0],TFBeta[0],'{status} Beta test: Beta0={val1} TFBeta0={val2} diff={diff}'))
    print(assertEqualFloatsMsg(Beta[1],TFBeta[1],'{status} Beta test: Beta1={val1} TFBeta1={val2} diff={diff}'))
    print(assertEqualFloatsMsg(Beta[2],TFBeta[2],'{status} Beta test: Beta2={val1} TFBeta2={val2} diff={diff}'))
    print(assertEqualMatricesMsg(TPred,TFTPred,'{status} TPred test: TPred={val1} TFTPred={val2} diff={diff}'))



###############################################################################
#
#if __name__ == '__main__':
#    main()
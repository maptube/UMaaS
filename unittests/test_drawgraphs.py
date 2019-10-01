"""
Unit test library for plotting the various graphs that we need to determine whether
the data behind the model is correct.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import exp

################################################################################

def plotLogCijLogTij(Tij,Cij):
    plt.figure(figsize=(10,8)) #what are these? inches!
    plt.title("Trip Counts vs Trip Times")
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("log(Cij) - minutes")
    plt.ylabel("log(Tij) - trip counts")
    plt.xlim(1, 1000.0) #don't use 0 for the lower limit - log(0) obviously
    plt.ylim(1, 10000.0)

    x = []
    y = []
    N = len(Tij)
    #N=500
    print("test_drawgraphs.py::plotLogCijLogTij n=",len(Tij))
    for i in range(0,N):
        for j in range(0,N):
            t = Tij[i,j]
            c = Cij[i,j]
            if t>0: #cut the non-value data out - you'll be waiting ages in Python otherwise
                x.append(c)
                y.append(t)
    plt.scatter(x,y,label="trips",color="black",marker="+",s=8)
    plt.grid(True)
    plt.savefig('plotLogCijLogTij.png')
    plt.show()


################################################################################

#NOTE: these were pinched from KerasGravityANN.py
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

def plotOiDj(Tij):
    plt.figure(figsize=(10,8)) #what are these? inches!
    plt.title("Origin Totals vs Destination Totals")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel("Oi - origin trip totals")
    plt.ylabel("Dj - destination trip totals")
    #plt.xlim(1, 1000.0)
    #plt.ylim(1, 10000.0)

    x = []
    y = []
    N = len(Tij)
    #N=500
    print("test_drawgraphs.py::plotOiDj n=",len(Tij))
    Oi = calculateOi(Tij)
    Dj = calculateDj(Tij)
    for i in range(0,N):
        for j in range(0,N):
            if Tij[i,j]>0:
                x.append(Oi[i])
                y.append(Dj[j])
    plt.scatter(x,y,label="OiDj",color="black",marker="+",s=8)
    plt.grid(True)
    plt.savefig('plotOiDj.png')
    plt.show()

###############################################################################

def plotCijHistogram(Cij):
    plt.figure(figsize=(10,8)) #what are these? inches!
    plt.title("Cij Trip Time Histogram")
    #plt.xscale('log')
    #plt.yscale('log')
    plt.xlabel("Cij trip time between zone pairs (minutes)")
    plt.ylabel("Count")
    plt.xlim(0, 600.0)
    #plt.ylim(1, 10000.0)

    #x = []
    #N = len(Cij)
    #N = 500
    #for i in range(0,N):
    #    for j in range(0,N):
    #        x.append(Cij[i,j])

    x = Cij.flatten()
    #plt.hist(x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None, *, data=None, **kwargs)
    plt.hist(x, bins=50)
    plt.grid(True)
    plt.savefig('plotCijHistogram.png')
    plt.show()

###############################################################################

def plotTijHistogram(Tij):
    plt.figure(figsize=(10,8)) #what are these? inches!
    plt.title("Tij Trip Numbers Histogram")
    #plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Number of trips (Tij) between zone pairs")
    plt.ylabel("Count of Tij value on x axis")
    plt.xlim(0, 2300.0)
    #plt.ylim(1, 10000.0)

    x = []
    N = len(Tij)
    #N = 500
    countzero=0
    maxTij=0
    sumTij=0
    for i in range(0,N):
        for j in range(0,N):
            if Tij[i,j]>0:
                x.append(Tij[i,j])
                sumTij+=Tij[i,j]
                if Tij[i,j]>maxTij:
                    maxTij=Tij[i,j]
            else:
                countzero+=1

    matrixFillPCT = (7201*7201-countzero)/(7201*7201)*100.0
    print("test_drawgraphs.py::plotTijHistogram countzero=",countzero, "matrixFillPCT=", matrixFillPCT, "maxTij=",maxTij," sumTij=",sumTij)

    #x = Tij.flatten() #can't use everything like the Cij, as the number of zeros in the data swamps everything else
    #plt.hist(x, bins=None, range=None, density=None, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, normed=None, *, data=None, **kwargs)
    plt.hist(x, bins=100)
    plt.grid(True)
    plt.savefig('plotTijHistogram.png')
    plt.show()

###############################################################################

def plotTijTail(Tij):
    plt.figure(figsize=(10,8)) #what are these? inches!
    plt.title("Tij Trip Numbers")
    #plt.xscale('log')
    #plt.yscale('log') #NOTE: you can plot a log y scale here, but it looks odd with the integer trips
    plt.xlabel("Index of all zone pairs in trips matrix Tij containing Tij>0 (sorted by Tij value)")
    plt.ylabel("Number of trips (Tij) between zone pairs")
    #plt.xlim(0, 2300.0)
    #plt.ylim(1, 10000.0)

    x = []
    y = []
    N = len(Tij)
    #N = 500
    count=0
    for i in range(0,N):
        for j in range(0,N):
            if Tij[i,j]>0:
                x.append(count)
                count+=1
                y.append(Tij[i,j])

    #sort y here!
    y.sort() #this might take some time

    plt.scatter(x,y,label="OiDj",color="black",marker="+",s=8)
    plt.grid(True)
    plt.savefig('plotTijTail.png')
    plt.show()

###############################################################################

def plotRegression(Tij,Cij):
    plt.figure(figsize=(10,8)) #what are these? inches!
    plt.title("Tij Cij Regression")
    #plt.xscale('log')
    #plt.yscale('log') #NOTE: you can plot a log y scale here, but it looks odd with the integer trips
    plt.xlabel("x")
    plt.ylabel("y")
    #plt.xlim(0, 2300.0)
    #plt.ylim(1, 10000.0)

    x = []
    y = []
    beta = 0.137 #from quant
    N = len(Tij)
    #N = 500
    Oi = calculateOi(Tij)
    Dj = calculateDj(Tij)
    count=0
    for i in range(0,N):
        Ai=0.0
        for j in range(0,N):
            Ai+=Dj[j]*exp(Cij[i,j])
        Ai=1/Ai
        for j in range(0,N):
            if Tij[i,j]>0:
                x.append(Ai*Oi[i]*Dj[j]*exp(Cij[i,j]))
                y.append(Tij[i,j])
                count+=1

    plt.scatter(x,y,label="OiDj",color="black",marker="+",s=8)
    plt.grid(True)
    plt.savefig('plotRegression.png')
    plt.show()

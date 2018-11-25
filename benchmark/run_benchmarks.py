"""
Run benchmark code
"""

from globals import *
from benchmark.benchmark_SingleDest import *

###############################################################################

def runBenchmarks():
    print("SingleDest tests")
    #cbar = benchmark_calculateCBar_slow()
    #cbar2 = benchmark_calculateCBar_fast()
    #assert cbar==cbar2?
    ms = timeit.timeit(setup='pass',stmt=benchmark_calculateCBar_slow,number=1)
    print(ms)
    ms = timeit.timeit(setup='pass',stmt=benchmark_calculateCBar_fast,number=1)
    print(ms)

#now matrix tests

###############################################################################

#Tensorflow CPU

###############################################################################

#Tensorflow GPU

###############################################################################

#Movidius???

###############################################################################

#vary N??

###############################################################################

#NN implementations?
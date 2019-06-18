#!/usr/bin/python
#data/rail/atoc_gbrail_20160506_gtfs.zip

#Numba
#networkx
#graph-tool

from numba import jit
import numpy as np
import networkx as nx


################################################################################

def main():
  print("Hello World!")


x = np.arange(100).reshape(10, 10)

@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def go_fast(a): # Function is compiled to machine code when called the first time
    trace = 0
    for i in range(a.shape[0]):   # Numba likes loops
        trace += np.tanh(a[i, i]) # Numba likes NumPy functions
    return a + trace              # Numba likes NumPy broadcasting



################################################################################

if __name__== "__main__":
  main()
  print(go_fast(x))
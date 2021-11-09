import numpy as np
import csv
import extrapolation as extrap
import sys
import matplotlib.pyplot as plt
from scipy.special import erf

A = -np.pi           # random offset lower bound
B =  np.pi           # random offset higher bound
SEED = 112358      # numpy.random seed for reproducibility

def random_jump_location():
    return np.random.rand(A,B), np.random.rand(A,B)

def initial_condition(x,y):
    return np.sin(x)*np.cos(y)

def exact_avg(x,y):
    return ( cos(x-B) - cos(x-A) )*( sin(y-A) - sin(y-B) )/(B-A)/(B-A)


M = 256
N = 128
x,y = np.mgrid[0:2*pi:N, 0:2*pi:N]
average = np.zeros_like(x)
for m in range(M):
    X,Y = random_jump_location()
    U = initial_condition(x-X,y-Y)
    average += U
average /= M
plt.subplot(311)
plt.contourf(x,y,average)
plt.colorbar()
plt.subplot(312)
plt.contourf(x,y,exact_avg(x,y))
plt.colorbar()
plt.subplot(313)
plt.contourf(x,y,average- exact_avg(x,y))
plt.colorbar()
plt.show()
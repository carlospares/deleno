import numpy as np
import csv
import sys
import matplotlib.pyplot as plt
from scipy.special import erf

A = -np.pi           # random offset lower bound
B =  np.pi           # random offset higher bound
SEED = 112358      # numpy.random seed for reproducibility

def unif(a,b):
    return a + np.random.rand()*(b-a)

def random_jump_location():
    return unif(A,B),unif(A,B)

def initial_condition(x,y):
    # return np.sin(x)*np.cos(y)
    return -np.cos(x)*np.sin(y)

def exact_avg(x,y):
    # return ( np.cos(x-B) - np.cos(x-A) )*( np.sin(y-A) - np.sin(y-B) )/(B-A)/(B-A)
    return -( np.cos(y-B) - np.cos(y-A) )*( np.sin(x-A) - np.sin(x-B) )/(B-A)/(B-A)

def exact_avg_square(x,y):
    # return (np.sin(2*(x-B))-np.sin(2*(x-A)) +2*(B-A) )*\
    #         (np.sin(2*(y-A))-np.sin(2*(y-B)) +2*(B-A))/(B-A)/(B-A)/16
    return (np.sin(2*(y-B))-np.sin(2*(y-A)) +2*(B-A) )*\
            (np.sin(2*(x-A))-np.sin(2*(x-B)) +2*(B-A))/(B-A)/(B-A)/16

def exact_var(x,y):
    return exact_avg_square(x,y) - np.square(exact_avg(x,y))

def make_meanvar(N,M):
    dx = 2*np.pi/N
    x,y = np.mgrid[0:2*np.pi:dx, 0:2*np.pi:dx]
    average = np.zeros((N,N))
    M2 = np.zeros((N,N))
    delta = np.zeros((N,N))
    for m in range(M):
        X,Y = random_jump_location()
        U = initial_condition(x-X,y-Y)
        delta = U - average
        average += delta/(m+1)
        delta2 = U - average
        M2 += delta*delta2
    return (average, M2/(M-1))


M = 1024
N = 64

average,var = make_meanvar(N,M)

dx = 2*np.pi/N
x,y = np.mgrid[0:2*np.pi:dx, 0:2*np.pi:dx]
plt.subplot(321)
plt.title("s mean")
plt.contourf(x,y,average)
plt.colorbar()
plt.subplot(323)
plt.title("e mean")
plt.contourf(x,y,exact_avg(x,y))
plt.colorbar()
plt.subplot(325)
plt.contourf(x,y,average- exact_avg(x,y))
plt.colorbar()

plt.subplot(322)
plt.title("s var")
plt.contourf(x,y,var)
plt.colorbar()
plt.subplot(324)
plt.title("e var")
plt.contourf(x,y,exact_var(x,y))
plt.colorbar()
plt.subplot(326)
plt.contourf(x,y,var- exact_var(x,y))
plt.colorbar()

plt.show()

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,y,average)
plt.show()

plt.close('all')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x,y,var)
plt.show()
import numpy as np
import matplotlib.pyplot as plt

def load_npy(filename):
    data = np.load(filename)
    N = int(np.sqrt(len(data[1])))
    if N*N != len(data[1]):
        raise Exception("Input data is malformed! Not of size NxN")
    data = np.reshape(data, (2,N,N))
    return data

def quick_plot(data):
    plt.contourf(data, 20)
    plt.colorbar()
    plt.show()

def quick_plot_compare(data1, data2):
    plt.subplot(311)
    plt.contourf(data1, 20)
    plt.colorbar()
    plt.subplot(312)
    plt.contourf(data2, 20)
    plt.colorbar()
    plt.subplot(313)
    plt.contourf(data1-data2, 20)
    plt.colorbar()
    plt.show()

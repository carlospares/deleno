import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import h5py

def load_data(filename):
    if filename.endswith(".npy"):
        return load_npy(filename)
    elif filename.endswith(".h5"):
        return load_h5(filename)

def load_npy(filename):
    data = np.load(filename)
    N = int(np.sqrt(len(data[1])))
    if N*N != len(data[1]):
        raise ValueError("Input data is malformed! Not of size NxN")
    data = np.reshape(data, (2,N,N))
    return data

def load_h5(filename):
    f = h5py.File(filename, 'r')
    u = np.array(f['Usmall_0'])
    # small hack to deal with an output issue of sphinx
    N = u.shape[0]
    u = u[:N, :N].T
    v = (np.array(f['Usmall_1'])[:N,:N]).T
    return np.stack((u,v))


def quick_plot(data):
    plt.contourf(data, 20)
    plt.colorbar()
    plt.show()

def quick_plot_compare(data1, data2):
    # assumes both data are the same size
    v = data1.shape[0]*data1.shape[1]
    errorL1 = np.sum(np.abs(data1-data2))/v
    print(f"Asumming [0,1]^2, difference in L1 is {errorL1}")
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

def quick_conv_plot(data, options):
    if options is not None:
        Ns = options.maxN / np.flip(2**np.arange(len(data)))
    else:
        Ns = 2**np.arange(len(data))
    plt.loglog(Ns, data)

    z = np.polyfit(np.log(Ns), np.log(data), 1)
    z = z[0].round(2)
    C = data[-1] * Ns[-1]**(-z)
    plt.loglog(Ns, [ C* N**z for N in Ns ], '--', label="N^{}".format(z))
    plt.legend()

    plt.grid(which='major', linestyle='-')
    plt.grid(which='minor', linestyle='--', linewidth='0.2')

    plt.show()

def compare_to_real(downscaled, options):
    K = len(downscaled)
    k = 1
    keep_going = True 
    gw = options.gw
    name = options.name
    N = options.maxN
    dists = deque()

    while k < K and keep_going:
        data_ds = downscaled[K-k-1][gw:-gw,gw:-gw]
        name_real = name.replace(str(N), str(int(N/ 2**k)))
        try:
            data_real = load_data(name_real)[0] # TODO: deal with v as well
            if options.verbose:
                print("compare_to_real :: Succesfully loaded {}".format(name_real))
            dist = np.sum(np.abs(data_ds - data_real)) / (options.maxN / (2**k))**2
            dists.appendleft(dist)
            k = k+1
        except IOError: # file does not exist
            if options.verbose:
                print("compare_to_real :: File {} not found...".format(name_real))
            keep_going = False
    return dists



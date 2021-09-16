import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import h5py
import vtk
from vtk.util.numpy_support import vtk_to_numpy
import extrapolation
from grid import Grid

def load_data(filename):
    if filename.endswith(".npy"):
        return load_npy(filename)
    elif filename.endswith(".h5"):
        return load_h5(filename)
    elif filename.endswith(".vts") or filename.endswith(".vtk"):
        return load_vtk(filename)
    else:
        raise IOError("Format of file {} not recognized!".format(filename))

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

def load_vtk(filename):
    reader = vtk.vtkXMLStructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    
    shape = reader.GetOutput().GetDimensions()
    if len(shape) >= 3 and shape[2] == 1:
        shape = shape[:-1]

    Ux = reader.GetOutput().GetPointData().GetArray("x")
    Uy = reader.GetOutput().GetPointData().GetArray("y")
    
    if Ux is None:
        # silently try to work around the luqness output filename issue:
        # print("x array is None. Trying again with name Vec_0xbe3060_0{x,y}")
        Ux = reader.GetOutput().GetPointData().GetArray("Vec_0xbe3060_0x")
        Uy = reader.GetOutput().GetPointData().GetArray("Vec_0xbe3060_0y")
    if Ux is None:
        raise IOError("vts file has unexpected format and can't be parsed")
    else:
        pass # with either name, data has been recovered
        #print("Using Vec_0xbe3060_0{x,y} worked :)")

    data = np.array([vtk_to_numpy(Ux), vtk_to_numpy(Uy)])
    N = int(np.sqrt(len(data[0])))
    return data.reshape(2, N, N)

def save_plot(name, options=None):
    if options is not None:
        prefix = options.prefix + "_"
    else:
        prefix = ""
    plt.savefig("figs/" + prefix + name)
    plt.close()


def quick_plot(data):
    plt.contourf(data, 20)
    plt.colorbar()

def quick_L1_diff(data1, data2):
    v = data1.shape[0]*data1.shape[1]
    errorL1 = np.sum(np.abs(data1-data2))/v
    normd2 = np.sum(np.abs(data2))/v
    return errorL1/normd2

def quick_plot_compare(data1, data2):
    # assumes both data are the same size
    v = data1.shape[0]*data1.shape[1]
    errorL1 = np.sum(np.abs(data1-data2))/v
    normd2 = np.sum(np.abs(data2))/v
    print(f"Asumming [0,1]^2, difference in L1 is {errorL1} (i.e. {round(100*errorL1/normd2, 2)}%)")
    plt.subplot(311)
    plt.contourf(data1, 20)
    plt.colorbar()
    plt.subplot(312)
    plt.contourf(data2, 20)
    plt.colorbar()
    plt.subplot(313)
    plt.contourf(data1-data2, 20)
    plt.colorbar()

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


def extrapolate_and_compare(options):
    minN = options.minN
    maxN = options.maxN
    gw = options.gw
    if str(maxN) not in options.name:
        raise Exception(f"Ground truth file {options.name} does not contain options.maxN={maxN}")
    name_coarse = options.name.replace(str(maxN), str(minN))
    name_mid    = options.name.replace(str(maxN), str(2*minN))
    name_fine   = options.name.replace(str(maxN), str(4*minN))
    name_truth  = options.name

    coarse = load_data(name_coarse)    
    mid    = load_data(name_mid)
    fine   = load_data(name_fine)
    truth  = load_data(name_truth)

    refs = int(np.log2(maxN/(4*minN)))

    op = extrapolation.choose_algorithm(options.extrap_which) #choose an extrapolation operator
    extrapolated = op(coarse, mid, fine, order=options.order, refinements=refs)

    # do some analysis of the data here:
    upscale, nghosts = extrapolation.get_upscale_nghosts(options.order)
    fine_upscale = extrapolation.iterative_upscale(upscale, fine, refs, nghosts)

    for comp in range(1):#truth.shape[0]):
        # print("Comparing extrapolated data to ground truth...")
        extrap_error = quick_L1_diff(extrapolated[comp], truth[comp])
        # print("Comparing finest data in extrapolation to ground truth... (error should be bigger!)")
        real_error = quick_L1_diff(fine_upscale[comp], truth[comp])
        if extrap_error < real_error:
            print(f"Component {comp}: success! {extrap_error} < {real_error}")
        else:
            print(f"Component {comp}: well this was pointless. {extrap_error} > {real_error}")
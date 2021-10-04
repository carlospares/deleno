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
    N = int(np.sqrt(len(data[0])))
    if N*N != len(data[0]):
        raise ValueError("Input data is malformed! Not of size NxN")
    data = np.reshape(data, (data.shape[0],N,N))

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
    v = data1.shape[-1]*data1.shape[-2]
    errorL1 = np.sum(np.abs(data1-data2))/v
    normd2 = np.sum(np.abs(data2))/v
    return errorL1/normd2

def norm_L2(data):
    """ compute L2 norm of an array, assumed equispaced in [0,1]^2.
    NOTE: if data is a 3d array of ncomp x nx x ny, compute norm of the full array
    this may or may not be what you want to do, be careful! """
    v = 1. / data.shape[-1] / data.shape[-2]
    return np.sqrt(np.sum( np.square(data)) * v) 

def quick_L2_diff(data1, data2):
    errorL2 = norm_L2(data1-data2)
    normd2 = norm_L2(data2)
    return errorL2/normd2


def quick_plot_compare(data1, data2):
    # assumes both data are the same size
    errorL2 = norm_L2(data1-data2)
    normd2 = norm_L2(data2)
    print(f"Asumming [0,1]^2, difference in L2 is {errorL2} (i.e. {round(100*errorL2/normd2, 2)}%);"
                    f" ground truth norm is {normd2}")
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
        extrap_error = quick_L1_diff(extrapolated[comp], truth[comp])
        real_error = quick_L1_diff(fine_upscale[comp], truth[comp])
        if extrap_error < real_error:
            print(f"Component {comp}: success! {extrap_error} < {real_error}")
        else:
            print(f"Component {comp}: well this was pointless. {extrap_error} > {real_error}")

def compare_upscale(options):
    minN = options.minN
    maxN = options.maxN
    nrefs = int(np.log2(maxN/minN))
    ps = [0,3,5]

    if options.groundtruth is None:
        ground_truth = load_data(options.name)
        suffix_gt = ""
        tag_gt = ""
    else:
        print(f"Loading expressly designated ground truth {options.groundtruth}")
        ground_truth = load_data(options.groundtruth)
        suffix_gt = "" if options.groundtruth is None else "_gt2048" # TODO generalize if needed
        tag_gt = " (gt 2048)"

    samples = []
    Ns = [minN*(2**k) for k in range(nrefs)]
    for k in range(nrefs):
        try:
            name = options.name.replace(str(maxN), str(int(minN*(2**k))))
            samples.append(load_data(name))
        except FileNotFoundError:
            """ Files generated with the 1024-at-all-resolutions dataset have names where the number
                of samples obviously is always 1024. Make a guess that this is the case and try
                again, but make sure to warn the user!"""
            name = options.name.replace(str(maxN), str(int(minN*(2**k))), 1)
            print(f"Original file failed. Trying {name} instead. Is this what you want?")
            samples.append(load_data(name))

    dists = np.zeros((len(ps), nrefs))
    upscales = [[] for p in ps]
    for n, p in enumerate(ps):
        if options.verbose:
            print(f"Upscaling with order p={p}...")
        upscale, nghosts = extrapolation.get_upscale_nghosts(p)
        for k in range(nrefs):
            upscaled = extrapolation.iterative_upscale(upscale, samples[k], nrefs-k, nghosts)
            upscales[n].append(upscaled)
            dists[n, k] = norm_L2(upscaled - ground_truth)
            if options.extraplots:
                for comp in range(samples[-1].shape[0]):
                    quick_plot_compare(upscaled[comp], ground_truth[comp])
                    plt.savefig(f"{options.prefix}_N{minN*(2**k)}_p{p}_comp{comp}{suffix_gt}")
    plt.clf()
    # plot absolute errors
    for n, p in enumerate(ps):
        label = f"ENO{p}" if p > 0 else "Trivial"
        plt.loglog(Ns, dists[n,:], '*-', base=2, label=label)
    plt.legend()
    plt.grid("minor")
    plt.title(f"Abs. error: {options.comment}{tag_gt}")
    plt.savefig(f"{options.prefix}_upscale{suffix_gt}")

    plt.clf()
    # plot relative errors
    print(norm_L2(ground_truth))
    for n, p in enumerate(ps):
        label = f"ENO{p}" if p > 0 else "Trivial"
        plt.loglog(Ns, 100*dists[n,:]/norm_L2(ground_truth), '*-', base=10, label=label)
    plt.legend()
    plt.grid("minor")
    plt.title(f"Rel. error: {options.comment}{tag_gt}")
    plt.ylabel("% of ground truth")
    plt.savefig(f"{options.prefix}_upscale_rel{suffix_gt}")

    # # test differences to trivial
    # # this is (as expected) absolute nonsense
    # plt.clf()
    # dist_to_triv = np.zeros((len(ps), nrefs))
    # for n in range(1, len(ps)):
    #     if ps[0] != 0:
    #         print(f"ps[0] = {ps[0]}. We assume here ps[0]=0, output labels won't be accurate.")
    #     p = ps[n]
    #     for k in range(nrefs):
    #         dist_to_triv[n, k] = norm_L2(upscales[n][k] - upscales[0][k])
    #     label = f"ENO{p}"
    #     plt.loglog(Ns, 100*dist_to_triv[n,:], '*-', base=2, label=label)
    # plt.legend()
    # plt.grid("minor")
    # plt.title(f"Distance to trivial upscaling: {options.comment}")
    # plt.ylabel("L2 dist")
    # plt.savefig(f"{options.prefix}_disttriv")



def check_regularity(options):
    minN = options.minN
    maxN = options.maxN
    nrefs = int(np.log2(maxN/minN))
    colors = ['b', 'r', 'g']

    samples = []
    Ns = [64, 128, 256, 512, 1024]
    for N in Ns:
        name = options.name.replace(str(maxN), str(N))
        samples.append(load_data(name))

    ncomps = samples[-1].shape[0]
    maxNormgrad = 0

    for comp in range(ncomps):
        for k, N in enumerate(Ns):
            dx = 1./N
            data = samples[k][comp]
            gradx = (np.roll(data, 1, 0) - np.roll(data, -1, 0))/(2*dx)
            grady = (np.roll(data, 1, 1) - np.roll(data, -1, 1))/(2*dx)
            normgrad = np.sqrt(gradx**2 + grady**2)
            normnormgrad = norm_L2(normgrad)
            maxNormgrad = max(maxNormgrad, normnormgrad)
            dot, = plt.plot(np.log2(N), normnormgrad, '*', c=colors[comp])
            if k == 0:
                dot.set_label(f"Comp. {comp}")
                lowgrad = normnormgrad
            if k == len(Ns)-1:
                highgrad = normnormgrad
        slope = np.round((highgrad-lowgrad)/4, 4)
        slopenorm = np.round(((highgrad-lowgrad)/4)/highgrad, 4)
        plt.plot([6, 10], [lowgrad, highgrad], '--', c=colors[comp], label=f"Slope: {slope} (norm. {slopenorm})")
    plt.xlabel("log2(N)")
    plt.ylabel("Norm of gradient")
    tag = "" if options.prefix is None else f"({options.prefix})"
    plt.title(f"Norm of FD gradient {tag}")
    plt.legend()
    axes = plt.gca()
    axes.set_ylim([0, 1.1*maxNormgrad])
    plt.grid()
    plt.show()
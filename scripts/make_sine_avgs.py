import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generates (possibly perturbed) sine cell avgs')
parser.add_argument('-ns', '--noisesize', type=float, default=0,
                        help='Amplitude of the white noise to add')
parser.add_argument('-nr', '--noiserate', type=float, default=0,
                        help='Rate of the decay of the white noise')
parser.add_argument('-d2', '--dim2', action='store_true', default=False,
                        help='If flag present, generate a field with two components')

f = lambda x,y,dx,dy : (-np.cos(2*np.pi*(x + dx/2)) + np.cos(2*np.pi*(x - dx/2)) ) * \
                                (-np.sin(2*np.pi*(y + dy/2)) + np.sin(2*np.pi*(y - dy/2)) ) / \
                                (dx*dy*4*np.pi*np.pi)
g = lambda x,y,dx,dy : 2*(np.sin(6*np.pi*(x + dx/2)) - np.sin(6*np.pi*(x - dx/2)) ) * \
                                (np.cos(8*np.pi*(y + dy/2)) - np.cos(8*np.pi*(y - dy/2)) ) / \
                                (dx*dy*48*np.pi*np.pi)
args = parser.parse_args()

ns = args.noisesize
nr = args.noiserate

for N in [64, 128, 256, 512, 1024]:
    dx = 1./N
    dy = dx
    x = np.arange(0, 1, dx)
    y = x
    if len(x) != N:
        raise Exception("Problem with round-off error!")

    X,Y = np.meshgrid(x + 0.5*dx, y + 0.5*dy)
    comp = f(X, Y, dx, dy).T

    ndims = 2 if args.dim2 else 1
    data = np.zeros((ndims, comp.shape[0], comp.shape[1]))
    data[0] = comp

    if args.dim2:
        data[1] = g(X, Y, dx, dy).T

    noise = ns* ((64/N)**nr )* np.random.random(data.shape)
    data = data + noise

    dimtag = "2d" if ndims==2 else ""
    outputname = f"testdata/sine{dimtag}_test_{N}"
    if ns != 0:
        outputname = outputname + f"_{str(ns).replace('.', ',')}"
    if nr != 0:
        outputname = outputname + f"_r{str(nr).replace('.', ',')}"
    plt.contourf(X,Y,data[-1])
    plt.colorbar()
    plt.savefig(outputname)
    plt.clf()
    np.save(outputname, data.reshape(ndims, N*N))

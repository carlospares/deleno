import numpy as np
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generates smooth shear layer steady state')
args = parser.parse_args()

rho = 0.05
# f = lambda x,y,dx,dy : np.tanh((y-0.25)/rho)*(y <= 0.5) + np.tanh((0.75-y)/rho)*(y > 0.5)
f = lambda x,y,dx,dy : (rho/dy)*(np.log(np.cosh((y + dy/2 - 0.25)/rho)) \
                                - np.log(np.cosh((y - dy/2 - 0.25)/rho)))*(y <= 0.5)\
                       + (rho/dy)*(-np.log(np.cosh((0.75 - y - dy/2)/rho)) \
                                + np.log(np.cosh((0.75 - y + dy/2)/rho)))*(y > 0.5)
g = lambda x,y,dx,dy : np.zeros_like(x)

for N in [64, 128, 256, 512, 1024]:
    dx = 1./N
    dy = dx
    x = np.arange(0, 1, dx)
    y = x
    if len(x) != N:
        raise Exception("Problem with round-off error!")

    X,Y = np.meshgrid(x + 0.5*dx, y + 0.5*dy)
    comp = f(X, Y, dx, dy)

    ndims = 2
    data = np.zeros((ndims, comp.shape[0], comp.shape[1]))
    data[0] = comp
    data[1] = g(X, Y, dx, dy)

    outputname = f"testdata/slsmooth_test_{N}"
    plt.contourf(X,Y,data[0],50)
    plt.colorbar()
    plt.savefig(outputname)
    plt.clf()
    np.save(outputname, data.reshape(ndims, N*N))

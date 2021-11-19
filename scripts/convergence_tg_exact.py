import numpy as np
import matplotlib.pyplot as plt
import sys


# This assumes that we are randomly perturbing 
# (u_0, v_0) = (sin(x)cos(y), -cos(x)sin(y))
# by a random translation in [A, B]^2

A = -np.pi
B = np.pi

def exact_avg(x,y):
    avg = np.zeros((2, x.shape[0], x.shape[1]))
    avg[0] = ( np.cos(x-B) - np.cos(x-A) )*( np.sin(y-A) - np.sin(y-B) )/(B-A)/(B-A)
    avg[1] = -( np.cos(y-B) - np.cos(y-A) )*( np.sin(x-A) - np.sin(x-B) )/(B-A)/(B-A)
    return avg

def exact_avg_square(x,y):
    avg = np.zeros((2, x.shape[0], x.shape[1]))
    avg[0] =  (np.sin(2*(x-B))-np.sin(2*(x-A)) +2*(B-A) )*\
            (np.sin(2*(y-A))-np.sin(2*(y-B)) +2*(B-A))/(B-A)/(B-A)/16
    avg[1] = (np.sin(2*(y-B))-np.sin(2*(y-A)) +2*(B-A) )*\
            (np.sin(2*(x-A))-np.sin(2*(x-B)) +2*(B-A))/(B-A)/(B-A)/16
    return avg

def exact_var(x,y):
    return exact_avg_square(x,y) - np.square(exact_avg(x,y))

def norm_L2(data):
    v = 2*np.pi / data.shape[-1] / data.shape[-2]
    return np.sqrt(np.sum( np.square(data)) * v)

def sanity_check(U, Uref, name):
    plt.subplot(321)
    plt.title(name)
    plt.contourf(U[0])
    plt.colorbar()
    plt.subplot(322)
    plt.contourf(U[1])
    plt.colorbar()
    plt.subplot(323)
    plt.contourf(Uref[0])
    plt.colorbar()
    plt.subplot(324)
    plt.contourf(Uref[1])
    plt.colorbar()
    plt.subplot(325)
    plt.contourf(Uref[0] - U[0])
    plt.colorbar()
    plt.subplot(326)
    plt.contourf(Uref[1] - U[1])
    plt.colorbar()
    plt.show()


def plot_convergence_to_exact_tg(lowest_name, lowest_res, highest_res):
    lowest_name_mean = lowest_name
    lowest_name_var = lowest_name.replace("mean", "var")
    n = lowest_res
    Ns = []
    error_mean = []
    error_var = []
    while n <= highest_res:
        mean = np.load(lowest_name_mean.replace(str(lowest_res), str(n)))
        nx = int(np.sqrt(mean.shape[-1]))
        mean = mean.reshape(2,nx,nx)
        var  = np.load(lowest_name_var.replace(str(lowest_res), str(n))).reshape(2,nx,nx)

        dx = 2*np.pi/nx;
        x = np.arange(0.5*dx, 2*np.pi, dx)
        X,Y = np.meshgrid(x,x)
        mean_exact = exact_avg(X,Y)
        var_exact = exact_var(X,Y)

        # sanity_check(mean, mean_exact, f"mean {n}")
        # sanity_check(var, var_exact, f"var {n}")

        Ns.append(n)
        error_mean.append(norm_L2(mean_exact - mean))
        error_var.append(norm_L2(var_exact - var))

        n *= 2
    return (Ns, error_mean, error_var)

N, err_mean, err_var = plot_convergence_to_exact_tg(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))

Np = 2*int(sys.argv[3])
dx = 2*np.pi/Np;
x = np.arange(0.5*dx, 2*np.pi, dx)
X,Y = np.meshgrid(x,x)
mean_exact = exact_avg(X,Y)
var_exact = exact_var(X,Y)
nmex = norm_L2(mean_exact)
nvex = norm_L2(var_exact)

# plt.subplot(211)
plt.title("Abs. L2 distance to exact solution")
plt.loglog(N, err_mean, 'b*-', label="mean", base=2)
plt.loglog(N, err_var, 'r*-', label="var", base=2)
plt.legend()
plt.grid()
# plt.subplot(212)
# plt.title("Rel. L2 distance to exact solution")
# plt.loglog(N, 100*np.array(err_mean)/nmex, 'b*-', label="mean", base=10)
# plt.loglog(N, 100*np.array(err_var)/nvex, 'r*-', label="var", base=10)
# plt.legend()
# plt.grid()


plt.show()





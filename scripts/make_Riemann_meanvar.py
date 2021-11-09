

import numpy as np
import csv
import extrapolation as extrap
import sys
import matplotlib.pyplot as plt
from scipy.special import erf

A = -2.5           # interval left
B = 2.5            # interval right
UL = 2             # Riemann problem left state
UR = 1             # Riemann problem right state
SEED = 112358      # numpy.random seed for reproducibility
UPSCALE_ORDER = 3 # if >= 0, ground truth is exact at higher resolution, upscale with ENO of order.
                   # if <0, ground truth is exact at same resolution. Can be overridden from CL

JUMP_DISTRIB = "uniform"
if JUMP_DISTRIB == "uniform":
    JUMP_OFFSET = -0.5 # unperturbed jump location
    MAG = 1            # jump distributed as JUMP_OFFSET + MAG*U[0,1]
elif JUMP_DISTRIB == "normal":
    JUMP_MEAN = 0
    JUMP_VAR = 0.1
PEEK_PLOTS = False
tag = f"_{JUMP_DISTRIB}"



# options for one-off tests
IS_REF_4096 = False

if len(sys.argv) > 1:
    UPSCALE_ORDER = int(sys.argv[1])

if UPSCALE_ORDER >= 0:
    OUTFILE = f"meanvar_Riemann_test_interp{UPSCALE_ORDER}{tag}.csv"
else:
    OUTFILE = f"meanvar_Riemann_test{tag}.csv"

if IS_REF_4096:
    OUTFILE = f"r4096at4096_{OUTFILE}"

def random_jump_location():
    if JUMP_DISTRIB == "uniform":
        return JUMP_OFFSET + MAG*np.random.rand()
    elif JUMP_DISTRIB == "normal":
        return np.random.normal(JUMP_MEAN, np.sqrt(JUMP_VAR))
    elif JUMP_DISTRIB == "atomic":
        r = np.random.rand()
        if r >= 0.5:
            return 0
        else:
            return 4*r - 1
    else:
        raise ValueError(f"Distribution {JUMP_DISTRIB} not known to random_jump_location")

def make_init_state(x, cellavgs, jump):
    for i in range(len(cellavgs)):
        xL = x[i]
        xR = x[i+1]
        dx = xR - xL
        if xR <= jump:
            cellavgs[i] = UL
        elif xL >= jump:
            cellavgs[i] = UR
        else:
            cellavgs[i] = ((jump-xL)*UL + (xR-jump)*UR)/dx
    return

def exact_mean_integrated_uniform(x):
    # exact avg: uL left of start_jump, uR right of end_jump, linear in the middle.
    # and now compute exact cell averages.
    start_jump = JUMP_OFFSET
    end_jump = JUMP_OFFSET + MAG
    slope = (UR-UL)/(end_jump - start_jump)
    return (x < start_jump)*UL*(x - start_jump) \
            + (x >= start_jump)*(x <= end_jump) * \
                    (UL*(x - start_jump) + 0.5*slope*((x-start_jump)**2)) \
            + (x > end_jump)*((end_jump-start_jump)*(UL + UR)*0.5 + (x-end_jump)*UR)

def exact_var_integrated_uniform(x):
    start_jump = JUMP_OFFSET
    end_jump = JUMP_OFFSET + MAG
    C = (UR - UL)*(UR - UL)/(end_jump - start_jump)
    D = C/(end_jump - start_jump)
    # exact solution is C (x-s) - D (x-s)^2, in [start, end], else 0,
    # with s=start_jump
    return (x >= start_jump)*(x <= end_jump) * \
                            (0.5*C*((x - start_jump)**2) - D/3*((x - start_jump)**3)) \
            + (x > end_jump)*(end_jump-start_jump)*(UR-UL)*(UR-UL)/6


def exact_mean_integrated_normal(x):
    # integral of the exact mean uL + 0.5*(UR-UL)*(1 + erf(scaledx))
    scaledx = (x - JUMP_MEAN)/(np.sqrt(2*JUMP_VAR))
    return UL*x + 0.5*(UR - UL)*( (x - JUMP_MEAN)*erf(scaledx) + \
                                     np.sqrt(2*JUMP_VAR/np.pi)*np.exp(-scaledx*scaledx) + x)


def exact_var_integrated_normal(x):
    # integral of the exact variance, with variables defined as below:
    # (1 + erf(scaledx))*0.5*jsq - 0.25*jsq*(1 + erf(scaledx))*(1 + erf(scaledx))
    mu = JUMP_MEAN
    var = JUMP_VAR
    scaledx = (x - mu)/(np.sqrt(2*var))
    esx = erf(scaledx)
    jsq = (UR-UL)*(UR-UL)

    return 0.5*jsq*((x - mu)*esx + np.sqrt(2*var/np.pi)*np.exp(-scaledx*scaledx) + x) - \
            0.25*jsq*((x - mu)*esx*esx + \
                        2*(-mu + np.sqrt(2*var/np.pi)*np.exp(-scaledx*scaledx)+x)*esx \
            - 2*np.sqrt(var/np.pi)*erf((x - mu)/(np.sqrt(var))) \
            + 2*np.sqrt(2*var/np.pi)*np.exp(-scaledx*scaledx)+x )

def exact_mean_integrated_atomic(x):
    return (x<-1)*UL*(x+1) + \
            (x>=-1)*(x<0)*(UL*(x+1) + (UR-UL)*(x+1)*(x+1)/8) + \
            (x>=0)*(x<1)*((7*UL+UR)/8 + UL*x + (UR-UL)/8*( (x+3)*(x+3) - 9 )) + \
            (x>=1)*( (UR + UL)  + UR*(x-1))

def exact_var_integrated_atomic(x):
    jsq = (UR - UL)*(UR - UL)
    return (x>=-1)*(x<0)*( jsq*(0.125*(x+1)*(x+1)  - (1./48)*(x+1)**3 ) ) + \
            (x >= 0)*(x<1)*jsq*( 5./48 + 0.125*((x+3)*(x+3) - 9) - (1./48)*((x+3)**3 - 27 )) +\
            (x >= 1)*jsq*5./24

def exact_mean(x):
    exact_avgs = np.zeros(len(x)-1)
    if JUMP_DISTRIB == "uniform":
        indef_int = lambda x : exact_mean_integrated_uniform(x)
    elif JUMP_DISTRIB == "normal":
        indef_int = lambda x : exact_mean_integrated_normal(x)
    elif JUMP_DISTRIB == "atomic":
        indef_int = lambda x : exact_mean_integrated_atomic(x)
    for i in range(len(exact_avgs)):
        xL = x[i]
        xR = x[i+1]
        exact_avgs[i] = (indef_int(xR) - indef_int(xL))/(xR - xL)

    return exact_avgs

def exact_var(x):
    exact_vars = np.zeros(len(x)-1)
    if JUMP_DISTRIB == "uniform":
        indef_int = lambda x : exact_var_integrated_uniform(x)
    elif JUMP_DISTRIB == "normal":
        indef_int = lambda x : exact_var_integrated_normal(x)
    elif JUMP_DISTRIB == "atomic":  
        indef_int = lambda x : exact_var_integrated_atomic(x)
                            
    for i in range(len(exact_vars)):
        xL = x[i]
        xR = x[i+1]
        exact_vars[i] = (indef_int(xR) - indef_int(xL))/(xR - xL)
    return exact_vars

def norm_L2(data, dx):
    return np.sqrt(np.sum( np.square(data)) * dx) 

def make_meanvar(N,M):
    realization_fv = np.zeros(N)
    x = np.linspace(A,B,N+1)
    average = np.zeros(N)
    M2 = np.zeros(N)
    delta = np.zeros(N)
    for m in range(M):
        jump_location = random_jump_location()
        make_init_state(x, realization_fv, jump_location)
        delta = realization_fv - average
        average += delta/(m+1)
        delta2 = realization_fv - average
        M2 += delta*delta2
    return (average, M2/(M-1))

def peek_mean_var(x, xgt, mean, mean_up, gt_mean, var, var_up, gt_var, M, N):
    plt.clf()
    plt.subplot(211)
    plt.plot((x[1:]+x[:-1])/2, mean, 'b', label="sample mean")
    if UPSCALE_ORDER >= 0:
        plt.plot((xgt[1:]+xgt[:-1])/2, mean_up, 'b--', label="sample mean, upscaled")
        plt.plot((xgt[1:]+xgt[:-1])/2, gt_mean, 'r', label="exact")
    else:
        plt.plot((x[1:]+x[:-1])/2, gt_mean, 'r', label="exact")
    plt.title(f"Mean, N={N}, M={M}")
    plt.legend()

    plt.subplot(212)
    plt.plot((x[1:]+x[:-1])/2, var, 'b', label="sample var")
    if UPSCALE_ORDER >= 0:
        plt.plot((xgt[1:]+xgt[:-1])/2, var_up, 'b--', label="sample var, upscaled")
        plt.plot((xgt[1:]+xgt[:-1])/2, gt_var, 'r', label="exact")
    else:
        plt.plot((x[1:]+x[:-1])/2, gt_var, 'r', label="exact")
    plt.title(f"Variance, N={N}, M={M}")
    plt.legend()
    plt.show()

expsN = range(6,13)
expsM = range(6,15)
Ns = [2**exp for exp in expsN]
Ms = [2**exp for exp in expsM]

np.random.seed(SEED)

if UPSCALE_ORDER<0 and IS_REF_4096:
    print("What you are trying to do makes no sense!")
    sys.exit()

xgt = None # for convenience in plotting routine later
if UPSCALE_ORDER>=0:
    N_truth = 8192 if not IS_REF_4096 else 4096
    if IS_REF_4096:
        xgt = np.linspace(A, B, 4097)
        print("Preparing ground truth...")
        base_gt_mean, base_gt_var = make_meanvar(4096, 4096)
        print("Ready!")
    else:
        xgt = np.linspace(A, B, N_truth+1)
        base_gt_mean = exact_mean(xgt)
        base_gt_var = exact_var(xgt)
    upscale, nghosts = extrap.get_upscale_nghosts_1d(UPSCALE_ORDER)

nNs = len(Ns)
nMs = len(Ms)
mean_rel = np.zeros((nNs, nMs))
var_rel = np.zeros((nNs, nMs))

for l,N in enumerate(Ns):
    x = np.linspace(A,B,N+1)
    dx = x[1]-x[0]
    if UPSCALE_ORDER>=0:
        niters = int(np.log2(N_truth/N))
        gt_mean = base_gt_mean
        gt_var = base_gt_var
    else:
        gt_mean = exact_mean(x)
        gt_var = exact_var(x)

    for k,M in enumerate(Ms):
        print(f"N={N}, M={M}")
        
        mean,var = make_meanvar(N,M)

        if UPSCALE_ORDER>=0:
            mean_up = extrap.iterative_upscale_1d(upscale, mean, niters, 
                                             nghosts, bcs=['E','E','E','E'])
            var_up  = extrap.iterative_upscale_1d(upscale, var, niters, 
                                             nghosts, bcs=['E','E','E','E'])
        else:
            mean_up = mean
            var_up  = var
        mean_rel[l,k] = norm_L2(mean_up-gt_mean, dx) / norm_L2(gt_mean, dx)
        var_rel[l,k]  = norm_L2(var_up-gt_var, dx) / norm_L2(gt_var, dx)

        if PEEK_PLOTS and M == Ms[-1]:
            peek_mean_var(x, xgt, mean, mean_up, gt_mean, var, var_up, gt_var, M, N)

with open(OUTFILE, 'w') as csvfile:
    writer = csv.writer(csvfile)
    header_row = ["" for M in range(nMs+1)]
    header_row[0] = "% error wrt ground truth"
    header_row[1:] = [f"M={M}" for M in Ms]
    row = ["" for M in range(nMs+1)]

    writer.writerow(["Mean"])
    writer.writerow(header_row)
    for l, N in enumerate(Ns):
        row[0] = f"N={N}"
        for k, M in enumerate(Ms):
            row[k+1] = str(100*round(mean_rel[l, k], 5))
        writer.writerow(row)

    writer.writerow(["Variance"])
    writer.writerow(header_row)
    for l, N in enumerate(Ns):
        row[0] = f"N={N}"
        for k, M in enumerate(Ms):
            row[k+1] = str(100*round(var_rel[l, k], 5))
        writer.writerow(row)
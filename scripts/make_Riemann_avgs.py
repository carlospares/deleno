import numpy as np
import csv
import extrapolation as extrap
import sys

A = -2.5           # interval left
B = 2.5            # interval right
JUMP_OFFSET = -0.5 # unperturbed jump location
MAG = 1            # jump distributed as JUMP_OFFSET + MAG*U[0,1]
UL = 2             # Riemann problem left state
UR = 1             # Riemann problem right state
SEED = 112358      # numpy.random seed for reproducibility
UPSCALE_ORDER = -1 # if >= 0, ground truth is exact at higher resolution, upscale with ENO of order.
                   # if <0, ground truth is exact at same resolution. Can be overridden from CL

if len(sys.argv) > 1:
    UPSCALE_ORDER = int(sys.argv[1])

if UPSCALE_ORDER >= 0:
    OUTFILE = f"Riemann_test_interp{UPSCALE_ORDER}.csv"
else:
    OUTFILE = "Riemann_test.csv"

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

def exact(x):
    exact_avgs = np.zeros(len(x)-1)
    start_jump = JUMP_OFFSET
    end_jump = JUMP_OFFSET + MAG
    slope = (UR-UL)/(end_jump - start_jump)

    for i in range(len(exact_avgs)):
        xL = x[i]
        xR = x[i+1]
        dx = xR - xL
        if xR <= start_jump: # left pre-jump region
            exact_avgs[i] = UL
        elif xL >= end_jump: # right post-jump region
            exact_avgs[i] = UR
        elif xL>=start_jump and xR<=end_jump: # middle sloped region
            vL = UL + slope*(xL - start_jump)
            vR = UL + slope*(xR - start_jump)
            exact_avgs[i] = 0.5*(vL + vR) # linear fct: trapezoidal QR is exact
        elif xL<start_jump and xR<=end_jump:
            vL = UL
            vR = UL + slope*(xR - start_jump)
            intR = 0.5*(vR + vL)*(xR - start_jump)
            intL = (start_jump-xL)*UL
            exact_avgs[i] = (intR + intL)/dx
        elif xL>=start_jump and xR>end_jump:
            vL = UL + slope*(xL - start_jump)
            vR = UR
            intL = 0.5*(vR + vL)*(end_jump - xL)
            intR = (xR-end_jump)*UR
            exact_avgs[i] = (intR + intL)/dx
        elif xL<start_jump and xR>end_jump:
            raise Exception("Cells too large! Please increase spatial resolution")
        else: # have I missed a possibility? I don't think so, but just in case...
            raise Exception("Oops! There's a bug in exact")
    return exact_avgs

def norm_L2(data, dx):
    return np.sqrt(np.sum( np.square(data)) * dx) 

expsN = range(8,10)
expsM = range(6,21)
Ns = [2**exp for exp in expsN]
Ms = [2**exp for exp in expsM]

np.random.seed(SEED)

if UPSCALE_ORDER>=0:
    N_truth = 8192#2**(expsM[-1]+1)
    x = np.linspace(A, B, N_truth+1)
    base_ground_truth = exact(x)
    upscale, nghosts = extrap.get_upscale_nghosts_1d(UPSCALE_ORDER)

row = ["" for M in Ms]
with open(OUTFILE, 'w') as csvfile:
    row[0] = "% error wrt ground truth"
    writer = csv.writer(csvfile)
    row[1:] = [f"M={M}" for M in Ms]
    writer.writerow(row)

    for N in Ns:
        row[0] = f"N={N}"
        x = np.linspace(A,B,N+1)
        dx = x[1]-x[0]
        realization_fv = np.zeros(N)

        if UPSCALE_ORDER>=0:
            niters = int(np.log2(N_truth/N))
            gt = base_ground_truth
        else:
            gt = exact(x)

        for k,M in enumerate(Ms):
            print(f"N={N}, M={M}")
            average = np.zeros(N)
            for m in range(M):
                jump_location = JUMP_OFFSET + MAG*np.random.rand()
                make_init_state(x, realization_fv, jump_location)
                average += realization_fv

            average /= M
            if UPSCALE_ORDER>=0:
                avg_up = extrap.iterative_upscale_1d(upscale, average, niters, 
                                                 nghosts, bcs=['E','E','E','E'])
            else:
                avg_up = average
            row[k+1] = str(100*round(norm_L2(avg_up-gt, dx) / norm_L2(gt, dx), 5))

        writer.writerow(row)
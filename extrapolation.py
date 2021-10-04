import eno_tools as eno
from grid import Grid
import numpy as np
import scipy.optimize as spo
from tqdm import tqdm
from boundary_condition import BoundaryCondition as BC

""" Bibliography:
 [MacLeod86]
 Acceleration of vector sequences by multi-dimensional Delta^2 methods, Allan J. MacLeod. 
 Communications in Applied Numerical Methods, Vol. 2, 385-392 (1986) """

EXTR_RICHARDSON = 0
EXTR_CWISERICHARDSON = 1
EXTR_RBS = 2
EXTR_ANDERSON = 3
EXTR_AITKEN = 4

extrap_dict = {
    "Richardson": EXTR_RICHARDSON,
    "cwiseRichardson": EXTR_CWISERICHARDSON,
    "RBS": EXTR_RBS,
    "Anderson": EXTR_ANDERSON,
    "Aitken": EXTR_AITKEN
}

def choose_algorithm(extrap_which):
    # choose the appropriate extrapolation algorithm
    if extrap_which == EXTR_RICHARDSON:
        return Richardson_extrapolation_fv
    if extrap_which == EXTR_CWISERICHARDSON:
        return cwiseRichardson_extrapolation_fv
    elif extrap_which == EXTR_RBS:
        return RBS_extrapolation_fv
    elif extrap_which == EXTR_ANDERSON:
        return Anderson_extrapolation_fv
    elif extrap_which == EXTR_AITKEN:
        return Aitken_extrapolation_fv
    else:
        raise ValueError(f"Value of extrap_which {extrap_which} unknown.")


def iterative_upscale(upscale, data, niters, nghosts):
    """ iteratively apply upscaling function upscale to data, niters times.
    
    input: data of shape (ncomp, grid.nx, grid.ny) [no ghost cells!]
    output: data of shape (ncomp, 2**niters*grid.nx, 2**niters*grid.ny) [no ghost cells!]
    
    cf get_upscale_nghosts for the shape of function upscale """
    data_upscaled = data
    for i in range(niters):
        grid = Grid(data_upscaled.shape, nghosts)
        data_upscaled = upscale(grid.bcs.extend_with_bc_2d(data_upscaled, nghosts), grid)
    return data_upscaled

def iterative_upscale_1d(upscale, data, niters, nghosts, direction=BC.AXIS_NS):
    """ iteratively apply upscaling function upscale to data, niters times.
    
    input: data of shape (ncomp, grid.nx, grid.ny) [no ghost cells!]
    output: data of shape (ncomp, 2**niters*grid.nx, 2**niters*grid.ny) [no ghost cells!]
    
    cf get_upscale_nghosts_1d for the shape of function upscale """
    data_upscaled = data
    for i in range(niters):
        N = len(data_upscaled)
        grid = Grid((N,N), nghosts)
        data_upscaled = upscale(grid.bcs.extend_with_bc_1d(data_upscaled, nghosts, direction), grid)
    return data_upscaled


def norm_2(data, L2=False):
    """ compute 2-norm of a number/array/matrix as vector 2-norm, regardless of object type
    NOTE: if data is a 3d array of ncomp x nx x ny, compute norm of the full array
    this may or may not be what you want to do, be careful! """
    return np.sqrt(np.sum( np.square(data)))


def Richardson_estimate_rate(coarse, mid, fine, plot_exp=False, step=2):
    """take data at three consecutive resolutions, all with shape = fine.shape
    If we denote (nx, ny) the resolution of fine, then  this means coarse must have been 
    upscaled twice from (nx/4, ny/4), and mid upscaled once from (nx/2, ny/2).
    return a Richardson approximation to the rate of convergence
    This assumes the mesh is refined in steps of 2 of course.
    In the unlikely case you want to use something else, it can be set in step. """

    # Minimize the norm of the difference
    F = lambda k: norm_2((fine-mid) + (fine-coarse)/(step**(2*k)-1) - (mid-coarse)/(step**k-1))
    # initial guess is 1; output actual value
    k = spo.minimize(F, 1, options={'disp': False}, tol=1e-8).x[0]

    if plot_exp: # sanity check for the exponent of Richardson extrapolation
        import matplotlib.pyplot as plt
        ks = []
        Fs = []
        for ko in np.arange(0.1, 2*k, 0.1):
            ks.append(ko)
            Fs.append(F(ko))
        plt.plot(ks, Fs)
        plt.show()
    return k

def Richardson_extrapolation_step(coarse, fine, rate):
    """ take data at two consecutive resolutions, upscaled to having shape = result.shape.
    As the mesh is refined by steps of 2, and we want to upscale, then this means that
    coarse must have been upscaled at least twice, and fine at least once.
    rate is the exponent of the leading term for the error.
    
    return the Richardson extrapolation at the resolution of fine """

    return ( (2**rate)*fine - coarse) / (2**rate - 1)

def Richardson_extrapolation_step_standalone(coarse, mid, fine, rate=None):
    if rate is None:
        rate = Richardson_estimate_rate(coarse, mid, fine)
    return Richardson_extrapolation_step(mid, fine, rate)

def get_upscale_nghosts(order):
    """ return necessary information for upscaling, depending on order """
    if order == 0:
        upscale = lambda data, grid: eno.trivial_fv_2d_predictor(data, grid)
    else:
        upscale = lambda data, grid: eno.fv_2d_predictor(data, order, grid)
    nghosts = order
    return (upscale, nghosts)

def get_upscale_nghosts_1d(order):
    """ return necessary information for upscaling, depending on order """
    if order == 0:
        upscale = lambda data, grid: eno.trivial_fv_1d_predictor(data, grid)
    else:
        upscale = lambda data, grid: eno.eno_upscale_avg_1d(data, order, grid)
    nghosts = order
    return (upscale, nghosts)

def Richardson_extrapolation_fv(coarse, mid, fine, order=0, refinements=1, rate=None, **kwargs):
    """ take three vector fields at successive resolutions, with cell averages.
    If fine has shape (nx, ny), then mid has shape (nx/2, ny/2) and coarse (nx/4, ny/4).
    output Richardson extrapolation to (2**refinements)*(nx, ny).
    if rate is given, use for Richardson extrapolation; otherwise, compute.
    if order > 0, use eno upscaling of that order. If order = 0, use trivial upscaling.
    refinement is the number of doublings of resolution between fine and the output"""

    upscale, nghosts = get_upscale_nghosts(order)

    # upscale coarse three times, mid twice, fine once (or more if refinements>1)
    coarse_upscaled = iterative_upscale(upscale, coarse, 2+refinements, nghosts)
    mid_upscaled    = iterative_upscale(upscale, mid,    1+refinements, nghosts)
    fine_upscaled   = iterative_upscale(upscale, fine,     refinements, nghosts)

    output = np.zeros(fine_upscaled.shape)
    for comp in range(coarse.shape[0]):
        if rate is None:
            # estimate the rate. We do this at the resolution of fine even if the output is higher
            est_rate = Richardson_estimate_rate(iterative_upscale(upscale, coarse[np.newaxis,comp], 2, nghosts), 
                                                iterative_upscale(upscale,    mid[np.newaxis,comp], 1, nghosts), 
                                                fine[np.newaxis, comp])
            # print(f"Richardson estimate of exponent for leading error term of comp. {comp}: {est_rate}")
        else:
            est_rate = rate

        output[comp] = Richardson_extrapolation_step(mid_upscaled[np.newaxis,comp], fine_upscaled[np.newaxis,comp],
                                                     est_rate)
    return output

def RBS_extrapolation_fv(coarse, mid, fine, order=0, refinements=1, **kwargs):
    """take three vector fields at successive resolutions, with cell averages.
    If fine has shape (nx, ny), then mid has shape (nx/2, ny/2) and coarse (nx/4, ny/4).
    Output Roothan--Bagus--Sack (Method 2 in [MacLeod86]) extrapolation
    if order > 0, use eno upscaling of that order. If order = 0, use trivial upscaling.
    refinement is the number of doublings of resolution between fine and the output"""

    upscale, nghosts = get_upscale_nghosts(order)

    coarse_upscaled = iterative_upscale(upscale, coarse, 2+refinements, nghosts)
    mid_upscaled    = iterative_upscale(upscale, mid,    1+refinements, nghosts)
    fine_upscaled   = iterative_upscale(upscale, fine,     refinements, nghosts)

    output = np.zeros(fine_upscaled.shape)
    for comp in range(coarse.shape[0]):
        c_u = coarse_upscaled[comp]
        m_u = mid_upscaled[comp]
        f_u = fine_upscaled[comp]

        alpha = norm_2(f_u - m_u)**2
        gamma = norm_2(m_u - c_u)**2
        delta = norm_2(f_u - 2*m_u + c_u)**2

        output[comp] = m_u + (gamma*(f_u - m_u) - alpha*(m_u - c_u))/delta

    return output

def Anderson_extrapolation_step(coarse, mid, fine):
    r2 = fine - mid
    r1 = mid - coarse 
    den = norm_2(r2 - r1)**2
    mu = np.sum(r2 * (r2 - r1))/den 
    return fine + mu*(mid - fine)

def Anderson_extrapolation_fv(coarse, mid, fine, order=0, refinements=1, **kwargs):
    """ take three vector fields at successive resolutions, with cell averages.
    If fine has shape (nx, ny), then mid has shape (nx/2, ny/2) and coarse (nx/4, ny/4).
    Output Anderson (Method 3 in [MacLeod86]) extrapolation
    if order > 0, use eno upscaling of that order. If order = 0, use trivial upscaling.
    refinement is the number of doublings of resolution between fine and the output """

    upscale, nghosts = get_upscale_nghosts(order)

    coarse_upscaled = iterative_upscale(upscale, coarse, 2+refinements, nghosts)
    mid_upscaled    = iterative_upscale(upscale, mid,    1+refinements, nghosts)
    fine_upscaled   = iterative_upscale(upscale, fine,     refinements, nghosts)

    output = np.zeros(fine_upscaled.shape)
    for comp in range(coarse.shape[0]):
        c_u = coarse_upscaled[comp]
        m_u = mid_upscaled[comp]
        f_u = fine_upscaled[comp]

        # r2 = f_u - m_u
        # r1 = m_u - c_u
        # den = norm_2(r2 - r1)**2
        # mu = np.sum(r2 * (r2-r1))/den
        # print(mu)

        output[comp] = Anderson_extrapolation_step(c_u, m_u, f_u) #f_u + mu*(m_u - f_u)

    return output

def Aitken_extrapolation_step(coarse, mid, fine, tol=1e-9):
    d = fine - 2*mid + coarse
    # safety catch, otherwise near-constant sequences blow up
    return (abs(d)<tol)*fine + (abs(d)>=tol)*np.nan_to_num((fine - np.square(fine - mid)/d))

def Aitken_extrapolation_fv(coarse, mid, fine, order=0, refinements=1, **kwargs):
    """take three vector fields at successive resolutions, with cell averages.
    If fine has shape (nx, ny), then mid has shape (nx/2, ny/2) and coarse (nx/4, ny/4).
    Output Aitken (Method 1 in [MacLeod86]) extrapolation
    if order > 0, use eno upscaling of that order. If order = 0, use trivial upscaling.
    refinement is the number of doublings of resolution between fine and the output"""

    upscale, nghosts = get_upscale_nghosts(order)

    coarse_upscaled = iterative_upscale(upscale, coarse, 2+refinements, nghosts)
    mid_upscaled    = iterative_upscale(upscale, mid,    1+refinements, nghosts)
    fine_upscaled   = iterative_upscale(upscale, fine,     refinements, nghosts)

    output = np.zeros(fine_upscaled.shape)
    for comp in range(coarse.shape[0]):
        c_u = coarse_upscaled[comp]
        m_u = mid_upscaled[comp]
        f_u = fine_upscaled[comp]

        output[comp] = Aitken_extrapolation_step(c_u, m_u, f_u) 

    return output


def cwiseRichardson_extrapolation_step(coarse, mid, fine, rate=None, order=0, refinements=1, 
                                                **kwargs):
    try:
        s = fine.shape
    except AttributeError:
        raise ValueError("componentwise Richardson can only be used with vectors.")

    output = np.zeros(fine.shape)
    for index,_ in np.ndenumerate(fine):
        if rate is None:
            est_rate = Richardson_estimate_rate(coarse[index],mid[index],fine[index])
        else:
            est_rate = rate
        output[index] = Richardson_extrapolation_step(mid[index],fine[index],est_rate)
    return output


def cwiseRichardson_extrapolation_fv(coarse, mid, fine, rate=None, order=0, refinements=1, **kwargs):
    upscale, nghosts = get_upscale_nghosts(order)

    # upscale coarse three times, mid twice, fine once (or more if refinements>1)
    coarse_upscaled = iterative_upscale(upscale, coarse, 2+refinements, nghosts)
    mid_upscaled    = iterative_upscale(upscale, mid,    1+refinements, nghosts)
    fine_upscaled   = iterative_upscale(upscale, fine,     refinements, nghosts)

    output = np.zeros(fine_upscaled.shape)
    for comp in range(coarse.shape[0]):
        for i in tqdm(range(fine_upscaled.shape[1])):
            for j in tqdm(range(fine_upscaled.shape[2])):
                c = coarse_upscaled[comp,i,j]
                m = mid_upscaled[comp,i,j]
                f = fine_upscaled[comp,i,j]
                if rate is None:
                    est_rate = Richardson_estimate_rate(c,m,f)
                else:
                    est_rate = rate

                output[comp, i, j] = Richardson_extrapolation_step(m, f, est_rate)
    return output
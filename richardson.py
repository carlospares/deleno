import eno_tools as eno
from grid import Grid

def richardson_estimate_rate(coarse, mid, fine):
    # take data at three consecutive resolutions, all with shape = fine.shape
    # If we denote (nx, ny) the resolution of fine, then  this means coarse must have been 
    # upscaled twice from (nx/4, ny/4), and mid upscaled once from (nx/2, ny/2).
    # return a Richardson approximation to the rate of convergence

    return 0.5 # todo

def richardson_extrapolation_step(coarse, fine, rate):
    # take data at two consecutive resolutions, upscaled to having shape = result.shape.
    # As the mesh is refined by steps of 2, and we want to upscale, then this means that
    # coarse must have been upscaled at least twice, and fine at least once.
    # rate is the exponent of the leading term for the error.
    #
    # return the Richardson extrapolation at the resolution of fine
    #
    # assuming correct upscaling and availability of initial data, this can be used iteratively.

    return ( (2**rate)*fine - coarse) / (2**rate - 1)

def iterative_upscale(upscale, data, niters, nghosts):
    # iteratively apply upscaling function upscale to data, niters times.
    data_upscaled = data
    for i in range(niters):
        grid = Grid(data_upscaled.shape, nghosts)
        grid.bcs.extend_with_bc_2d(data_upscaled, nghosts)
        data_upscaled = upscale(data_upscaled, grid)
    return data_upscaled



def richardson_extrapolation_fv(coarse, mid, fine, rate=None, order=0):
    # take three vector fields at successive resolutions, with cell averages.
    # If fine has shape (nx, ny), then mid has shape (nx/2, ny/2) and coarse (nx/4, ny/4).
    # output Richardson extrapolation to (2*nx, 2*ny).
    # if rate is given, use for Richardson extrapolation; otherwise, compute.
    # if order > 0, use eno upscaling of that order. If order = 0, use trivial upscaling.

    if order == 0:
        upscale = lambda data, grid: eno.trivial_fv_2d_predictor(data, grid)
        nghosts = 0
    else:
        upscale = lambda data, grid: eno.fv_2d_predictor(data, order, grid)
        nghosts = order
    # upscale coarse three times, mid twice, fine once
    coarse_upscaled = iterative_upscale(upscale, coarse, 3, nghosts)
    mid_upscaled    = iterative_upscale(upscale, mid,    2, nghosts)
    fine_upscaled   = iterative_upscale(upscale, fine,   1, nghosts)
    
    if rate is None:
        rate = richardson_estimate_rate(coarse_upscaled, mid_upscaled, fine_upscaled)

    return richardson_extrapolation_step(mid_upscaled, fine_upscaled, rate)
from boundary_condition import BoundaryCondition
import eno_tools as eno
from grid import Grid

def compressed_decoding_staggered(f0, d, options):
    bc = BoundaryCondition(options.bcs)
    gw = options.order
    N0 = options.minN
    if f0.shape == (N0, N0): # ghost cells not included
        fhat_old = bc.extend_with_bc_2d(f0, gw)
    elif f0.shape == (N0 + 2*gw, N0 + 2*gw):
        # this already comes with ghost cells included, we do not need to do anything
        pass
    else:
        raise Exception(f"decoding received f0 of shape {f0.shape}. Does this include ghost cells?")

    # check data is not malformed
    for k in range(len(d)):
        if d[k].shape == (N0*(2**(k+1)), N0*(2**(k+1))):
            # ghost cells not included! Expand
            d[k] = bc.extend_with_bc_2d(d[k], gw)
        elif d[k].shape == (N0*(2**(k+1)) + 2*gw, N0*(2**(k+1)) + 2*gw):
            # ghost cells already included! Nothing to do
            pass
        else:
            raise Exception(f"decoding received d[{k}] of shape {d[k].shape}. \
                                Does this include ghost cells?")

    # if we are here, we have f0 and d[:] with the correct sizes and ghost cells included.

    fhat = f0
    for k in range(len(d)): # \le
        grid = Grid((N0*(2**k), N0*(2**k)), gw, options)
        fhat = grid.bcs.extend_with_bc_2d( 
                    eno.interpolate_2d_predictor_staggered(fhat, options.order, grid, eno.COMP_HOR),
                    gw) + d[k]
    return fhat[gw:-gw, gw:-gw]

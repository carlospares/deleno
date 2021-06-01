import eno_tools as eno
import numpy as np
import data_handler as dh
from grid import Grid

def truncate(arr_data, threshold):
    for i in range(arr_data.shape[0]):
        for j in range(arr_data.shape[1]):
            if abs(arr_data[i,j]) < threshold:
                arr_data[i,j] = 0

def idiv_tuple(t, d):
    # integer-divide all elements in t by d
    return tuple(int(e/d) for e in t)

def compressed_encoding_staggered(options):
    eps=0.
    t=1.
    print("TODO: fix epsilon and t to proper values!!")
    fK = dh.load_npy(options.name)[0]
    K = int(round(np.log2(options.maxN/options.minN)))
    gw = options.order
    
    f_u = [ [] for i in range(K+1)]
    fhat_u = [ [] for i in range(K+1)]
    d_u = [ [] for i in range(K+1)]
    ftilde_u = [ [] for i in range(K+1)]
    dhat_u = [ [] for i in range(K+1)]

    grid = Grid(fK.shape, gw)
    f_u[K] = grid.bcs.extend_with_bc_2d(fK, gw)

    for k in range(K, 0, -1):
        grid_fine = Grid( idiv_tuple(fK.shape, 2**(K-k)) , gw)
        grid_coarse = Grid( idiv_tuple(fK.shape, 2**(K-k+1)) , gw)

        f_u[k-1] = grid_coarse.bcs.extend_with_bc_2d(
                        eno.interpolate_2d_decimator_staggered(f_u[k], gw, grid_fine, eno.COMP_HOR),
                        gw)

    fhat_u[0] = f_u[0]
    for k in range(1, K+1):
        grid_fine = Grid( idiv_tuple(fK.shape, 2**(K-k)) , gw)
        grid_coarse = Grid( idiv_tuple(fK.shape, 2**(K-k+1)) , gw)

        ftilde_u[k] = grid_fine.bcs.extend_with_bc_2d(
                                eno.interpolate_2d_predictor_staggered(fhat_u[k-1], gw, 
                                                                        grid_coarse, eno.COMP_HOR),
                                gw)

        d = f_u[k] - ftilde_u[k]
        truncate(d, eps*(t ** (K-k)))
        dhat_u[k] = d
        fhat_u[k] = ftilde_u[k] + dhat_u[k]

    return (f_u[0], dhat_u)
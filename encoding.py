import eno_tools as eno
import numpy as np
import data_handler as dh
from grid import Grid

def truncate(arr_data, threshold):
    num_clipped = 0
    for i in range(arr_data.shape[0]):
        for j in range(arr_data.shape[1]):
            if abs(arr_data[i,j]) < threshold:
                arr_data[i,j] = 0
                num_clipped += 1
    return num_clipped

def idiv_tuple(t, d):
    # integer-divide all elements in t by d
    return tuple(int(e/d) for e in t)


def compressed_encoding(options, t=0.2):
    fK = dh.load_data(options.name)
    K = int(round(np.log2(options.maxN/options.minN)))
    gw = options.gw
    
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

        if options.is_grid_staggered():
            f_u[k-1] = grid_coarse.bcs.extend_with_bc_2d(
                            eno.interpolate_2d_decimator_staggered(f_u[k], gw, grid_fine, eno.COMP_HOR),
                            gw)
        elif options.is_grid_fv():
            f_u[k-1] = grid_coarse.bcs.extend_with_bc_2d(
                        eno.fv_2d_decimator(f_u[k], grid_fine), gw)
        else:
            raise Exception("encoding :: Type of grid not known!")

    fhat_u[0] = f_u[0]
    for k in range(1, K+1):
        grid_fine = Grid( idiv_tuple(fK.shape, 2**(K-k)) , gw)
        grid_coarse = Grid( idiv_tuple(fK.shape, 2**(K-k+1)) , gw)

        if options.is_grid_staggered():
            ftilde_u[k] = grid_fine.bcs.extend_with_bc_2d(
                                    eno.interpolate_2d_predictor_staggered(fhat_u[k-1], gw, 
                                                                            grid_coarse, eno.COMP_HOR),
                                    gw)
        elif options.is_grid_fv():
            ftilde_u[k] = grid_fine.bcs.extend_with_bc_2d(
                                eno.fv_2d_predictor(fhat_u[k-1], gw, grid_coarse), gw)

        d = f_u[k] - ftilde_u[k]
        compression = truncate(d, options.epsilon*(t ** (K-k)))
        Nfine = grid_fine.nxg*grid_fine.nyg
        if options.verbose:
            print(f"Compression rate: {compression*100./Nfine}% ({compression}/{Nfine})")
        dhat_u[k] = d
        fhat_u[k] = ftilde_u[k] + dhat_u[k]

    if options.compare:
        return (f_u, dhat_u[1:])
    else:
        return (f_u[0], dhat_u[1:]) # dhat_u[0] is 0
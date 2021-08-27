import numpy as np
from boundary_condition import BoundaryCondition as BC

BIAS_RIGHT = 1
BIAS_NONE = 0

COMP_HOR = 0
COMP_VERT = 1


ENO3_INTERP_COEF = np.array([ [0.375,0.75,-0.125], 
                                [-0.125,0.75,0.375], 
                                [0.375,-1.25,1.875] ]) #loffset: 0, 1, 2
ENO5_INTERP_COEF = np.array([[0.2734375,1.09375,-0.546875,0.21875,-0.0390625], 
                                [-0.0390625,0.46875,0.703125,-0.15625,0.0234375],
                                [0.0234375,-0.15625,0.703125,0.46875,-0.0390625],
                                [-0.0390625,0.21875,-0.546875,1.09375,0.2734375],
                                [0.2734375,-1.40625,2.953125,-3.28125,2.4609375]]) #loffset: 0..4
ENO7_INTERP_COEF = np.array([[0.2255859375,1.353515625,-1.1279296875,0.90234375,-0.4833984375,0.150390625,-0.0205078125],  
                                [-0.0205078125,0.369140625,0.9228515625,-0.41015625,0.1845703125,-0.052734375,0.0068359375],  
                                [0.0068359375,-0.068359375,0.5126953125,0.68359375,-0.1708984375,0.041015625,-0.0048828125],  
                                [-0.0048828125,0.041015625,-0.1708984375,0.68359375,0.5126953125,-0.068359375,0.0068359375],  
                                [0.0068359375,-0.052734375,0.1845703125,-0.41015625,0.9228515625,0.369140625,-0.0205078125],  
                                [-0.0205078125,0.150390625,-0.4833984375,0.90234375,-1.1279296875,1.353515625,0.2255859375],  
                                [0.2255859375,-1.599609375,4.8876953125,-8.37890625,8.7978515625,-5.865234375,2.9326171875 ]])
ENO9_INTERP_COEF = np.array([[0.196380615234375,1.571044921875,-1.8328857421875,2.199462890625,-1.96380615234375,1.221923828125,-0.4998779296875,0.120849609375,-0.013092041015625],  
                                [-0.013092041015625,0.314208984375,1.0997314453125,-0.733154296875,0.54986572265625,-0.314208984375,0.1221923828125,-0.028564453125,0.003021240234375],  
                                [0.003021240234375,-0.040283203125,0.4229736328125,0.845947265625,-0.35247802734375,0.169189453125,-0.0604248046875,0.013427734375,-0.001373291015625],
                                [-0.001373291015625,0.015380859375,-0.0897216796875,0.538330078125,0.67291259765625,-0.179443359375,0.0538330078125,-0.010986328125,0.001068115234375],  
                                [0.001068115234375,-0.010986328125,0.0538330078125,-0.179443359375,0.67291259765625,0.538330078125,-0.0897216796875,0.015380859375,-0.001373291015625],  
                                [-0.001373291015625,0.013427734375,-0.0604248046875,0.169189453125,-0.35247802734375,0.845947265625,0.4229736328125,-0.040283203125,0.003021240234375],  
                                [0.003021240234375,-0.028564453125,0.1221923828125,-0.314208984375,0.54986572265625,-0.733154296875,1.0997314453125002,0.314208984375,-0.013092041015625],  
                                [-0.013092041015625,0.120849609375,-0.4998779296875,1.221923828125,-1.96380615234375,2.199462890625,-1.8328857421875,1.571044921875,0.196380615234375],  
                                [0.196380615234375,-1.780517578125,7.1905517578125,-16.995849609375,25.96588134765625,-26.707763671875,18.695434570312496,-8.902587890625,3.338470458984375]])

# I can't find a source for these, so I have derived them myself
# see scripts/FV_coefs_upscale.m
ENO3_AVG_UPSCALE_COEF_L = np.array([ [ 1.375, -0.5, 0.125 ],
                                    [0.125, 1, -0.125], 
                                    [-0.125, 0.5, 0.625] ])
ENO3_AVG_UPSCALE_COEF_R = np.array([ [0.625, 0.5, -0.125],
                                     [-0.125, 1, 0.125],
                                     [0.125, -0.5, 1.375] ])
ENO5_AVG_UPSCALE_COEF_L = np.array([[1.5078125,-0.953125,0.6875,-0.296875,0.0546875],
                                    [0.0546875,1.234375,-0.40625,0.140625,-0.0234375],
                                    [-0.0234375,0.171875,1,-0.171875,0.0234375],
                                    [0.0234375,-0.140625,0.40625,0.765625,-0.0546875],
                                    [-0.0546875,0.296875,-0.6875,0.953125,0.4921875]])
ENO5_AVG_UPSCALE_COEF_R = np.array([[0.4921875,0.953125,-0.6875,0.296875,-0.0546875],
                                    [-0.0546875,0.765625,0.40625,-0.140625,0.0234375],
                                    [0.0234375,-0.171875,1,0.171875,-0.0234375],
                                    [-0.0234375,0.140625,-0.40625,1.234375,0.0546875],
                                    [0.0546875,-0.296875,0.6875,-0.953125,1.5078125]])
ENO7_AVG_UPSCALE_COEF_L = np.array([[1.5810546875,-1.3515625,1.5810546875,-1.3515625,0.7431640625,-0.234375,0.0322265625],
                                    [0.0322265625,1.35546875,-0.6748046875,0.453125,-0.2236328125,0.06640625,-0.0087890625],
                                    [-0.0087890625,0.09375,1.1708984375,-0.3671875,0.1455078125,-0.0390625,0.0048828125],
                                    [0.0048828125,-0.04296875,0.1962890625,1,-0.1962890625,0.04296875,-0.0048828125],
                                    [-0.0048828125,0.0390625,-0.1455078125,0.3671875,0.8291015625,-0.09375,0.0087890625],
                                    [0.0087890625,-0.06640625,0.2236328125,-0.453125,0.6748046875,0.64453125,-0.0322265625],
                                    [-0.0322265625,0.234375,-0.7431640625,1.3515625,-1.5810546875,1.3515625,0.4189453125]])
ENO7_AVG_UPSCALE_COEF_R = np.array([[0.4189453125,1.3515625,-1.5810546875,1.3515625,-0.7431640625,0.234375,-0.0322265625],
                                    [-0.0322265625,0.64453125,0.6748046875,-0.453125,0.2236328125,-0.06640625,0.0087890625],
                                    [0.0087890625,-0.09375,0.8291015625,0.3671875,-0.1455078125,0.0390625,-0.0048828125],
                                    [-0.0048828125,0.04296875,-0.1962890625,1,0.1962890625,-0.04296875,0.0048828125],
                                    [0.0048828125,-0.0390625,0.1455078125,-0.3671875,1.1708984375,0.09375,-0.0087890625],
                                    [-0.0087890625,0.06640625,-0.2236328125,0.453125,-0.6748046875,1.35546875,0.0322265625],
                                    [0.0322265625,-0.234375,0.7431640625,-1.3515625,1.5810546875,-1.3515625,1.5810546875]])



def undivided_differences_1d(input_data, order, grid, bias=BIAS_NONE, include_left=False):
    """
        Returns an array with the left stencils (in {0, ,.., order-1}) for ENO interpolation, 
        selected with undivided differences, computed for all non-ghost cells

        input_data:   1d array of N physical cells + 2*grid.gw ghost cells
        order:        up to which udd to consider (must agree with order of ENO rec/interp)
        grid:         grid (from which to deduce number of ghost cells needed)
        bias:         if BIAS_NONE, start with cell i. If BIAS_RIGHT, start with stencil {i, i+1}.
        include_left: if True, return an (N+1)-array which has in [0] the undivided difference for
                      the last left ghost cells 
    """

    if bias not in [BIAS_NONE, BIAS_RIGHT]:
        raise Exception("Bias must be one of BIAS_NONE, BIAS_RIGHT")

    N = len(input_data) - 2*grid.gw + (1 if include_left else 0)
    gw_l = grid.gw-1 if include_left else grid.gw
    udd = np.zeros([order, len(input_data)] ) # we iteratively produce undivided diffs
    udd[0,:] = input_data
    for k in range(1,order):
        udd[k,:-1] = udd[k-1, 1:] - udd[k-1, :-1]

    loffsets = np.zeros(N,dtype=int)  #this is called r in the DeLENO paper
    start = 1 if bias==BIAS_NONE else 2

    for k in range(start,order):
        for i in range(N):
            if abs(udd[k, i+gw_l-loffsets[i]]) > abs(udd[k, i+gw_l-loffsets[i]-1]):
                loffsets[i] += 1
    return loffsets

def interpolate_1d(input_data, order, grid, bias=BIAS_NONE, include_left=False):
    """ returns ENO interpolations a half mesh-step to the right (no ghost cells)

        input_data: 1d array of N physical cells + 2*grid.gw ghost cells
    """

    # First set the ENO coefficients
    # If loffset[i] = k, then the scalar product <c[k], data in stencil> is the ENO reconstruction
    if order == 3:
        c = ENO3_INTERP_COEF
    elif order == 5:
        c = ENO5_INTERP_COEF
    elif order == 7:
        c = ENO7_INTERP_COEF
    elif order == 9:
        c = ENO9_INTERP_COEF
    else:
        raise Exception("ENO interpolation not implemented for order {}".format(order))

    loffsets = undivided_differences_1d(input_data, order, grid, bias, include_left)
    N = len(input_data)-2*grid.gw + (1 if include_left else 0)
    gw_l = grid.gw-1 if include_left else grid.gw
    enorecs = np.zeros(N)
    for i in range(N):
        lshift = loffsets[i]
        start = i+gw_l-lshift
        enorecs[i] = np.dot(c[lshift], input_data[start : start+order])
    return enorecs


def interpolate_2d_predictor_staggered(input_data, order, grid):
    """
        Receives
        input_data: a 3d, ncomp x (grid.nx + 2*grid.gw) x (grid.ny + 2*grid.gw) array  (ghost cells 
                    appropriately filled) with all components of velocity (staggered as Arakawa-C)

        Returns
        a 3d, ncomp x (2*grid.nx) x (2*grid.ny) array, ie with no ghost cells, containing the 
        upscaling with ENO<order> of input_data to the finer grid
    """
    output = np.zeros((input_data.shape[0], 2*grid.nx, 2*grid.ny))
    for comp in range(input_data.shape[0]):
        if comp == 0:
            component = COMP_HOR
        elif comp == 1:
            component = COMP_VERT
        else:
            raise Exception("Staggered grid only implemented for 2D")
        output[comp] = interpolate_2d_predictor_staggered_comp(input_data[comp], order, grid,
                                                                component)
    return output


def interpolate_2d_predictor_staggered_comp(input_data, order, grid, component=COMP_HOR):
    """
        Receives
        input_data: a 2d (grid.nx + 2*grid.gw) x (grid.ny + 2*grid.gw) array  (ghost cells 
                    appropriately filled) with one component of velocity (staggered as in Arakawa-C)
        component: COMP_HOR for u, COMP_VERT for v

        Returns
        a 2d, (2*grid.nx) x (2*grid.ny) array, ie with no ghost cells, containing the upscaling 
                    with ENO<order> of input_data to the finer grid

        Remark:
        As the mesh is staggered, the nodes for the coarse grid and fine grid are disjoint.
        E.g. reconstructing the horizontal (resp. v) velocity in the vertical  (resp. h) direction,
        one needs velocities at +0.25dy and +0.75dy. (resp. dx).

        We choose to do this as a two-step interpolation in this direction. In the first step we
        interpolate at +0.5dy (resp. dx), and in the second we interpolate at +0.25dy and +0.75dy.
        This could be done more efficiently by computing the ENO coefficients for interpolation at
        +0.25dx / +0.75dx (as opposed to the +0.5dx), but I am labelling this as TODO for now.
    """
    gw = grid.gw
    fine = np.zeros((2*grid.nx,2*grid.ny))

    if component == COMP_HOR:
        end_across = grid.ny
        end_along = grid.nx
        axis_across = grid.bcs.AXIS_NS
        axis_along = grid.bcs.AXIS_EW
    elif component == COMP_VERT:
        end_across = grid.nx
        end_along = grid.ny
        axis_across = grid.bcs.AXIS_EW
        axis_along = grid.bcs.AXIS_NS

    midgrid = np.zeros(2*end_across + 2*gw)
    for i in range(end_along):
        # intermediate step: interpolate at +0.5 mesh step
        if component == COMP_HOR:
            in_data = input_data[i+gw,:]
        else: # vertical component
            in_data = input_data[:,i+gw]
        midpoints = interpolate_1d(in_data, order, grid, bias=BIAS_RIGHT, include_left=True)

        midgrid[gw:(gw+2*end_across):2] = midpoints[0:-1]
        # we ignored the last element here ^ since this is an intermediate step for a right-biased
        # interpolation (if periodic, should coincide with the first)
        midgrid[(gw+1):(gw+2*end_across):2] = in_data[gw:-gw]
        grid.bcs.apply_bc_1d(midgrid, gw, axis_across)

        if component == COMP_HOR:
            fine[2*i,:] =  interpolate_1d(midgrid, order, grid, bias=BIAS_RIGHT)
        else: # vertical component
            fine[:, 2*i] =  interpolate_1d(midgrid, order, grid, bias=BIAS_RIGHT)

    midgrid = np.zeros(end_along + 2*gw)
    if component == COMP_HOR:
        for j in range(2*end_across):
            midgrid[gw:-gw] = fine[0::2, j] 
            grid.bcs.apply_bc_1d(midgrid, gw, axis_along)
            fine[1:(2*end_along):2, j] = interpolate_1d(midgrid, order, grid, bias=BIAS_RIGHT)
    else: # vertical component
        for j in range(2*end_across):
            midgrid[gw:-gw] = fine[j, 0::2] 
            grid.bcs.apply_bc_1d(midgrid, gw, axis_along)
            fine[j, 1:(2*end_along):2] = interpolate_1d(midgrid, order, grid, bias=BIAS_RIGHT)

    return fine


def interpolate_2d_decimator_staggered(input_data, order, grid):
    """
        Receives
        input_data: a 3d ncomp x (grid.nx + 2*grid.gw) x (grid.ny + 2*grid.gw) array  (ghost cells 
                    appropriately filled) with all components of velocity (staggered as Arakawa-C)

        Returns
        a 3d, ncomp x (grid.nx/2) x (grid.ny/2) array, ie with no ghost cells, containing the 
        downscaling with ENO<order> of input_data to the coarser grid. Note that this is non-trivial
        due to staggering.
    """
    output = np.zeros((input_data.shape[0], grid.nx//2, grid.ny//2))
    for comp in range(input_data.shape[0]):
        if comp == 0:
            component = COMP_HOR
        elif comp == 1:
            component = COMP_VERT
        else:
            raise Exception("Staggered grid only implemented for 2D")
        output[comp] = interpolate_2d_decimator_staggered_comp(input_data[comp], order, grid,
                                                                component)
    return output



def interpolate_2d_decimator_staggered_comp(input_data, order, grid, component=COMP_HOR):
    """
        Receives
        input_data: a 2d (grid.nx + 2*grid.gw) x (grid.ny + 2*grid.gw) array  (ghost cells 
                    appropriately filled) with one component of velocity (staggered as in Arakawa-C)
        component: COMP_HOR for u, COMP_VERT for v

        Returns
        a 2d, (grid.nx/2) x (grid.ny/2) array, ie with no ghost cells, containing the downscaling 
        with ENO<order> of input_data to the coarser grid. Note that this is non-trivial
        due to staggering.
    """
    gw = grid.gw
    if grid.nx % 2 != 0 or grid.ny % 2 != 0:
        raise Exception("Decimator will not work on meshes with odd dimensions!")
    coarse = np.zeros((grid.nx//2,grid.ny//2))

    if component==COMP_HOR:
        for i in range(0,grid.nx,2):
            coarse[i//2,:] = interpolate_1d(input_data[i+gw,:], order, grid, bias=BIAS_RIGHT)[0::2]
    else: # vertical component
        for j in range(0,grid.ny,2):
            coarse[:,j//2] = interpolate_1d(input_data[:,j+gw], order, grid, bias=BIAS_RIGHT)[0::2]
    return coarse


def eno_upscale_avg_1d(input_data, order, grid, include_left=False):
    """ returns ENO interpolations a half mesh-step to the right (no ghost cells)

        input_data: 1d array of N physical cells + 2*grid.gw ghost cells
    """

    # First set the ENO coefficients
    # If loffset[i] = k, then the scalar product <c[k], data in stencil> is the ENO reconstruction
    if order == 3:
        cL = ENO3_AVG_UPSCALE_COEF_L
        cR = ENO3_AVG_UPSCALE_COEF_R
    elif order == 5:
        cL = ENO5_AVG_UPSCALE_COEF_L
        cR = ENO5_AVG_UPSCALE_COEF_R
    elif order == 7:
        cL = ENO7_AVG_UPSCALE_COEF_L
        cR = ENO7_AVG_UPSCALE_COEF_R
    else:
        raise Exception("ENO interpolation not implemented for order {}".format(order))

    loffsets = undivided_differences_1d(input_data, order, grid, BIAS_NONE, include_left)
    N = len(input_data)-2*grid.gw + (1 if include_left else 0)
    gw_l = grid.gw-1 if include_left else grid.gw
    enorecs = np.zeros(2*N)
    for i in range(N):
        lshift = loffsets[i]
        start = i+gw_l-lshift
        enorecs[2*i] = np.dot(cL[lshift], input_data[start : start+order])
        enorecs[2*i + 1] = np.dot(cR[lshift], input_data[start : start+order])
    return enorecs

    loffsets = undivided_differences_1d(input_data, order, grid, BIAS_NONE, include_left)
    N = len(input_data)-2*grid.gw + (1 if include_left else 0)
    gw_l = grid.gw-1 if include_left else grid.gw
    enorecs = np.zeros(2*N)
    for i in range(N):
        lshift = loffsets[i]
        start = i+gw_l-lshift
        enorecs[2*i] = np.dot(cL[lshift], input_data[start : start+order])
        enorecs[2*i + 1] = np.dot(cR[lshift], input_data[start : start+order])
    return enorecs


def fv_2d_predictor(input_data, order, grid):
    """
        Receives
        input_data: a 3d ncomp x (grid.nx + 2*grid.gw) x (grid.ny + 2*grid.gw) array  (ghost cells 
                    appropriately filled) with cell avgs for all components of velocity

        Returns
        a 3d, ncomp x (2*grid.nx) x (2*grid.ny) array, ie with no ghost cells, containing the 
        upscaling with ENO<order> of input_data to cell avgs in the finer grid
    """
    output = np.zeros((input_data.shape[0], grid.nx*2, grid.ny*2))
    for comp in range(input_data.shape[0]):
        output[comp] = fv_2d_predictor_comp(input_data[comp], order, grid)
    return output


def fv_2d_predictor_comp(input_data, order, grid):
    """
        Receives
        input_data: a 2d (grid.nx + 2*grid.gw) x (grid.ny + 2*grid.gw) array  (ghost cells 
                    appropriately filled) with cell avgs for one component of velocity

        Returns
        a 2d, (2*grid.nx) x (2*grid.ny) array, ie with no ghost cells, containing the upscaling 
                    with ENO<order> of input_data to cell avgs in the finer grid
    """
    gw = grid.gw
    fine = np.zeros((2*grid.nx,2*grid.ny))

    for i in range(grid.nx):
        fine[2*i,:] =  eno_upscale_avg_1d(input_data[i+gw,:], order, grid)

    midgrid = np.zeros(grid.nx + 2*gw)
    for j in range(2*grid.ny):
        midgrid[gw:-gw] = fine[0::2, j] 
        grid.bcs.apply_bc_1d(midgrid, gw, grid.bcs.AXIS_EW)
        fine[:, j] = eno_upscale_avg_1d(midgrid, order, grid)
        #^ this intentionally overwrites the previous results. Can one be neater about it?

    return fine

def fv_2d_decimator(input_data, grid):
    """
        Receives
        input_data: a 3d ncomp x (grid.nx + 2*grid.gw) x (grid.ny + 2*grid.gw) array  (ghost cells 
                    appropriately filled) with all components of velocity (cell avgs)

        Returns
        a 3d, ncomp x (grid.nx/2) x (grid.ny/2) array, ie with no ghost cells, containing the 
        downscaling of the cell avgs of input_data to a coarser grid. This operation is exact.
    """
    output = np.zeros((input_data.shape[0], grid.nx//2, grid.ny//2))
    for comp in range(input_data.shape[0]):
        output[comp] = fv_2d_decimator_comp(input_data[comp], grid)
    return output

def fv_2d_decimator_comp(input_data, grid):
    """
        Receives
        input_data: a 2d (grid.nx + 2*grid.gw) x (grid.ny + 2*grid.gw) array (ghost cells 
                    appropriately filled) with one component of velocity (cell avgs)

        Returns
        a 2d, (grid.nx/2) x (grid.ny/2) array, ie with no ghost cells, containing the downscaling 
        of the cell avgs of input_data to a coarser grid. This operation is exact.
    """
    gw = grid.gw
    if grid.nx % 2 != 0 or grid.ny % 2 != 0:
        raise Exception("Decimator will not work on meshes with odd dimensions!")
    coarse = np.zeros((grid.nx//2, grid.ny//2))

    for i in range(0,grid.nx,2):
        for j in range(0, grid.ny,2):
            ig = i+gw
            jg = j+gw
            coarse[i//2,j//2] = 0.25*(input_data[ig,jg] + input_data[ig+1,jg] + \
                                                input_data[ig,jg+1] + input_data[ig+1,jg+1])
    return coarse

def trivial_fv_2d_predictor(input_data, grid):
    """
        Receives
        input_data: a 3d ncomp x (grid.nx + 2*grid.gw) x (grid.ny + 2*grid.gw) array  (ghost cells 
                    appropriately filled) with cell avgs for one component of velocity

        Returns
        a 3d, ncomp x (2*grid.nx) x (2*grid.ny) array, ie with no ghost cells, containing the 
        trivial FV upscaling (ie repeating cells) of input_data to cell avgs in the finer grid
    """
    output = np.zeros((input_data.shape[0], grid.nx*2, grid.ny*2))
    for comp in range(input_data.shape[0]):
        output[comp] = trivial_fv_2d_predictor_comp(input_data[comp], grid)
    return output


def trivial_fv_2d_predictor_comp(input_data, grid):
    """
        Receives
        input_data: a 2d (grid.nx + 2*grid.gw) x (grid.ny + 2*grid.gw) array  (ghost cells 
                    appropriately filled) with cell avgs for one component of velocity

        Returns
        a 2d, (2*grid.nx) x (2*grid.ny) array, ie with no ghost cells, containing the trivial
                    FV upscaling of input_data to cell avgs in the finer grid
    """
    gw = grid.gw
    fine = np.zeros((2*grid.nx,2*grid.ny))
    for i in range(grid.nx):
        for j in range(grid.ny):
            datum = input_data[i+gw, j+gw]
            fine[2*i, 2*j] = datum
            fine[2*i+1, 2*j] = datum
            fine[2*i, 2*j+1] = datum
            fine[2*i+1, 2*j+1] = datum

    return fine
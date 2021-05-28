import numpy as np
from boundary_condition import BoundaryCondition as BC

BIAS_LEFT = -1
BIAS_RIGHT = 1
BIAS_NONE = 0

ENO3_INTERP_COEF = np.array([ [0.375,0.75,-0.125], 
								[-0.125,0.75,0.375], 
								[0.375,-1.25,1.875] ]) #loffset: 0, 1, 2
ENO5_INTERP_COEF = np.array([[0.2734375,1.09375,-0.546875,0.21875,-0.0390625], 
								[-0.0390625,0.46875,0.703125,-0.15625,0.0234375],
								[0.0234375,-0.15625,0.703125,0.46875,-0.0390625],
								[-0.0390625,0.21875,-0.546875,1.09375,0.2734375],
								[0.2734375,-1.40625,2.953125,-3.28125,2.4609375]])
ENO7_INTERP_COEF = np.array([[2.9326171875,-5.865234375,8.7978515625,-8.37890625,4.8876953125,-1.599609375,0.2255859375],
								[0.2255859375,1.353515625,-1.1279296875,0.90234375,-0.4833984375,0.150390625,-0.0205078125],  
								[-0.0205078125,0.369140625,0.9228515625,-0.41015625,0.1845703125,-0.052734375,0.0068359375],  
								[0.0068359375,-0.068359375,0.5126953125,0.68359375,-0.1708984375,0.041015625,-0.0048828125],  
								[-0.0048828125,0.041015625,-0.1708984375,0.68359375,0.5126953125,-0.068359375,0.0068359375],  
								[0.0068359375,-0.052734375,0.1845703125,-0.41015625,0.9228515625,0.369140625,-0.0205078125],  
								[-0.0205078125,0.150390625,-0.4833984375,0.90234375,-1.1279296875,1.353515625,0.2255859375],  
								[0.2255859375,-1.599609375,4.8876953125,-8.37890625,8.7978515625,-5.865234375,2.9326171875 ]])
ENO9_INTERP_COEF = np.array([[3.338470458984375,-8.902587890625,18.6954345703125,-26.707763671875,25.96588134765625,-16.995849609375,7.1905517578125,-1.780517578125,0.196380615234375],  
								[0.196380615234375,1.571044921875,-1.8328857421875,2.199462890625,-1.96380615234375,1.221923828125,-0.4998779296875,0.120849609375,-0.013092041015625],  
								[-0.013092041015625,0.314208984375,1.0997314453125,-0.733154296875,0.54986572265625,-0.314208984375,0.1221923828125,-0.028564453125,0.003021240234375],  
								[0.003021240234375,-0.040283203125,0.4229736328125,0.845947265625,-0.35247802734375,0.169189453125,-0.0604248046875,0.013427734375,-0.001373291015625],
								[-0.001373291015625,0.015380859375,-0.0897216796875,0.538330078125,0.67291259765625,-0.179443359375,0.0538330078125,-0.010986328125,0.001068115234375],  
								[0.001068115234375,-0.010986328125,0.0538330078125,-0.179443359375,0.67291259765625,0.538330078125,-0.0897216796875,0.015380859375,-0.001373291015625],  
								[-0.001373291015625,0.013427734375,-0.0604248046875,0.169189453125,-0.35247802734375,0.845947265625,0.4229736328125,-0.040283203125,0.003021240234375],  
								[0.003021240234375,-0.028564453125,0.1221923828125,-0.314208984375,0.54986572265625,-0.733154296875,1.0997314453125002,0.314208984375,-0.013092041015625],  
								[-0.013092041015625,0.120849609375,-0.4998779296875,1.221923828125,-1.96380615234375,2.199462890625,-1.8328857421875,1.571044921875,0.196380615234375],  
								[0.196380615234375,-1.780517578125,7.1905517578125,-16.995849609375,25.96588134765625,-26.707763671875,18.695434570312496,-8.902587890625,3.338470458984375]])

def undivided_differences_1d(input_data, order, grid, bias=BIAS_NONE):
	"""
		Returns an array with the left stencils (in {0, ,.., order-1}) for
		ENO interpolation, selected with undivided differences 

		input_data: 1d array of N physical cells + 2*grid.gw ghost cells
		order: up to which udd to consider (must agree with order of ENO rec/interp)
		grid: grid (from which to deduce number of ghost cells needed)
		bias: if BIAS_NONE, start with cell i. If BIAS_RIGHT, start with stencil {i, i+1}. 
				If BIAS_LEFT, start with {i-1,i}.
	"""

	if bias not in [BIAS_LEFT, BIAS_NONE, BIAS_RIGHT]:
		raise Exception("Bias must be one of BIAS_LEFT, BIAS_NONE, BIAS_RIGHT")

	N = len(input_data) - 2*grid.gw
	udd = np.zeros([order, len(input_data)] ) # we iteratively produce undivided diffs
	udd[0,:] = input_data
	for k in range(1,order):
		udd[k,:-1] = udd[k-1, 1:] - udd[k-1, :-1]

	loffsets = np.zeros(N,dtype=int)  #this is called r in the DeLENO paper
	start = 1 if bias==BIAS_NONE else 2
	if bias==BIAS_LEFT:
		loffsets += 1

	for k in range(start,order):
		for i in range(N):
			if abs(udd[k, i+grid.gw-loffsets[i]]) > abs(udd[k, i+grid.gw-loffsets[i]-1]):
				loffsets[i] += 1
	return loffsets

def eno_interpolate_1d(input_data, order, grid, bias=BIAS_NONE):
	""" returns ENO interpolations a half mesh-step to the right

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

	loffsets = undivided_differences_1d(input_data, order, grid, bias)
	N = len(input_data)-2*grid.gw
	enorecs = np.zeros(N)
	for i in range(N):
		lshift = loffsets[i]
		start = i+grid.gw-lshift
		enorecs[i] = np.dot(c[lshift], input_data[start : start+order])
	return enorecs

def eno_interpolate_2d_predictor_staggered(input_data, order, grid):
	"""
		Receives
		input_data: a 3d, 2 x (grid.nx + 2*grid.gw) x (grid.ny + 2*grid.gw) array 
					(ghost cells appropriately filled)
					with components of velocity (staggered as in Arakawa-C)
		Returns
		a 3d, 2 x (2*grid.nx) x (2*grid.ny) array, ie with no ghost cells,
			containing the upscaling with ENO<order> of input_data to the finer grid
	"""

	# TO DO
	return




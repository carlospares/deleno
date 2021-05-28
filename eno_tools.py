import numpy as np
from boundary_condition import BoundaryCondition as BC

BIAS_LEFT = -1
BIAS_RIGHT = 1
BIAS_NONE = 0

# class EnoReconstruction:
# 	def __init__(self, options):
# 		self.scratch = np.zeros(options.N)

# 	
def undivided_differences_1d(input_data, order, grid, bias=BIAS_NONE):
	# input_data: 1d array of N physical cells + 2*grid.gw ghost cells
	# order: up to which udd to consider (must agree with order of ENO rec/interp)
	# grid: grid (from which to deduce number of ghost cells needed)
	# bias: if BIAS_NONE, start with cell i. If BIAS_RIGHT, start with stencil {i, i+1}. 
	#		If BIAS_LEFT, start with {i-1,i}.

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
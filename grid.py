import numpy as np
from boundary_condition import BoundaryCondition as BC

class Grid:
	def __init__(self, datashape, nghosts, options=None):
		# domain limits
		self.xmin = 0.
		self.xmax = 1.
		self.ymin = 0.
		self.ymax = 1.

		# nx : #rows (ie number of cells in y axis), NS axis, i in range(nx)
		# ny : #cols (ie number of cells in x axis), EW axis, j in range(ny)
		# data accessed like data[i,j]
		if len(datashape) == 3: # velocity pair
			self.nx = datashape[1]
			self.ny = datashape[2]
		elif len(datashape) == 2: # single variable
			self.nx = datashape[0]
			self.ny = datashape[1]
		self.gw = nghosts
		if options is None:
			self.bcs = BC() if options is None else BC(options.bcs)
		self.generate_params()

	def generate_params(self):
		# derives all non-input parameters

		self.nxg = self.nx + 2*self.gw
		self.nyg = self.ny + 2*self.gw
		self.dx = (self.xmax-self.xmin)/self.nx
		self.dy = (self.ymax-self.ymin)/self.ny

		# cell centers
		self.x = self.dx/2. + self.dx*np.arange(self.nx)
		self.y = self.dy/2. + self.dy*np.arange(self.ny)
		self.xg = self.bcs.extend_with_bc_1d(self.x, self.gw, self.bcs.AXIS_NS)
		self.yg = self.bcs.extend_with_bc_1d(self.y, self.gw, self.bcs.AXIS_EW)



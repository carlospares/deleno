import numpy as np

class BoundaryCondition:

	AXIS_NS = 0
	AXIS_EW = 1

	BC_PER = 'P'

	def __init__(self, bcs=None):
		if bcs is None:
			self.bcN = self.BC_PER
			self.bcS = self.BC_PER
			self.bcE = self.BC_PER
			self.bcW = self.BC_PER
		else:
			self.bcN = bcs[0]
			self.bcS = bcs[1]
			self.bcW = bcs[2]
			self.bcE = bcs[3]

			# ^ is XOR. For each axis, either both are P or none is.
			if ((self.bcN == self.BC_PER) ^ (self.bcS == self.BC_PER)) or \
					((self.bcE == self.BC_PER) ^ (self.bcW == self.BC_PER)):
				raise Exception("One-sided periodic BCs are not allowed")

	def apply_bc_1d(self, data, nghosts, direction=AXIS_NS):
		# overwrites ghost cells in data with appropriate values
		# data: array of N+2*nghosts cells; the left and right nghosts cells
		#       will be overwritten with appropriate data
		# direction: AXIS_NS or AXIS_EW. Defaults to AXIS_NS, to avoid typing it
		#       if it is not relevant
		if direction == self.AXIS_NS:
			bcL = self.bcS
			bcR = self.bcN
		elif direction == self.AXIS_EW:
			bcL = self.bcW
			bcR = self.bcE
		else:
			raise Exception("Direction not known to apply_bc_1d")

		N = len(data) - 2*nghosts
		if N < 0:
			raise Exception("There can't be more ghosts cells than physical!")
		if bcL == self.BC_PER: #...and then bcR is too...
			data[:nghosts] = data[N:N+nghosts]
			data[N+nghosts:] = data[nghosts: 2*nghosts]
		else:
			raise Exception("Type of BC not known to apply_bc_1d")


	def apply_bc_2d(self, data, nghosts):
		# overwrites ghost cells in data with appropriate values
		# data: 2d array of (N+2*nghosts) x (N+2*nghosts) cells
		N = data.shape[0] - 2*nghosts
		for i in range(N):
			self.apply_bc_1d(data[i+nghosts,:], nghosts, self.AXIS_EW)
		for j in range(N + 2*nghosts):
			self.apply_bc_1d(data[:,j], nghosts, self.AXIS_NS)
		return

	def extend_with_bc_1d(self, data, nghosts, direction=AXIS_NS):
		# creates a new array of size (len(data) + 2*nghosts) with correct BCs
		# out of a 1d array with physical cells only, data
		wBCs = np.zeros(len(data) + 2*nghosts)
		wBCs[nghosts:len(data)+nghosts] = data
		self.apply_bc_1d(wBCs, nghosts, direction)
		return wBCs

	def extend_with_bc_2d(self, data, nghosts):
		# creates a new array of size (len(data) + 2*nghosts)^2 with correct BCs
		# out of a 2d array with physical cells only, data
		nx = data.shape[0]
		ny = data.shape[1]
		wBCs = np.zeros((nx + 2*nghosts, ny + 2*nghosts))
		wBCs[nghosts:nx+nghosts, nghosts:ny+nghosts] = data
		self.apply_bc_2d(wBCs, nghosts)
		return wBCs
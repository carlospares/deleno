import argparse
from boundary_condition import BoundaryCondition as BC

class Options:

	def __init__(self):
		parser = argparse.ArgumentParser(description='Computes details as in DeLENO')
		parser.add_argument('-n', '--name', type=str, required=True, 
								help='Name of file to be computed')
		parser.add_argument('-minN', '--minN', type=int, required=True, 
								help='Number of cells in one direction (coarsest)')
		parser.add_argument('-maxN', '--maxN', type=int, required=True, 
								help='Number of cells in one direction (finest)')
		parser.add_argument('-bc', '--bc', type=str, nargs=4, default=['P','P','P','P'],
								help='Type of BCs (default: periodic), in order N,S,W,E')
		parser.add_argument('-o', '--order', type=int, default=3,
								help='Order of reconstruction/interpolation')
		args = parser.parse_args()
		self.name = args.name
		self.minN = args.minN
		self.maxN = args.maxN
		if self.minN >= self.maxN:
			raise Exception("minN >= maxN, no details can be computed!")
		self.order = args.order
		self.bcs = BC(args.bc)
		return
import unittest
from boundary_condition import BoundaryCondition
import numpy as np
import eno_tools as eno

class TestUndividedStencil(unittest.TestCase):
	def test_1d(self):
		"""
		Test stencil selection algorithm
		"""
		data = [4,8,15,16,23]
		p = 3
		BC = BoundaryCondition(["nonsense","nonsense","P","P"])
		wBC = BC.extend_with_bc_1d(data, 3, BC.AXIS_EW)
		np.testing.assert_array_equal(eno.undivided_differences_1d(wBC, 3, 3), [0,1,0,1,2])

	def test_biases(self):
		"""
		Test stencil selection algorithm when forcing biases
		"""
		data = [4,8,15,16,23,42]
		p = 4
		gw = 3
		BC = BoundaryCondition(["nonsense","nonsense","P","P"])
		wBC = BC.extend_with_bc_1d(data, gw, BC.AXIS_EW)
		np.testing.assert_array_equal(eno.undivided_differences_1d(wBC, p, gw, bias=eno.BIAS_NONE),  [0,1,0,1,2,3])
		np.testing.assert_array_equal(eno.undivided_differences_1d(wBC, p, gw, bias=eno.BIAS_LEFT),  [1,1,2,1,2,3])
		np.testing.assert_array_equal(eno.undivided_differences_1d(wBC, p, gw, bias=eno.BIAS_RIGHT), [0,1,0,1,2,0])

if __name__ == '__main__':
	unittest.main()
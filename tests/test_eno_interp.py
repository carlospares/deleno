import unittest
from boundary_condition import BoundaryCondition
import numpy as np
import eno_tools as eno
from grid import Grid

class TestEnoInterpolation(unittest.TestCase):
	def test_linear_1d(self):
		"""
		Test ENO reconstructions (linear case)
		"""
		data = [0,2,4,6,8]
		p = 3
		gw = 3

		BC = BoundaryCondition()
		grid = Grid((5,5), gw)
		wBC = BC.extend_with_bc_1d(data, grid.gw)
		np.testing.assert_array_equal(eno.eno_interpolate_1d(wBC, p, grid), [1,3,5,7,9])

	def test_quadratic_1d(self):
		"""
		Test ENO reconstructions (parabola case)
		"""
		data = [0,4,16,36,64]
		p = 3
		gw = 3

		BC = BoundaryCondition()
		grid = Grid((5,5), gw)
		wBC = BC.extend_with_bc_1d(data, grid.gw)
		np.testing.assert_array_equal(eno.eno_interpolate_1d(wBC, p, grid), [1,9,25,49,81])

	def test_cubic_1d(self):
		"""
		Test ENO reconstructions (cubic case)
		"""
		data = [0,2**3,4**3,6**3,8**3]
		p1 = 3
		p2 = 5
		gw = 3

		BC = BoundaryCondition()
		grid = Grid((5,5), gw)
		wBC = BC.extend_with_bc_1d(data, grid.gw)

		# ENO3 is not exact for cubic polynomials...
		np.testing.assert_raises(AssertionError, 
								np.testing.assert_array_equal, 
								eno.eno_interpolate_1d(wBC, p1, grid), 
								[1**3,3**3,5**3,7**3,9**3])
		# ...but ENO5 is
		np.testing.assert_array_equal(eno.eno_interpolate_1d(wBC, p2, grid), 
										[1**3,3**3,5**3,7**3,9**3])

	def test_biases(self):
		"""
		Test biased ENO reconstructions (parabola case)
		Periodic BCs mean that forcing a left bias will not be exact for first,
		forcing a right bias will not be exact for last
		"""
		data = [0,4,16,36,64]
		p = 3
		gw = 3

		BC = BoundaryCondition()
		grid = Grid((5,5), gw)
		wBC = BC.extend_with_bc_1d(data, grid.gw)

		interp_r = eno.eno_interpolate_1d(wBC, p, grid, bias=eno.BIAS_RIGHT)
		interp_l = eno.eno_interpolate_1d(wBC, p, grid, bias=eno.BIAS_LEFT)
		np.testing.assert_array_equal(interp_r[:-1], [1,9,25,49])
		np.testing.assert_array_equal(interp_l[1:], [9,25,49,81])
		self.assertNotEqual(interp_r[4], 81)
		self.assertNotEqual(interp_l[0], 1)

if __name__ == '__main__':
	unittest.main()
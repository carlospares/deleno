import unittest
from boundary_condition import BoundaryCondition
import numpy as np
import eno_tools as eno
from grid import Grid

class TestUndividedStencil(unittest.TestCase):
    def test_1d(self):
        """
        Test stencil selection algorithm
        """
        data = [4,8,15,16,23]
        p = 3
        gw = 3

        BC = BoundaryCondition()
        grid = Grid((5,5), gw)
        wBC = BC.extend_with_bc_1d(data, grid.gw)
        np.testing.assert_array_equal(eno.undivided_differences_1d(wBC, p, grid), [0,1,0,1,2])


    def test_biases(self):
        """
        Test stencil selection algorithm when forcing biases
        """
        data = [4,8,15,16,23,42]
        p = 4
        gw = 3

        BC = BoundaryCondition()
        grid = Grid((6,6), gw)
        wBC = BC.extend_with_bc_1d(data, grid.gw)
        np.testing.assert_array_equal(eno.undivided_differences_1d(wBC, p, grid, bias=eno.BIAS_NONE),  [0,1,0,1,2,3])
        np.testing.assert_array_equal(eno.undivided_differences_1d(wBC, p, grid, bias=eno.BIAS_RIGHT), [0,1,0,1,2,0])

    
    def test_higher_1d(self):
        """
        Test stencil selection algorithm for higher orders in a trivial case
        """
        data = np.arange(0,20,2)
        N = len(data)
        for p in [3,5,7,9]:
            gw = p
            grid = Grid((5,5), gw)
            wBC = grid.bcs.extend_with_bc_1d(data, grid.gw)
            udds = eno.undivided_differences_1d(wBC, p, grid)
            # Algorithm defaults to the right when both options are equal.
            # So we expect to get left shifts of 0 except when within p-1 cells of right BC
            np.testing.assert_array_equal(udds[0:-(p-1)], np.zeros(N - p + 1))
            np.testing.assert_array_equal(udds[-(p-1):], np.arange(1, p))

if __name__ == '__main__':
    unittest.main()
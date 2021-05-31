import unittest
from boundary_condition import BoundaryCondition
import numpy as np
import eno_tools as eno
from grid import Grid

class TestEnoInterpolation(unittest.TestCase):

    def test_all_orders_exact(self):
        """
        Test exactness of ENO interpolations for polynomials.
        ENOp should exactly interpolate polynomials up to degree (p-1), included.
        """
        basedata = np.arange(0., 20., 2.)
        baseref = np.arange(1., 21., 2.)
        N = len(basedata)
        for p in [3,5,7,9]:
            for exp in range(1,11):
                data = basedata**exp
                grid = Grid((N,N), p)
                wBC = grid.bcs.extend_with_bc_1d(data, grid.gw)
                if exp < p:
                    np.testing.assert_almost_equal(eno.interpolate_1d(wBC, p, grid),  
                                                    baseref**exp, decimal=3)
                else:
                    np.testing.assert_raises(AssertionError, 
                                np.testing.assert_almost_equal, 
                                eno.interpolate_1d(wBC, p, grid), 
                                baseref**exp, decimal=3)

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

        interp_r = eno.interpolate_1d(wBC, p, grid, bias=eno.BIAS_RIGHT)
        np.testing.assert_array_equal(interp_r[:-1], [1,9,25,49])
        self.assertNotEqual(interp_r[4], 81)

    def test_order_1d(self):
        """
        Test that the 1d ENOp interpolation produces an EOC of roughly p.
        We compare at resolutions 64 and 128: fine enough that orders should be close to limit,
        but coarse enough that ENO7 and 9 won't reach machine precision.
        """
        f = lambda x : np.sin(2*np.pi*x)
        for p in [3,5,7,9]:
            gw = p
            errors = []
            for N in [64,128]:
                grid = Grid((N,N), gw)
                data = f(grid.x)
                wBC = grid.bcs.extend_with_bc_1d(data, grid.gw)
                interp = eno.interpolate_1d(wBC, p, grid)
                fullinterp = np.zeros(2*N)
                fullinterp[0::2] = data
                fullinterp[1::2] = interp

                fine_grid = Grid((2*N, 2*N), gw)
                fine_data = f(fine_grid.x)
                errors.append(np.sum(np.abs(fine_data - fullinterp))/N)
            eoc = -(np.log(errors[1]) - np.log(errors[0]) )/np.log(2)
            np.testing.assert_almost_equal(eoc, p, decimal=1)

    def test_eno_2d(self):
        """
        Test 2d ENO interpolation
        """
        f = lambda x,y : np.sin(2*np.pi*x)*np.cos(2*np.pi*y)
        for p in [3,5,7,9]:
            gw = p
            errors = []
            for N in [64, 128]:
                grid = Grid((N,N), gw)
                X,Y = np.meshgrid(grid.x, grid.y + 0.5*grid.dy)
                data = f(X,Y).T
                wBC = grid.bcs.extend_with_bc_2d(data, grid.gw)
                interp = eno.interpolate_2d_predictor_staggered(wBC, p, grid)

                fine_grid = Grid((2*N, 2*N), gw)
                Xf, Yf = np.meshgrid(fine_grid.x, fine_grid.y + 0.5*fine_grid.dy)
                fine_data = f(Xf,Yf).T
                errors.append(np.sum(np.abs(fine_data - interp))/N/N)

            eoc = -(np.log(errors[1]) - np.log(errors[0]) )/np.log(2)
            np.testing.assert_almost_equal(eoc, p, decimal=1)


if __name__ == '__main__':
    unittest.main()
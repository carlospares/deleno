import unittest
from boundary_condition import BoundaryCondition
import numpy as np
import eno_tools as eno
from grid import Grid

class TestEnoUpscaling(unittest.TestCase):
    def test_all_orders_exact(self):
        """
        Test exactness of ENO interpolations for polynomials.
        ENOp should exactly upscale polynomials up to degree (p-1), included.
        """
        basedata = np.arange(0., 20., 2.)
        baseref = np.arange(0., 19., 1.)
        N = len(basedata)
        for p in [3,5,7]:
            for exp in range(1,11):
                data = 0.5*(basedata[1:]**(exp+1) - basedata[:-1]**(exp+1))/(p+1)
                ref =  (baseref[1:]**(exp+1) - baseref[:-1]**(exp+1))/(p+1)
                grid = Grid((N,N), p)
                wBC = grid.bcs.extend_with_bc_1d(data, grid.gw)
                if exp < p:
                    np.testing.assert_almost_equal(eno.eno_upscale_avg_1d(wBC, p, grid),  
                                                    ref, decimal=3)
                else:
                    np.testing.assert_raises(AssertionError, 
                                np.testing.assert_almost_equal, 
                                eno.eno_upscale_avg_1d(wBC, p, grid), 
                                ref**exp, decimal=3)

    def test_eno_fv_2d_predictor(self):
        """
        Test 2d ENO avg to avg reconstruction for upscaling
        """
        f = lambda x,y,dx,dy : (-np.cos(2*np.pi*(x + dx/2)) + np.cos(2*np.pi*(x - dx/2)) ) * \
                                (-np.sin(2*np.pi*(y + dy/2)) + np.sin(2*np.pi*(y - dy/2)) ) / \
                                (dx*dy*4*np.pi*np.pi)
        for p in [3,5,7]:
            gw = p
            errors = []
            for N in [64, 128]:
                grid = Grid((1,N,N), gw)
                X,Y = np.meshgrid(grid.x + 0.5*grid.dx, grid.y + 0.5*grid.dy)

                data = f(X,Y,grid.dx, grid.dy).T
                wBC = grid.bcs.extend_with_bc_2d(data[np.newaxis], grid.gw)
                interp = eno.fv_2d_predictor(wBC, p, grid)

                fine_grid = Grid((1,2*N, 2*N), gw)
                Xf, Yf = np.meshgrid(fine_grid.x + 0.5*fine_grid.dx, fine_grid.y + 0.5*fine_grid.dy)

                fine_data = f(Xf,Yf, fine_grid.dx, fine_grid.dy).T
                errors.append(np.sum(np.abs(fine_data - interp))/N/N)

            eoc = -(np.log(errors[1]) - np.log(errors[0]) )/np.log(2)
            np.testing.assert_almost_equal(eoc, p, decimal=1)

    def test_fv_2d_decimator(self):
        """
        Test 2d FV downscaling. This operation must be exact.
        """
        f = lambda x,y,dx,dy : (-np.cos(2*np.pi*(x + dx/2)) + np.cos(2*np.pi*(x - dx/2)) ) * \
                                (-np.sin(2*np.pi*(y + dy/2)) + np.sin(2*np.pi*(y - dy/2)) ) / \
                                (dx*dy*4*np.pi*np.pi)
        
        gw = 4
        for N in [128, 256]:
            grid = Grid((1,N,N), gw)
            X,Y = np.meshgrid(grid.x + 0.5*grid.dx, grid.y + 0.5*grid.dy)

            data = f(X,Y,grid.dx, grid.dy).T
            wBC = grid.bcs.extend_with_bc_2d(data[np.newaxis], grid.gw)
            avgd = eno.fv_2d_decimator(wBC, grid)

            coarse_grid = Grid((1, N/2, N/2), gw)
            Xc, Yc = np.meshgrid(coarse_grid.x + 0.5*coarse_grid.dx, coarse_grid.y + 0.5*coarse_grid.dy)

            coarse_data = f(Xc,Yc, coarse_grid.dx, coarse_grid.dy).T
            error = np.sum(np.abs(coarse_data - avgd))
            np.testing.assert_almost_equal(error, 0, decimal=8)

if __name__ == '__main__':
    unittest.main()
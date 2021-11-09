import unittest
import extrapolation as extr
import data_handler as dh
from grid import Grid
import numpy as np
import eno_tools as eno

import matplotlib.pyplot as plt

class TestExtrapolations(unittest.TestCase):

    def test_iterative_upscale_trivial_1d_fv(self):
        input_data = np.array([0,1])
        upscale,nghosts = extr.get_upscale_nghosts_1d(0)
        output_data = extr.iterative_upscale_1d(upscale, input_data, 2, nghosts)
        np.testing.assert_almost_equal(output_data, np.array([0,0,0,0,1,1,1,1]))

    def test_iterative_upscale_trivial_2d_fv(self):
        input_data = np.array([[[0,1],[2,3]]])
        upscale,nghosts = extr.get_upscale_nghosts(0)
        output_data = extr.iterative_upscale(upscale, input_data, 1, nghosts)
        np.testing.assert_almost_equal(output_data, 
            np.array([[[0,0,1,1],[0,0,1,1],[2,2,3,3],[2,2,3,3]]]))

    def test_iterative_upscale_fv(self):
        """
        Test exactness of ENO interpolations for polynomials.
        ENOp should exactly upscale polynomials up to degree (p-1), included.
        """
        n = 28 # if p > n/4, eno must pick boundary cells
        basedata = np.arange(0., n+4., 4.)
        baseref = np.arange(0.,  n+1., 1.)
        N = len(basedata)
        for p in [3,5,7]:
            upscale,nghosts = extr.get_upscale_nghosts_1d(p)
            for exp in range(1,11):
                data = 0.25*(basedata[1:]**(exp+1) - basedata[:-1]**(exp+1))/(exp+1)
                ref =  (baseref[1:]**(exp+1) - baseref[:-1]**(exp+1))/(exp+1)
                output_data = extr.iterative_upscale_1d(upscale, data, 2, nghosts)
                if exp < p:
                    np.testing.assert_almost_equal(output_data, ref, decimal=3)
                else:
                    np.testing.assert_raises(AssertionError, 
                                np.testing.assert_almost_equal, 
                                output_data, ref, decimal=3)


    def test_Richardson_rate(self):
        N = 10
        h = 0.5
        real_exp = 3
        # simulate a sequence converging to 1 with leading error term exponent real_exp
        x = [1 + (h**k)**real_exp + (h**k)**(real_exp + 1) for k in range(3,N)]

        for k in range(2,len(x)):
            est_rate = extr.Richardson_estimate_rate(x[k-2], x[k-1], x[k])
            np.testing.assert_almost_equal(est_rate, real_exp, decimal=1)

    def test_Richardson_step_scalar(self):
        est_rate = extr.Richardson_estimate_rate(16, 8, 4)
        np.testing.assert_almost_equal(est_rate, 1, decimal=5)
        np.testing.assert_almost_equal(extr.Richardson_extrapolation_step(8, 4, est_rate),
            0, decimal=5)
        
        est_rate = extr.Richardson_estimate_rate(81, 27, 9)
        np.testing.assert_almost_equal(est_rate, np.log2(3), decimal=5)
        np.testing.assert_almost_equal(extr.Richardson_extrapolation_step(27, 9, est_rate),
            0, decimal=5)


    def test_Aitken_step(self):
        np.testing.assert_almost_equal(
            extr.Aitken_extrapolation_step(np.array([32, 81]), np.array([16, 27]), np.array([8,9])),
            [0,0],
            decimal=5)

    def test_Anderson_step(self):
        np.testing.assert_almost_equal(
            extr.Anderson_extrapolation_step(np.array([32, 81]), np.array([16, 27]), np.array([8,9])),
            [3.811764,-0.423529],
            decimal=5)

    def extrapolation_scalar_framework(self, algo, osc=False, Cn=False):
        limit = 1
        N = 10
        h = 0.5
        real_exp = 3
        real_exp_second = 5
        if real_exp_second <= real_exp:
            raise Exception("Bad input! real_exp must be leading error, not real_exp_second")

        hs = [h**k for k in range(1,N)]
        base = -1 if osc else 1
        if Cn:
            C = 100*np.random.random(len(hs))
        else:
            C = np.ones(len(hs))

        # simulate a sequence converging to 1 with leading error term exponent real_exp
        x = [limit + C[i] * (base**i) * hs[i]**real_exp + hs[i]**real_exp_second for i in range(len(hs))]
        error_sequence = [np.abs(xa - limit) for xa in x]

        order_base = [np.log2(j/i) for i, j in zip(error_sequence[1:],error_sequence[:-1])]
        np.testing.assert_almost_equal(np.median(order_base), real_exp, decimal=2)

        error_extrap = np.zeros(len(x) - 2)
        for k in range(2,len(x)):
            est_rate = extr.Richardson_estimate_rate(x[k-2], x[k-1], x[k])
            extrap = algo(x[k-2], x[k-1], x[k])
            error_extrap[k-2] = np.abs(limit - extrap)

        order_acc = [np.log2(j/i) for i, j in zip(error_extrap[1:],error_extrap[:-1])]
        np.testing.assert_almost_equal(np.median(order_acc), real_exp_second, decimal=2)

    def test_extrapolation_scalar_Richardson(self):
        self.extrapolation_scalar_framework(extr.Richardson_extrapolation_step_standalone)

    def test_extrapolation_scalar_Anderson(self):
        self.extrapolation_scalar_framework(extr.Anderson_extrapolation_step)

    def test_extrapolation_scalar_Aitken(self):
        self.extrapolation_scalar_framework(extr.Aitken_extrapolation_step)

    def test_extrapolation_scalar_Richardson_osc(self):
        # Richardson extrapolation won't work with an error C(n) h^exp
        with np.testing.assert_raises(AssertionError):
            self.extrapolation_scalar_framework(extr.Richardson_extrapolation_step_standalone, 
                osc=True)

    def test_extrapolation_scalar_Anderson_osc(self):
        self.extrapolation_scalar_framework(extr.Anderson_extrapolation_step, osc=True)

    def test_extrapolation_scalar_Aitken_osc(self):
        self.extrapolation_scalar_framework(extr.Aitken_extrapolation_step, osc=True)

    # These must fail: extrapolation with a leading error C(h)*(h**alpha) shouldn't improve things
    def test_extrapolation_scalar_Richardson_Cn(self):
        with np.testing.assert_raises(AssertionError):
            self.extrapolation_scalar_framework(extr.Richardson_extrapolation_step_standalone, 
                                                Cn=True)

    def test_extrapolation_scalar_Anderson_Cn(self):
        with np.testing.assert_raises(AssertionError):
            self.extrapolation_scalar_framework(extr.Anderson_extrapolation_step, Cn=True)

    def test_extrapolation_scalar_Aitken_Cn(self):
        with np.testing.assert_raises(AssertionError):
            self.extrapolation_scalar_framework(extr.Aitken_extrapolation_step, Cn=True)



    def extrapolation_vector_fixed_size_framework(self, algo, osc=False, Cn=False):
        x = np.arange(0, 1, 0.05)
        sol = np.sin(2*np.pi*x)
        limit = 1
        N = 10
        h = 0.5
        real_exp = 3
        real_exp_second = 5
        if real_exp_second <= real_exp:
            raise Exception("Bad input! real_exp must be leading error, not real_exp_second")

        hs = [h**k for k in range(1,N)]
        # simulate a series of approximations to sol with errors of orders real_exp[_second]
        approxs = []
        D = np.random.rand(len(x))
        base = -1 if osc else 1
        if Cn:
            C = 1 + 0.1*np.random.random(len(hs))
        else:
            C = np.ones(len(hs))
        for k, ha in enumerate(hs):
            approxs.append(np.array([sol[i] + (base**k) * C[k] * D[i]*ha**real_exp + ha**real_exp_second 
                                        for i in range(len(sol))]))

        error_sequence = [extr.norm_2(sol - approx) for approx in approxs]
        error_extrap = np.zeros(len(approxs) - 2)

        for k in range(2,len(approxs)):
            extrap = algo(approxs[k-2],approxs[k-1],approxs[k])
            error_extrap[k-2] = extr.norm_2(sol - extrap)

        order_base = [np.log2(j/i) for i, j in zip(error_sequence[1:],error_sequence[:-1])]
        order_acc = [np.log2(j/i) for i, j in zip(error_extrap[1:],error_extrap[:-1])]

        np.testing.assert_almost_equal(np.median(order_base), real_exp, decimal=1)
        np.testing.assert_almost_equal(np.median(order_acc), real_exp_second, decimal=1)

    def test_extrapolation_vector_fixed_size_Richardson(self):
        self.extrapolation_vector_fixed_size_framework(
                                        extr.Richardson_extrapolation_step_standalone)

    def test_extrapolation_vector_fixed_size_Anderson(self):
        self.extrapolation_vector_fixed_size_framework(extr.Anderson_extrapolation_step)

    def test_extrapolation_vector_fixed_size_Aitken(self):
        self.extrapolation_vector_fixed_size_framework(extr.Aitken_extrapolation_step)

    # def test_extrapolation_vector_fixed_size_cwiseRichardson(self):
    #     self.extrapolation_vector_fixed_size_framework(
    #                                     extr.cwiseRichardson_extrapolation_step)

    def test_extrapolation_vector_fixed_size_Richardson_osc(self):
        with np.testing.assert_raises(AssertionError):
            self.extrapolation_vector_fixed_size_framework(
                                        extr.Richardson_extrapolation_step_standalone, osc=True)

    def test_extrapolation_vector_fixed_size_Anderson_osc(self):
        self.extrapolation_vector_fixed_size_framework(extr.Anderson_extrapolation_step, osc=True)

    def test_extrapolation_vector_fixed_size_Aitken_osc(self):
        self.extrapolation_vector_fixed_size_framework(extr.Aitken_extrapolation_step, osc=True)

    # def test_extrapolation_vector_fixed_size_cwiseRichardson_osc(self):
    #     self.extrapolation_vector_fixed_size_framework(
    #                                     extr.cwiseRichardson_extrapolation_step, osc=True)

    def test_extrapolation_vector_fixed_size_Richardson_Cn(self):
        with np.testing.assert_raises(AssertionError):
            self.extrapolation_vector_fixed_size_framework(
                                        extr.Richardson_extrapolation_step_standalone, Cn=True)

    def test_extrapolation_vector_fixed_size_Anderson_Cn(self):
        with np.testing.assert_raises(AssertionError):
            self.extrapolation_vector_fixed_size_framework(extr.Anderson_extrapolation_step,
                                                            Cn=True)

    def test_extrapolation_vector_fixed_size_Aitken_Cn(self):
        with np.testing.assert_raises(AssertionError):
            self.extrapolation_vector_fixed_size_framework(extr.Aitken_extrapolation_step, Cn=True)

    # def test_extrapolation_vector_fixed_size_cwiseRichardson_Cn(self):
    #     self.extrapolation_vector_fixed_size_framework(
    #                                     extr.cwiseRichardson_extrapolation_step, Cn=True)

    def discontinuous_f_cellavgs(self, x, dx):
        np.testing.assert_array_less(dx, 1) # otherwise integrals don't work
        h = dx/2

        return  (1./dx)*( \
                              (x+h<=0.5)   *-0.5*((x + h)**2 - (x - h)**2) + \
                    (x-h<0.5)*(x+h>0.5)   *(-0.5*(0.25 - (x - h)**2) \
                            + (0.3/np.pi)*(-np.cos(10*np.pi*(x + h)) -1)) + \
                    (x-h>=0.5)*(x+h<=1.5) *(0.3/np.pi)*(-np.cos(10*np.pi*(x + h)) + \
                                             np.cos(10*np.pi*(x - h))) + \
                    (x-h<1.5)*(x+h>1.5)*  ((0.3/np.pi)*(1 + np.cos(10*np.pi*(x - h))) + \
                                            (-20./3)*( (x+h-2)**3 - (-0.5)**3 )) + \
                    (x-h>=1.5)*(x+h<=2.5) *(-20./3)*( (x+h-2)**3 - (x-h-2)**3 ) + \
                    (x-h<2.5)*(x+h>2.5)   *((-20./3)*( (0.5)**3 - (x-h-2)**3) + 3*(x+h - 2.5)) + \
                    (x-h>=2.5)            *3*dx )

    def discontinuous_f(self, x):
        return  (x<=0.5)*          (-x) + \
                (x>0.5)*(x<=1.5)*(3*np.sin(10*np.pi*x)) + \
                (x>1.5)*(x<=2.5)*(-20*(x-2)**2) + \
                (x>2.5)*3

    def test_f_discont_cellavgs(self):
        dx = 0.001
        a = 0
        b = 3
        x = np.arange(a + dx/2, b, dx)
        np.testing.assert_almost_equal(self.discontinuous_f(x),
                                       self.discontinuous_f_cellavgs(x, dx), decimal=3)

    def extrapolation_vector_upscaling_framework(self, algo, name="", order=0, plot_approxs=False,
                                                 plot_extraps=False, plot_cells=False, 
                                                 plot_cell_errors=False, plot_global=False,
                                                 use_discontinuous=False):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        nc = len(colors)
        
        upscale, nghosts = extr.get_upscale_nghosts_1d(order)
        if use_discontinuous:
            a = 0
            b = 3
            f = lambda x, dx: self.discontinuous_f_cellavgs(x, dx)
            gt = lambda x, dx: self.discontinuous_f(x)
        else:
            a = 0
            b = 1
            f = lambda x, dx: (-np.cos(2*np.pi*(x + dx/2)) + np.cos(2*np.pi*(x - dx/2)))/(dx*2*np.pi)
            gt = f
        approxs = []
        Ns = [64, 128, 256, 512, 1024]

        N_truth = 2*Ns[-1]
        #compute ground truth:
        dx = (b-a)/N_truth
        x_truth = np.arange(a + dx/2, b, dx)
        ground_truth = gt(x_truth, dx)

        for N in Ns:
            dx = (b-a)/N
            x = np.arange(a + dx/2, b, dx)
            approxs.append( f(x, dx) )
        approxs_upsc = []
        for k,N in enumerate(Ns):
            niters = int(np.log2(N_truth/N))
            approxs_upsc.append(extr.iterative_upscale_1d(upscale, approxs[k],
                                                                   niters, nghosts))
            # plt.plot(x, approxs_upsc[-1], "*-", label=f"{N}", c=colors[k])
        error_sequence = [extr.norm_2(ground_truth - ap) for ap in approxs_upsc]
        error_extrap = np.zeros(len(approxs) - 2)


        extraps = []
        for k in range(2, len(Ns)):
            extrap = algo(approxs_upsc[k-2], approxs_upsc[k-1], approxs_upsc[k])
            extraps.append(extrap)
            error_extrap[k-2] = extr.norm_2(ground_truth - extrap)

        if plot_approxs:
            for k in range(2, len(Ns)):
                plt.plot(x_truth, approxs_upsc[k], "*--", label=f"{Ns[k]}", c=colors[k%nc])
            plt.legend()
            plt.show()

        if plot_extraps:
            for k in range(2, len(Ns)):
                plt.plot(x_truth, extraps[k-2], "*--", label=f"{Ns[k]}", c=colors[k%nc])
            plt.legend()
            plt.show()

        if plot_cells:
            fig = plt.figure()
            offset = 1690
            for i in range(offset, offset + int(N_truth/Ns[0]), 2):
                plt.plot(np.log2(Ns), [a[i] for a in approxs_upsc], 
                           '-X', c=colors[i%nc])#,label=f"Error cell {i}"
                plt.plot(np.log2(Ns[2:]), [a[i] for a in extraps], 
                           '--*', c=colors[i%nc])#,label=f"Extrap cell {i}"
                plt.plot(np.log2(Ns[-1]+50), ground_truth[i], 'o', c=colors[i%nc])
            plt.legend(['real value', 'accelerated', 'ground truth'], loc="upper left")
            plt.grid()
            # ax = fig.gca()
            # ax.set_yticks(np.arange(0, 0.1, 0.005))
            plt.title(f"{name} ({order})")
            plt.show()

        if plot_cell_errors:
            for i in range(0, int(N_truth/Ns[0])):
                plt.loglog(Ns, np.abs([a[i] for a in approxs_upsc] - ground_truth[i]), 
                           '-',label=f"Error cell {i}", base=2, c=colors[i%nc])
                plt.loglog(Ns[2:], np.abs([a[i] for a in extraps] - ground_truth[i]), 
                           '--*',label=f"Extrap cell {i}", base=2, c=colors[i%nc])
            plt.loglog(Ns, 1./np.square(Ns), '--', base=2, label="N^(-2)")
            plt.grid("minor")
            plt.legend()
            plt.show()

        if plot_global:
            plt.loglog(Ns, error_sequence, 'b*-', base=2)
            plt.loglog(Ns[2:], error_extrap, 'r*-', base=2)
            plt.grid()
            plt.title(f"{name} ({order})")
            plt.show()

    def test_extrapolation_vector_upscaling_Richardson(self):
        self.extrapolation_vector_upscaling_framework(
                                        extr.Richardson_extrapolation_step_standalone, 
                                        order=0, name="Richardson")
        self.extrapolation_vector_upscaling_framework(
                                        extr.Richardson_extrapolation_step_standalone, 
                                        order=3, name="Richardson")

    def test_extrapolation_vector_upscaling_Anderson(self):
        self.extrapolation_vector_upscaling_framework(
                                        extr.Anderson_extrapolation_step, order=0, name="Anderson")
        self.extrapolation_vector_upscaling_framework(
                                        extr.Anderson_extrapolation_step, order=3, name="Anderson")

    def test_extrapolation_vector_upscaling_Aitken(self):
        self.extrapolation_vector_upscaling_framework(
                                        extr.Aitken_extrapolation_step, order=0, name="Aitken")
        self.extrapolation_vector_upscaling_framework(
                                        extr.Aitken_extrapolation_step, order=3, name="Aitken")

    def test_extrapolation_scalar_Anderson_disc(self):
        self.extrapolation_vector_upscaling_framework(
                                        extr.Anderson_extrapolation_step, order=0, name="Anderson",
                                        use_discontinuous=True, plot_global=False, plot_cells=False)
        self.extrapolation_vector_upscaling_framework(
                                        extr.Anderson_extrapolation_step, order=3, name="Anderson",
                                        use_discontinuous=True, plot_global=False, plot_cells=False)


    

    


    # def test_Richardson_estimate(self):
    #     self.common_extrapolation_testing_framework_2d("Richardson", rate=2)
    # def test_Anderson_estimate(self):
    #     self.common_extrapolation_testing_framework_2d("Anderson")
    # def test_RBS_estimate(self):
    #     self.common_extrapolation_testing_framework_2d("RBS")


    # def common_extrapolation_testing_framework_2d(self, name, rate=None):
    #     f = lambda x,y,dx,dy : ((-np.cos(2*np.pi*(x + dx/2)) + np.cos(2*np.pi*(x - dx/2)) ) * \
    #                     (-np.sin(2*np.pi*(y + dy/2)) + np.sin(2*np.pi*(y - dy/2)) ) / \
    #                     (dx*dy*4*np.pi*np.pi))
    #     order = 3
    #     gw = order
    #     upscale, _ = extr.get_upscale_nghosts(order)

    #     for N in [64, 128]:
    #         # coarse
    #         gc = Grid((1,N,N), gw)
    #         Xc,Yc = np.meshgrid(gc.x + 0.5*gc.dx, gc.y + 0.5*gc.dy)
    #         data_c = (f(Xc,Yc, gc.dx, gc.dy).T)[np.newaxis]

    #         # mid
    #         gm = Grid((1,2*N,2*N), gw)
    #         Xm,Ym = np.meshgrid(gm.x + 0.5*gm.dx, gm.y + 0.5*gm.dy)
    #         data_m = (f(Xm,Ym, gm.dx, gm.dy).T)[np.newaxis]

    #         # fine
    #         gf = Grid((1,4*N,4*N), gw)
    #         Xf,Yf = np.meshgrid(gf.x + 0.5*gf.dx, gf.y + 0.5*gf.dy)
    #         data_f = (f(Xf,Yf, gf.dx, gf.dy).T)[np.newaxis]

    #         # truth
    #         gt = Grid((1, 8*N, 8*N), gw)
    #         Xt, Yt = np.meshgrid(gt.x + 0.5*gt.dx, gt.y + 0.5*gt.dy)
    #         data_t = (f(Xt,Yt, gt.dx, gt.dy).T)[np.newaxis]

    #         data_f_upscale = extr.iterative_upscale(upscale, data_f, 1, gw)
        
    #         # choose an extrapolation operator
    #         op = extr.choose_algorithm(extr.extrap_dict[name])
    #         #... and extrapolate
    #         extrapolated = op(data_c, data_m, data_f, order=order, refinements=1, rate=3)

    #         extrap_error = dh.quick_L1_diff(extrapolated, data_t)
    #         real_error = dh.quick_L1_diff(data_f_upscale, data_t)

    #         print(f"Algorithm {name} for N={N}:")
    #         print(f"Did we do anything? {dh.quick_L1_diff(data_f_upscale, extrapolated)}")
    #         if extrap_error < real_error:
    #             print(f"Success! {extrap_error} < {real_error} ({round(100*extrap_error/real_error, 2)})")
    #         else:
    #             print(f"Failure... {extrap_error} > {real_error} ({round(100*extrap_error/real_error, 2)})")


if __name__ == '__main__':
    unittest.main()
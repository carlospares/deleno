#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from options import Options
import data_handler as dh
import encoding
import decoding

from extrapolation import iterative_upscale
from grid import Grid
import eno_tools as eno

options = Options() # parse command line
print(options.name)

if options.compare or options.details:
    f0, dhats = encoding.compressed_encoding(options)

if options.compare:
    # compare to real data
    dists = dh.compare_to_real(f0, options)
    dh.quick_conv_plot(dists, options)
    dh.save_plot("comp_real", options)

if options.details:
    # examine size of details
    details_L1size = []
    fine = dh.load_data(options.name)
    dh.quick_plot(fine[0], remove_ghost=options.gw)
    dh.save_plot("fine", options)
    dh.quick_plot(f0, remove_ghost=options.gw)
    dh.save_plot("base", options)
    for k, d in enumerate(dhats):
        dh.quick_plot(d, remove_ghost=options.gw)
        dh.save_plot(str(k), options)
        plt.clf()
        details_L1size.append(np.sum(np.abs(d))/ ((options.minN*(2**(k+1))))**2 )
    dh.quick_conv_plot(details_L1size, options, title="L1 norm of details")
    dh.save_plot("details_L1", options)

if options.extrapolate:
    dh.extrapolate_and_compare(options)

if options.compareupscale:
    dh.compare_upscale(options)

if options.checkregularity:
    dh.check_regularity(options)


# if "1024" not in options.name:
#     pass
# else:
# truth = dh.load_data(options.name)
# if options.order == 0:
#     upscale = lambda data, grid: eno.trivial_fv_2d_predictor(data, grid)
#     nghosts = 0
# else:
#     upscale = lambda data, grid: eno.fv_2d_predictor(data, options.order, grid)
#     nghosts = options.order

# for N in [64, 128, 256, 512]:
#     print(f"Comparing to real data, N={N}")
#     nrefs = int(np.log2(1024/N))
#     file = options.name.replace("1024", str(N))
#     data = dh.load_data(file)
#     data = iterative_upscale(upscale, data, nrefs, nghosts)
#     dh.quick_plot_compare(data[0], truth[0])


# decoded = decoding.compressed_decoding(f0, dhats, options)
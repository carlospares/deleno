#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from options import Options
import data_handler as dh
import encoding
import decoding

options = Options() # parse command line

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
    for k, d in enumerate(dhats):
        dh.quick_plot(d)
        dh.save_plot(str(k), options)
        plt.clf()
        details_L1size.append(np.sum(np.abs(d))/ ((options.minN*(2**(k+1))))**2 )
    dh.quick_conv_plot(details_L1size, options)
    dh.save_plot("details_L1", options)

if options.richardson:
    dh.richardson_and_compare(options)


# decoded = decoding.compressed_decoding(f0, dhats, options)
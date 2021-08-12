#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from options import Options
import data_handler as dh
from encoding import compressed_encoding_staggered
from decoding import compressed_decoding_staggered

options = Options() # parse command line
f0, dhats = compressed_encoding_staggered(options)

# details_L1size = []
# for k, d in enumerate(dhats):
#     # dh.quick_plot(d)
#     details_L1size.append(np.sum(np.abs(d))/ ((options.minN*(2**(k+1))))**2 )
# print(options.name)
# dh.quick_conv_plot(details_L1size, options)

if options.compare:
    dists = dh.compare_to_real(f0, options)
    dh.quick_conv_plot(dists, options)

# decoded = compressed_decoding_staggered(f0, dhats, options)
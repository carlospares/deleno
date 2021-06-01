#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from options import Options
import data_handler as dh
from encoding import compressed_encoding_staggered
from decoding import compressed_decoding_staggered

options = Options() # parse command line
f0, dhats = compressed_encoding_staggered(options)

decoded = compressed_decoding_staggered(f0, dhats, options)

dh.quick_plot_compare(dh.load_npy(options.name)[0], decoded)


# print([np.sum(np.abs(d)) for d in dhats])

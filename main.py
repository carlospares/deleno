#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from options import Options
import data_handler as dh
#import eno_tools as eno
#from grid import Grid
from encoding import compressed_encoding_staggered

options = Options() # parse command line
f0, dhats = compressed_encoding_staggered(options)

print([np.sum(np.abs(d)) for d in dhats])
dh.quick_plot(f0)


#data = dh.load_npy(options.name)
#grid = Grid(datashape=data.shape, nghosts=options.order, options=options)
#dh.quick_plot(data[1])

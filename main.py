#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from options import Options
import data_handler as dh
import eno_tools as eno

options = Options() # parse command line
data = dh.load_npy(options.name)
dh.quick_plot(data[1])
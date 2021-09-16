import numpy as np
from boundary_condition import BoundaryCondition as BC

class Grid:
    def __init__(self, datashape, nghosts, options=None, limits=[0., 1., 0., 1.]):
        """initializes a grid with basic information"""

        # domain limits
        self.xmin = limits[0]
        self.xmax = limits[1]
        self.ymin = limits[2]
        self.ymax = limits[3]

        # nx : number of cells in x axis (ie #cols), i in range(nx); ~x
        # ny : number of cells in y axis (ie #rows), j in range(ny); ~y
        # data accessed like data[i,j]
        if len(datashape) == 3: # velocity pair
            self.nx = int(datashape[1])
            self.ny = int(datashape[2])
        elif len(datashape) == 2: # single variable
            self.nx = int(datashape[0])
            self.ny = int(datashape[1])
        self.gw = nghosts
        self.bcs = BC() if options is None else BC(options.bcs)
        self.generate_params()

    def generate_params(self):
        """ derives all non-input parameters """

        self.nxg = self.nx + 2*self.gw
        self.nyg = self.ny + 2*self.gw
        self.dx = (self.xmax-self.xmin)/self.nx
        self.dy = (self.ymax-self.ymin)/self.ny
        
        # cell interfaces (left/bottom):
        self.x = self.dx*np.arange(self.nx)
        self.y = self.dy*np.arange(self.ny)



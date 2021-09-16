import argparse
from extrapolation import extrap_dict
import sys

STAGGERED = 0
FV = 1

class Options:

    def is_grid_staggered():
        return self.grid == STAGGERED
    def is_grid_fv():
        return self.grid == FV

    def __init__(self):
        parser = argparse.ArgumentParser(description='Computes details as in DeLENO')
        parser.add_argument('-n', '--name', type=str, required=True, 
                                help='Name of file to be computed')
        parser.add_argument('-minN', '--minN', type=int, required=True, 
                                help='Number of cells in one direction (coarsest)')
        parser.add_argument('-maxN', '--maxN', type=int, required=True, 
                                help='Number of cells in one direction (finest)')
        parser.add_argument('-g', '--gridtype', type=str, required=True,
                                help='Type of grid (one of "staggered", "fv"')
        parser.add_argument('-bc', '--bc', type=str, nargs=4, default=['P','P','P','P'],
                                help='Type of BCs (default: periodic), in order N,S,W,E')
        parser.add_argument('-o', '--order', type=int, default=3,
                                help='Order of reconstruction/interpolation')
        parser.add_argument('-c', '--compare', action='store_true', default=False,
                                help="Compare decimator to real samples")
        parser.add_argument('-d', '--details', action='store_true', default=False,
                                help="Plot size of details and details")
        parser.add_argument('-e', '--extrapolate', type=str, default=None,
                                help="Compute extrapolation; see extrapolation.py for values")
        parser.add_argument('-v', '--verbose', action='store_true', default=False,
                                help='Enable more detailed output (default=false)')
        parser.add_argument('-p', '--prefix', type=str, required=False, 
                                help='Prefix this at end of output files')
        parser.add_argument('-eps', '--epsilon', type=float, default=0.,
                                help='Threshold for compression')
        args = parser.parse_args()
        self.name = args.name
        self.minN = args.minN
        self.maxN = args.maxN
        self.order = args.order
        self.bcs = args.bc
        self.compare = args.compare
        self.details = args.details
        self.verbose = args.verbose
        self.epsilon = args.epsilon
        self.prefix = args.prefix
        self.gw = self.order
        self.extrapolate = args.extrapolate is not None
        if self.extrapolate:
            if args.extrapolate in extrap_dict:
                self.extrap_which = extrap_dict[args.extrapolate]
            else:
                print(f"Type of extrapolation {args.extrapolate} not known!")
                print(f"Please choose one of the following: {[k for k in extrap_dict.keys()]}")
                sys.exit(1)

        if args.gridtype.lower() == "staggered":
            self.grid = STAGGERED
        elif args.gridtype.lower() == "fv":
            self.grid = FV
        else:
            raise Exception(f"Type of grid {self.gridtype} unknown!")

        # some basic sanity checks
        if self.minN >= self.maxN:
            raise Exception("minN >= maxN!")
        if self.epsilon <= 0 and (self.compare or self.details):
            print("Epsilon is not positive; data will not be compressed. Is this what you want?")
        return

import argparse

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
        parser.add_argument('-R', '--Richardson', action='store_true', default=False,
                                help="Compute Richardson extrapolation, compare to ground truth")
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
        self.richardson = args.Richardson


        if args.gridtype.lower() == "staggered":
            self.grid = STAGGERED
        elif args.gridtype.lower() == "fv":
            self.grid = FV
        else:
            raise Exception(f"Type of grid {self.gridtype} unknown!")

        # some basic sanity checks
        if self.minN >= self.maxN:
            raise Exception("minN >= maxN, no details can be computed!")
        if self.epsilon <= 0:
            print("Epsilon is not positive; data will not be compressed. Is this what you want?")
        return

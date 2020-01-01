from __future__ import print_function
#import warnings
#warnings.filterwarnings("ignore",category=DeprecationWarning)

import numpy
import sys
import os
import h5py
import argparse
from irbasis_util.four_point import from_PH_convention

from irbasis_util.gf import LocalGf2CP
from irbasis_util.solver import FourPointBasisTransform, mpi_split
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
is_master_node = comm.Get_rank() == 0

parser = argparse.ArgumentParser(
    prog='multiply_PH.py',
    description='tensor regression code.',
    epilog='end',
    usage='$ ',
    add_help=True)
parser.add_argument('path_input_file_left', action='store', default=None, type=str, help="name of file containing left gf")
parser.add_argument('path_input_file_right', action='store', default=None, type=str, help="name of file containing right gf")
parser.add_argument('path_output_file', action='store', default=None, type=str, help="output file name.")
parser.add_argument('--niter', default=20, type=int, help='Number of iterations')
parser.add_argument('--D', default=1, type=int, help='Rank of decomposition')
parser.add_argument('--rtol', default=1e-8, type=float, help='rtol')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--n_max_inner', default=100, type=int, help='Number of inner frequency points')
parser.add_argument('--scut', default=1e-4, type=float, help='Cutoff value for singular values')
parser.add_argument('--alpha', default=1e-8, type=float, help='regularization parameter')

args = parser.parse_args()
if os.path.isfile(args.path_input_file_left) is False:
    print("Input file is not exist.")
    sys.exit(-1)
if os.path.isfile(args.path_input_file_right) is False:
    print("Input file is not exist.")
    sys.exit(-1)

D = args.D

with h5py.File(args.path_input_file_left, 'r') as hf:
    gf_left = LocalGf2CP.load(hf, '/D{}'.format(D))

with h5py.File(args.path_input_file_right, 'r') as hf:
    gf_right = LocalGf2CP.load(hf, '/D{}'.format(D))

assert gf_left.Lambda == gf_right.Lambda
assert gf_left.beta == gf_right.beta

beta = gf_left.beta
wmax = gf_left.wmax
Lambda = gf_left.Lambda

if rank == 0:
    print("Lambda = ", Lambda)
    print("beta = ", beta)

transform = FourPointBasisTransform(beta, wmax, scut=args.scut, comm=comm)

gf = transform.multiply_LocalGf2CP_PH(gf_left, gf_right, args.n_max_inner, D, args.niter, args.rtol, args.alpha)

if is_master_node:
    with h5py.File(args.path_output_file, 'a') as hf:
        gf.save(hf, '/D'+str(D))

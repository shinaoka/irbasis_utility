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
    prog='regression_four_point_cthyb.py',
    description='tensor regression code.',
    epilog='end',
    usage='$ ',
    add_help=True)
parser.add_argument('path_input_file', action='store', default=None, type=str, help="input file name.")
parser.add_argument('path_output_file', action='store', default=None, type=str, help="output file name.")
parser.add_argument('--niter', default=20, type=int, help='Number of iterations')
parser.add_argument('--D', default=1, type=int, help='Rank of decomposition')
parser.add_argument('--Lambda', default=1000.0, type=float, help='Lambda')
parser.add_argument('--rtol', default=1e-8, type=float, help='rtol')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--nesterov', default=False, action='store_true', help='nesterov')
parser.add_argument('--alpha', default=1e-8, type=float, help='regularization parameter')
parser.add_argument('--scut', default=1e-4, type=float, help='Cutoff value for singular values')
parser.add_argument('--vertex', default=False, action='store_true', help='Vertex or not')
parser.add_argument('--restart', default=False, action='store_true', help='Restart')

args = parser.parse_args()
if os.path.isfile(args.path_input_file) is False:
    print("Input file is not exist.")
    sys.exit(-1)

Lambda = args.Lambda
D = args.D

with h5py.File(args.path_input_file, 'r') as hf:
    freqs_PH = hf['/G2/matsubara/freqs_PH'][()]
    n_freqs = freqs_PH.shape[0]
    data = hf['/G2/matsubara/data'][()].reshape((-1,n_freqs,2))
    beta = hf['/parameters/model.beta'][()]
    G2iwn = data[:,:,0] + 1J * data[:,:,1]
    num_o = data.shape[0]
wmax = Lambda/beta

if rank == 0:
    print("Lambda = ", Lambda)
    print("beta = ", beta)
    print("nesterov = ", args.nesterov)
    print("restart = ", args.restart)

# From PH to fermionic convention
freqs = []
for i in range(n_freqs):
    freqs.append(from_PH_convention(freqs_PH[i,:]))


sizes, offsets = mpi_split(n_freqs, comm.size)
n_freqs_local = sizes[rank]
start, end = offsets[rank], offsets[rank]+sizes[rank]
G2iwn_local = G2iwn[:, start:end]

sp_local = numpy.array(freqs)[start:end, :]
sp_local = [tuple(sp_local[i,:]) for i in range(sp_local.shape[0])]

transform = FourPointBasisTransform(beta, wmax, scut=args.scut, comm=comm, sp_F=sp_local)

basis = {True: transform.basis_vertex, False: transform.basis_G2}[args.vertex]
Nl = basis.Nl

if rank == 0:
    print("Num of freqs = ", n_freqs)
    print("Vertex = ", args.vertex)
    print("Nl = ", Nl)

if args.restart:
    if is_master_node:
        print("Reading ", args.path_output_file)
    with h5py.File(args.path_output_file, 'r') as hf:
        gf = LocalGf2CP.load(hf, '/D'+str(D))
else:
    gf = LocalGf2CP(Lambda, Nl, num_o, D, None, args.vertex)

# Regression
transform.fit_LocalGf2CP(gf, G2iwn_local.transpose(), args.niter, not args.restart, args.rtol, args.alpha, 1, args.seed)

if is_master_node:
    with h5py.File(args.path_output_file, 'a') as hf:
        gf.save(hf, '/D'+str(D))

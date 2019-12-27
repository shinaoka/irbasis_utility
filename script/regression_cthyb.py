from __future__ import print_function

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import numpy
import sys
from itertools import *
import h5py
import copy
import argparse
import irbasis
import matplotlib.pylab as plt
from irbasis_util.four_point_ph_view import FourPointPHView

#from irbasis_util.internal import *
#from irbasis_util.regression import *

from irbasis_util.tensor_regression_auto_als import fit, predict
from mpi4py import MPI 

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
is_master_node = comm.Get_rank() == 0

def mpi_split(work_size, comm_size):
    base = work_size // comm_size
    leftover = int(work_size % comm_size)

    sizes = numpy.ones(comm_size, dtype=int) * base
    sizes[:leftover] += 1

    offsets = numpy.zeros(comm_size, dtype=int)
    offsets[1:] = numpy.cumsum(sizes)[:-1]

    return sizes, offsets

parser = argparse.ArgumentParser(
    prog='regression_cthyb.py',
    description='tensor regression code.',
    epilog='end',
    usage='$ ',
    add_help=True)
parser.add_argument('path_input_file', action='store', default=None, type=str, help="input file name.")
parser.add_argument('path_output_file', action='store', default=None, type=str, help="output file name.")
parser.add_argument('--bfreq', default=0, type=int, help='Bosonic frequnecy')
parser.add_argument('--niter', default=20, type=int, help='Number of iterations')
parser.add_argument('--D', default=1, type=int, help='Rank of decomposition')
parser.add_argument('--Lambda', default=0, type=float, help='Lambda')
parser.add_argument('--beta', default=0, type=float, help='beta')
parser.add_argument('--nesterov', default=False, action='store_true', help='nesterov')
parser.add_argument('--alpha', default=1e-8, type=float, help='regularization parameter')
parser.add_argument('--scut', default=1e-4, type=float, help='Cutoff value for singular values')
parser.add_argument('--vertex', default=False, action='store_true', help='Vertex or not')
parser.add_argument('--restart', default=False, action='store_true', help='Restart')

args = parser.parse_args()
if os.path.isfile(args.path_input_file) is False:
    print("Input file is not exist.")
    sys.exit(-1)
boson_freq = args.bfreq

beta = args.beta
Lambda = args.Lambda

with h5py.File(args.path_input_file, 'r') as hf:
    freqs_PH_all = hf['/G2/matsubara/freqs_PH'][()]
    nfreqs_all = len(freqs_PH_all)
    data = hf['/G2/matsubara/data'][()].reshape((-1,nfreqs_all,2))
    data = data[:,:,0] + 1J * data[:,:,1]
    num_o = data.shape[0]

if rank == 0:
    print("Lambda = ", Lambda)
    print("beta = ", beta)
    print("nesterov = ", args.nesterov)
    print("restart = ", args.restart)

n_freqs = numpy.sum([freqs_PH_all[i,2]==boson_freq for i in range(freqs_PH_all.shape[0])])

if n_freqs == 0:
    print("Data for boson_freq{} was not found.".format(boson_freq))
    sys.exit(-1)
idx = freqs_PH_all[:,2] == boson_freq
freqs_PH = freqs_PH_all[idx,:]
G2iwn = data[:,:,:,:,idx]

# Find active orbital components
tmp = numpy.sqrt(numpy.sum(numpy.abs(G2iwn)**2, axis=-1)).ravel()
orb_idx = tmp/numpy.amax(tmp) > 1e-5
num_o_nonzero = numpy.sum(orb_idx)
if rank == 0:
    print("Num of active orbital components = ", num_o_nonzero)

sizes, offsets = mpi_split(n_freqs, comm.size)
n_freqs_local = sizes[rank]
start, end = offsets[rank], offsets[rank]+sizes[rank]
G2iwn_local = G2iwn[:,:,:,:,start:end]

G2iwn_dict = {}
for i in range(n_freqs):
    G2iwn_dict[(freqs_PH[i,0],freqs_PH[i,1])] = G2iwn[:,:,:,:,i]

wmax = Lambda / beta

phb = FourPointPHView(boson_freq, Lambda, beta, args.scut, True)
Nl = phb.Nl
sp = [tuple(freqs_PH[i,:2]) for i in range(freqs_PH.shape[0])]

sp_local = numpy.array(sp)[start:end,:]
sp_local = [(sp_local[i,0], sp_local[i,1]) for i in range(sp_local.shape[0])]

# Regression
def kruskal_complex_Ds(tensors_A, y, Ds, cutoff=1e-5):
    """
    
    Parameters
    ----------
    A
    y

    Returns
    -------
    

    """

    coeffs_D = []
    squared_errors_D = []
    model_D = []
    Nw, Nr, linear_dim = tensors_A[0].shape
    alpha_init = 0
    y = y.reshape((-1, num_o))

    for i, D in enumerate(Ds):
        if rank == 0:
            print("D ", D)
        model = OvercompleteGFModel(Nw, Nr, 2, num_o_nonzero, linear_dim, tensors_A, y[:, orb_idx], alpha_init, D)
        info = optimize_als(model, args.niter, tol_rmse = 1e-8, optimize_alpha=1e-8, verbose = 1, print_interval=1)
        xs = copy.deepcopy(model.x_tensors())
        x_orb_full = numpy.zeros((D, num_o), dtype=complex)
        x_orb_full[:, orb_idx] = xs[-1]
        xs[-1] = x_orb_full
        coeffs_D.append(xs)
        squared_errors_D.append(info['rmses'][-1]**2)
        model_D.append(model)
    
    squared_errors_D = numpy.array(squared_errors_D)
    
    return coeffs_D, squared_errors_D, model_D

def construct_prj(sp):
    n_sp = len(sp)
    prj = phb.projector_to_matsubara_vec(sp, decomposed_form=True)
    for i in range(2):
        prj[i] = prj[i].reshape((n_sp, 12, Nl))
        
    return prj

prj = construct_prj(sp_local)

Ds = [args.D]
y = G2iwn_local.transpose((1,0))
coeffs_D, se_D, model_D = kruskal_complex_Ds(prj, y, Ds)

if is_master_node:
    with h5py.File(args.path_output_file, 'a') as hf:
        for i, D in enumerate(Ds):
            if '/D'+ str(D) in hf:
                del hf['/D'+ str(D)]
            for i, x in enumerate(coeffs_D[i]):
                hf['/D'+ str(D) + '/x' + str(i)] = x

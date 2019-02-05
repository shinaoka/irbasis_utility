import numpy
import sys
from itertools import *
import h5py
import argparse
import irbasis
import matplotlib.pylab as plt
from irbasis_util.four_point_ph_view import *
from irbasis_util.internal import *
from irbasis_util.regression import *

from irbasis_util.tensor_regression import *
from mpi4py import MPI 

#enable_MPI() #for tensor_regression

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

args = parser.parse_args()
if os.path.isfile(args.path_input_file) is False:
    print("Input file is not exist.")
    sys.exit(-1)
boson_freq = args.bfreq
print("boson_freq", boson_freq)

with h5py.File(args.path_input_file, 'r') as hf:
    G1_IR = hf["G1_IR"].value[:,:,:,0] + 1J * hf["G1_IR"].value[:,:,:,1]
    Lambda = hf["/parameters/measurement.Lambda"].value
    beta = hf["/parameters/model.beta"].value
    nflavors = hf["/parameters/model.sites"].value * hf["/parameters/model.spins"].value

    data = hf['/G2/matsubara/data'].value[:,:,:,:,:,0] + 1J * hf['/G2/matsubara/data'].value[:,:,:,:,:,1]
    data = data.reshape((nflavors,nflavors,nflavors,nflavors,-1))
    freqs_PH_all = hf['/G2/matsubara/freqs_PH'].value

Nl = G1_IR.shape[2]

basis = irbasis.load('F', Lambda)

if is_master_node:
    plt.figure(1)
    plt.plot(numpy.abs(G1_IR[0,0,:]), marker='o')
    plt.grid()
    plt.yscale("log")
    plt.savefig("G1IR.pdf")
    plt.close()

    plt.figure(1)
    min_n = -100
    max_n = 100
    Unl = (numpy.sqrt(beta) * basis.compute_unl(numpy.arange(min_n, max_n+1)))[:,0:Nl]
    Giw = numpy.dot(Unl, G1_IR[0,0,:])
    n = numpy.arange(min_n, max_n+1)
    plt.plot(n, Giw.imag, marker='x')
    plt.savefig("Giwn.pdf")
    plt.close()

n_freqs = numpy.sum([freqs_PH_all[i,2]==boson_freq for i in range(freqs_PH_all.shape[0])])

if n_freqs == 0:
    print("Data for boson_freq{} was not found.".format(boson_freq))
    sys.exit(-1)
idx = freqs_PH_all[:,2] == boson_freq
freqs_PH = freqs_PH_all[idx,:]
G2iwn = data[:,:,:,:,idx]

sizes, offsets = mpi_split(n_freqs, comm.size)
n_freqs_local = sizes[rank]
start, end = offsets[rank], offsets[rank]+sizes[rank]
G2iwn_local = G2iwn[:,:,:,:,start:end]

G2iwn_dict = {}
for i in range(n_freqs):
    G2iwn_dict[(freqs_PH[i,0],freqs_PH[i,1])] = G2iwn[:,:,:,:,i]

if is_master_node:
    plt.figure(1)
    plt.plot(freqs_PH[:,0], freqs_PH[:,1], ls='', marker='o')
    plt.savefig("freqs_PH.pdf")
    plt.close()

wmax = Lambda / beta

phb = FourPointPHView(boson_freq, Lambda, beta, 1e-4, True)
Nl = phb.Nl
sp = [tuple(freqs_PH[i,:2]) for i in range(freqs_PH.shape[0])]

sp_local = numpy.array(sp)[start:end,:]
sp_local = [(sp_local[i,0], sp_local[i,1]) for i in range(sp_local.shape[0])]

# Regression
def kruskal_complex_Ds(tensors_A, y, Ds):
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
    print(Nw, Nr, linear_dim)
    alpha_init = 0
    for i, D in enumerate(Ds):
        print("D ", D, y.shape)
        #print("debugA", y)
        #print("debugB", tensors_A)
        model = OvercompleteGFModel(Nw, Nr, 2, nflavors**4, linear_dim, tensors_A, y.reshape((-1,nflavors**4)),
                                    alpha_init, D)
        info = optimize_als(model, args.niter, tol_rmse = 1e-6,
                            optimize_alpha=1e-8, verbose = 1, print_interval=1)
        coeffs = model.full_tensor_x()
        coeffs_D.append(coeffs)
        e = model.mse()
        print("D = ", D, e, " num_ite", len(info['losss']))
        squared_errors_D.append(e)
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
y = G2iwn_local.transpose((-1,0,1,2,3))
coeffs_D, se_D, model_D = kruskal_complex_Ds(prj, y, Ds)

with h5py.File(args.path_output_file, 'a') as hf:
    for i, D in enumerate(Ds):
        if '/D'+ str(D) in hf:
            del hf['/D'+ str(D)]
        hf['/D'+ str(D)] = coeffs_D[i]

if is_master_node:
    plt.figure(1)
    up, down = 0, 1
    y_predict = model_D[-1].predict_y().reshape((-1,nflavors,nflavors,nflavors,nflavors))
    plt.plot(numpy.abs(y[:,up,up,down,down]), c='r', ls='', marker='x')
    plt.plot(numpy.abs(y_predict[:,up,up,down,down]), c='r')
    plt.yscale("log")
    plt.xlim([0,100])
    plt.savefig("fit.pdf")

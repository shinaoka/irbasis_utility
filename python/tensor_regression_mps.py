from __future__ import print_function

import sys
import copy
import autograd.numpy as numpy  # Thinly-wrapped numpy
from autograd import grad  # The only autograd function you may ever need
from autograd.misc.optimizers import adam

#import jax.numpy as numpy
#from jax import grad  # The only autograd function you may ever need

from itertools import compress
import time

is_enabled_MPI = False

def enable_MPI():
    global is_enabled_MPI
    global MPI

    is_enabled_MPI = True
    _temp = __import__('mpi4py', globals(), locals(), ['MPI'], 0)
    MPI = _temp.MPI

def mpi_split(work_size, comm_size):
    base = work_size // comm_size
    leftover = int(work_size % comm_size)

    sizes = numpy.ones(comm_size, dtype=int) * base
    sizes[:leftover] += 1

    offsets = numpy.zeros(comm_size, dtype=int)
    offsets[1:] = numpy.cumsum(sizes)[:-1]

    return sizes, offsets


def squared_L2_norm(x):
    """
    Squared L2 norm
    """
    # CAUTION: autograd wrongly handle numpy.linalg.norm for a complex array
    x = x.ravel()
    return numpy.real(numpy.dot(numpy.conj(x.transpose()), x))


def predict_slow(prj, x_tensors, path=None):
    assert isinstance(x_tensors, list)

    freq_dim = len(prj)
    if freq_dim == 2:
        subscripts = 'wrl,wrm, Ar,ABl,BCm,Co -> wo'
    elif freq_dim == 3:
        subscripts = 'wrl,wrm,wrn, Ar,ABl,BCm,CDn,Do -> wo'
    else:
        raise RuntimeError("Invalid freq_dim")
        #subscripts = 'wrl,wrm, Dr, LMl, Mm'

    assert len(x_tensors) == freq_dim+2

    #print(type(prj), type(x_tensors))
    operands = prj + x_tensors
    if path is None:
        from numpy import einsum_path
        path, string_repr = einsum_path(subscripts, *operands, optimize=('greedy', 1E+18))
        print(string_repr)


    t1 = time.time()
    t = numpy.einsum(subscripts, *operands, optimize=False)
    t2 = time.time()
    print(t2-t1, path)
    return t, path

def predict(prj, x_tensors, path=None):
    assert isinstance(x_tensors, list)

    freq_dim = len(prj)
    assert len(x_tensors) == freq_dim+2

    #subscripts = 'wrl,wrm, Ar,ABl,BCm,Co -> wo'

    # Ux contains (wrAB, wrBC)
    Ux = []
    for i in range(freq_dim):
        # (wrl, ABl) -> (wrAB)
        Ux.append(numpy.tensordot(prj[i], x_tensors[i+1], axes=([2],[2])))

    # (Ar, wrAB)->(wrAB)
    tmp = numpy.transpose(x_tensors[0])[None,:,:,None] * Ux[0]

    # wrAB -> wrB
    Ux[0] = numpy.sum(tmp, axis=2)

    # (wrB) * (wrBC) -> (wrC)
    tmp = numpy.sum(Ux[0][:,:,:,None] * Ux[1], axis=2)
    if freq_dim==3:
        # (wrC) * (wrCD) -> (wrD) [only if freq_dim==3]
        tmp = numpy.sum(tmp[:,:,:,None] * Ux[2], axis=2)

    # wrC -> wC
    tmp2 = numpy.sum(tmp, axis=1)

    #(wC, Co) -> wo
    tmp3 = numpy.dot(tmp2, x_tensors[-1])

    return tmp3, None


def fit(y, prj, D, nite, verbose=0, random_init=True, x0=None, optimize_alpha=-1, comm=None, seed=1, method='L-BFGS-B'):
    """
    Alternating least squares

    Parameters
    ----------
    model:  object of OvercompleteGFModel
        Model to be optimized

    nite: int
        Number of max iterations.

    rtol: float
        Stopping condition of optimization.
        Iterations stop if abs(||r_k|| - ||r_{k-1}||) < rtol * ||y||
        r_k is the residual vector at iteration k.

    verbose: int
        0, 1, 2

    min_norm: float
        The result is regarded as 0 if the norm drops below min_norm during optimization.

    optimize_alpha: float
        If a positive value is given, the value of the regularization parameter alpha is automatically determined.
        (regularization term)/(squared norm of the residual) ~ optimize_alpha.

    Returns
    -------

    """
    from scipy.optimize import minimize
    from numpy import random

    rank = 0
    if comm is None and is_enabled_MPI:
        comm = MPI.COMM_WORLD

    if is_enabled_MPI:
        rank = comm.Get_rank()
        num_proc = comm.Get_size()
    else:
        num_proc = 1

    freq_dim = len(prj)
    num_w, num_rep, linear_dim = prj[0].shape
    num_o = y.shape[1]

    # Over all processess
    #num_w_all = comm.allreduce(num_w) if is_enabled_MPI else num_w

    assert y.shape[0] == num_w
    assert y.shape[1] == num_o

    def _init_random_array(*shape):
        return random.rand(*shape) + 1J * random.rand(*shape)

    # Init x_tensors
    x_shapes = [(D,num_rep)] + freq_dim*[(D,D,linear_dim)] + [(D,num_o)]
    x_sizes = [numpy.prod(shape) for shape in x_shapes]
    if random_init:
        assert x0 is None
        random.seed(seed)
        x_tensors = [_init_random_array(*shape) for shape in x_shapes]
    elif not x0 is None:
        x_tensors = copy.deepcopy(x0)
    else:
        x_tensors = [numpy.zeros(*shape, dtype=complex) for shape in x_shapes]
    if is_enabled_MPI:
        for i in range(len(x_tensors)):
            x_tensors[i] = comm.bcast(x_tensors[i], root=0)

    # Init alpha
    alpha = 0.0

    # Optimal construction path
    _, path = predict(prj, x_tensors, None)

    def _to_x_tensors(x):
        tmp = x.reshape((2, -1))
        x1d = tmp[0, :] + 1J * tmp[1, :]
        offset = 0
        x_tensors = []
        for i in range(len(x_shapes)):
            block = x1d[offset:offset+x_sizes[i]]
            x_tensors.append(block.reshape(x_shapes[i]))
            offset += x_sizes[i]
        return x_tensors

    def _from_x_tensors(x_tensors):
        x1d = numpy.hstack([numpy.ravel(x) for x in x_tensors])
        return numpy.hstack((x1d.real, x1d.imag))

    def _se_local(x_tensors):
        y_pre, _ = predict(prj, x_tensors, path)
        return squared_L2_norm(y - y_pre)

    def _snorm_local(x_tensors):
        return numpy.sum(numpy.array([squared_L2_norm(t) for t in x_tensors]))

    def _se_mpi(x_tensors):
        if is_enabled_MPI:
            return comm.allreduce(_se_local(x_tensors))
        else:
            return _se_local(x_tensors)

    def _snorm_mpi(x_tensors):
        if is_enabled_MPI:
            return comm.allreduce(_snorm_local(x_tensors))
        else:
            return _snorm_local(x_tensors)

    #def _loss(x_tensors):
        #"""
        #Cost function
        #"""
        #return _se_mpi(x_tensors) + alpha * _snorm_mpi(x_tensors)

    def loss_local(x):
        x_tensors = _to_x_tensors(x)
        return _se_local(x_tensors) + alpha * _snorm_local(x_tensors) / num_proc

    grad_loss_local = grad(loss_local)
        
    def loss(x):
        if is_enabled_MPI:
            return comm.allreduce(loss_local(x))
        else:
            return loss_local(x)

    def grad_loss(x):
        if is_enabled_MPI:
            return comm.allreduce(grad_loss_local(x))
        else:
            return grad_loss_local(x)

    def callback(xk):
        if optimize_alpha > 0:
            x_tensors = _to_x_tensors(xk)
            reg = _snorm_mpi(x_tensors)
            se = _se_mpi(x_tensors)
            alpha = optimize_alpha * se/reg
            if rank == 0:
                #print("debug ", rank, numpy.linalg.norm(x_tensors[0]))
                print("alpha = ", alpha)
        sys.stdout.flush()

    x0 = _from_x_tensors(x_tensors)
    t1 = time.time()
    loss0 = loss(x0)
    t2 = time.time()
    grad_loss0 = grad_loss(x0)
    t3 = time.time()
    if rank == 0:
        print("timings : ", t2-t1, t3-t2)

    if method == 'L-BFGS-B':
        res = minimize(fun=loss, x0=x0, jac=grad_loss, method="L-BFGS-B", options={'maxiter': nite, 'disp': rank==0},
                   callback=callback)
        return _to_x_tensors(res.x)
    elif method == 'adam':
        grad_tmp = lambda x, i : grad_loss(x)
        def callback_tmp(x, i, g):
            if i % 10 == 0:
                print('Iteration ', i, loss(x))
        return _to_x_tensors(adam(grad_tmp, x0, callback=callback_tmp, num_iters=nite))
    else:
        raise RuntimeError("Unknown method " + method)


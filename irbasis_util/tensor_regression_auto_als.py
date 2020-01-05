from __future__ import print_function

import sys
import numpy
import copy

#from itertools import compress
import time
from .tensor_network import Tensor, TensorNetwork
from .auto_als import AutoALS


def predict(prj, x_tensors):
    xs_l = x_tensors[1:-1]
    freq_dim = len(prj)
    t1 = time.time()
    nw, R, D = prj[0].shape[0], prj[0].shape[1], xs_l[0].shape[0]
    num_o = x_tensors[-1].shape[1]

    tmp_wrd = numpy.full((nw, R, D), complex(1.0))
    # O(Nw R D Nl)
    for i in range(freq_dim):
        tmp_wrd *= numpy.einsum('wrl, dl -> wrd', prj[i], xs_l[i], optimize=True)
    t2 = time.time()
    # O(Nw R D)
    tmp_wd = numpy.einsum('wrd, dr->wd', tmp_wrd, x_tensors[0], optimize=True)
    t3 = time.time()
    # O(Nw D No)
    tmp_wo = numpy.einsum('wd, do->wo', tmp_wd, x_tensors[-1], optimize=True).reshape((nw, num_o))
    t4 = time.time()
    #print(t2-t1, t3-t2, t4-t3)
    return tmp_wo

def fit(y, prj, D, nite,  rtol = 1e-3, verbose=0, random_init=True, x0=None, alpha=1e-8, comm=None, seed=1, nesterov=True):
    """
    Alternating least squares with L2 regularization


    Parameters
    ----------
    y:  2D ndarray of complex numbers
        Data to be fitted.
        The shape must be (number of sampling frequencies, number of spin-orbital components).
        When MPI is used, sampling frequencies are distributed to MPI processes.

    prj:
        Tensor network representation of coefficient matrix

    nite: int
        Number of max iterations.

    rtol: float
        Stopping condition of optimization.

    verbose: int
        0, 1, 2

    seed: int
        Seed for random number generator

    comm:
        Can be MPI.COMM_WORLD

    nesterov: bool
        Use Nesterov's acceleration


    Returns
    -------
    x_tensors: list of ndarray
        CPD tensors
    """

    num_w, num_rep, linear_dim = prj[0].shape
    num_o = y.shape[1]

    assert y.shape[0] == num_w
    assert y.shape[1] == num_o

    Y_tn = TensorNetwork([Tensor('Y', (num_w, num_o))], [(0,1)], external_subscripts={0,1})

    return fit_impl(Y_tn, {'Y': y}, prj, D, nite,  rtol, verbose, random_init, x0, alpha, comm, seed, nesterov)


def fit_impl(Y_tn, Y_tensors, prj, D, nite,  rtol = 1e-3, verbose=0, random_init=True, x0=None, alpha=1e-8, comm=None, seed=1, nesterov=True):
    """

    Parameters
    ----------
    Y_tn: TensorNetwork
        Tensor network to be fitted.
        The shape is (num_w, num_o) [num_w = number of frequencies, num_o = num of spin-orbital components].

    Y_tensors: dict
        Actual data of Y

    The others are the same as those of fit().

    """

    if comm is None:
        rank = 0
        num_proc = 1
        is_enabled_MPI = False
    else:
        import mpi4py
        rank = comm.Get_rank()
        num_proc = comm.Get_size()
        is_enabled_MPI = True

    freq_dim = len(prj)
    num_w, num_rep, linear_dim = prj[0].shape
    num_o = Y_tn.shape[1]

    assert Y_tn.shape[0] == num_w
    assert Y_tn.shape[1] == num_o

    def _init_random_array(*shape):
        return numpy.random.rand(*shape) + 1J * numpy.random.rand(*shape)

    tensors_value = {}

    # Tensor network to be fitted
    tensors_value.update(Y_tensors)

    ## Tensor network to fit by
    # Fitting parameters for representations
    tY_tensors = [Tensor('x0', (D, num_rep))]
    tY_subs = [(20, 10)]
    target_tensors = [tY_tensors[-1]]

    # Define projectors
    for i in range(freq_dim):
        name = 'U{}'.format(i+1)
        tY_tensors.append(Tensor(name, (num_w, num_rep, linear_dim)))
        tY_subs.append((0, 10, 30+i))
        tensors_value[name] = prj[i]

    # Fitting parameters for IR
    for i in range(freq_dim):
        name = 'x{}'.format(i+1)
        tY_tensors.append(Tensor(name, (D, linear_dim)))
        tY_subs.append((20, 30+i))
        target_tensors.append(tY_tensors[-1])

    # Fitting parameters for orbital sector
    name = 'x{}'.format(freq_dim+1)
    tY_tensors.append(Tensor(name, (D, num_o)))
    tY_subs.append((20, 1))
    target_tensors.append(tY_tensors[-1])

    tY_tn = TensorNetwork(tY_tensors, tY_subs, external_subscripts={0,1})

    # Init x_tensors
    x_shapes = [(D, num_rep)] + freq_dim * [(D, linear_dim)] + [(D, num_o)]
    if random_init:
        assert x0 is None
        numpy.random.seed(seed)
        x_tensors = [_init_random_array(*shape) for shape in x_shapes]
    elif not x0 is None:
        x_tensors = copy.deepcopy(x0)
    else:
        x_tensors = [numpy.zeros(shape, dtype=complex) for shape in x_shapes]

    # Insert fitting parameters
    for i in range(freq_dim + 2):
        name = 'x{}'.format(i)
        tensors_value[name] = x_tensors[i]
        if is_enabled_MPI:
            tensors_value[name][:] = comm.bcast(tensors_value[name], root=0)

    distributed_subscript = 0 if is_enabled_MPI else None
    als = AutoALS(Y_tn, tY_tn, target_tensors, reg_L2=alpha, comm=comm, distributed_subscript=distributed_subscript)
    als.fit(nite, tensors_value, rtol=rtol, nesterov=nesterov, verbose=verbose)

    for i in range(freq_dim + 2):
        name = 'x{}'.format(i)
        x_tensors[i][:] = tensors_value[name]

    return x_tensors
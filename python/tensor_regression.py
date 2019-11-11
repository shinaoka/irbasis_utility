from __future__ import print_function

import sys
import numpy
from scipy.sparse.linalg import LinearOperator

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
    return numpy.linalg.norm(x)**2

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

def linear_operator_r(N1, N2, tensors_A, x_r, xs_l, x_orb):
    """
    Linear operator of the coefficient matrix A(W, O; d, r) (for optimizing x^{(0)}
    """
    num_w, R, linear_dim = tensors_A[0].shape
    D = x_r.shape[0]
    num_o = x_orb.shape[1]

    freq_dim = len(tensors_A)
    # O(Nw D R Nl)
    # tmp_wrd = B(W, r, d) in the paper
    tmp_wrd = numpy.full((num_w, R, D), complex(1.0))
    for i in range(freq_dim):
        tmp_wrd *= numpy.einsum('wrl, dl -> wrd', tensors_A[i], xs_l[i], optimize=True)

    def matvec(x):
        """
        Implementation A x^{(0}}
        """
        x = x.reshape((D, R))
        t1 = time.time()
        # (wrd) * (dr) => (wd): O(Nw D R)
        tmp_wd_ = numpy.einsum('wrd, dr->wd', tmp_wrd, x, optimize=True)
        t2 = time.time()
        # (wd) * (do) => (wo): O(Nw D No)
        tmp_wo_ = numpy.einsum('wd, do->wo', tmp_wd_, x_orb, optimize=True).ravel()
        t3 = time.time()
        #print("debug A", MPI.COMM_WORLD.Get_rank(), t2-t1, t3-t2)
        return tmp_wo_

    def rmatvec(y):
        """
        Implementation A^\dagger y
        """
        y_c = y.reshape((num_w, num_o)).conjugate()
        t1 = time.time()
        # O(Nw D No)
        tmp_wd_ = numpy.einsum('do, wo -> wd', x_orb, y_c, optimize=True)
        t2 = time.time()
        # O(Nw D R)
        x_c = numpy.einsum('wrd, wd -> dr', tmp_wrd, tmp_wd_, optimize=True).ravel()
        t3 = time.time()
        #print("debug B", MPI.COMM_WORLD.Get_rank(), t2-t1, t3-t2)
        return x_c.conjugate()

    return LinearOperator((N1, N2), matvec=matvec, rmatvec=rmatvec)

def linear_operator_l(N1, N2, tensors_A_masked, tensors_A_pos, x_r, xs_l_masked, x_orb):
    """
    Linear operator of the coefficient matrix for optimizing x^{(1)}, x^{(2)}... (any tensor depends on IR basis)
    """
    freq_dim = len(tensors_A_masked)+1
    num_w, R, linear_dim = tensors_A_masked[0].shape
    D = x_r.shape[0]
    num_o = x_orb.shape[1]

    t1 = time.time()

    # O(Nw D R Nl)
    tmp_wrd1 = numpy.full((num_w, R, D), complex(1.0))
    for i in range(freq_dim-1):
        tmp_wrd1 *= numpy.einsum('wrl, dl -> wrd', tensors_A_masked[i], xs_l_masked[i], optimize=True)
    tmp_wrd1 = numpy.einsum('wrd, dr->wrd', tmp_wrd1, x_r, optimize=True)
    # tmp_wdn: (W, d, l1)
    tmp_wdn = numpy.einsum('wrd, wrn->wdn', tmp_wrd1, tensors_A_pos, optimize=True)

    tmp_wd = numpy.zeros((num_w, D), dtype=complex)
    tmp_wdn_conj = numpy.conj(tmp_wdn)

    t2 = time.time()
   
    def matvec(x):
        # x                 dn
        # tensors_A_pos     wrn
        #   ===> wrd
        # x_orb             do
        t1 = time.time()
        x = x.reshape((D, linear_dim))
        # O(Nw D Nl)
        # 'dn, wdn -> wd'
        tmp_wd[:,:] = numpy.einsum('dn,wdn->wd', x, tmp_wdn, optimize=True)
        t2 = time.time()
        # O(Nw D No)
        #  'wd, do -> wo'
        tmp_wo = numpy.dot(tmp_wd, x_orb).ravel()
        t3 = time.time()
        #if MPI.COMM_WORLD.Get_rank() == 0:
        #print("debug A", MPI.COMM_WORLD.Get_rank(), t2-t1, t3-t2, tmp_wd.shape, x_orb.shape)
        return tmp_wo

    def rmatvec(y):
        # (wdn), (do), (wo) -> (dn)
        # (do), (wo) -> (wd) : O(Nw D No)
        # (wd), (wdn) -> (dn) : O(Nw D Nl)
        t1 = time.time()
        y = y.reshape((num_w, num_o))
        # O(Nw D No)
        #  Equivalent to tmp_wd = numpy.einsum('do, wo -> wd', numpy.conj(x_orb), y, optimize=True)
        tmp_wd = numpy.dot(y, numpy.conj(x_orb).transpose())
        t2 = time.time()
        # O(Nw D Nl)
        tmp_dn = numpy.einsum('wd, wdn -> dn', tmp_wd, tmp_wdn_conj, optimize=True)
        t3 = time.time()
        #if MPI.COMM_WORLD.Get_rank() == 0:
        #print("debug B", MPI.COMM_WORLD.Get_rank(), t2-t1, t3-t2)
        return tmp_dn.ravel()

    return LinearOperator((N1, N2), matvec=matvec, rmatvec=rmatvec)

def __ridge_complex_lsqr(N1, N2, A, y, alpha, num_data=1, verbose=0, x0=None, comm=None):
    from .lsqr import lsqr
    if is_enabled_MPI:
        if comm is None:
            raise RuntimeError("comm is None")
    r = lsqr(A, y.ravel(), damp=numpy.sqrt(alpha), x0=x0, comm=comm)
    return r[0]


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
        Tensor network represenation of coefficient matrix

    nite: int
        Number of max iterations.

    rtol: float
        Stopping condition of optimization.
        Iterations stop if abs(||r_k|| - ||r_{k-1}||) < rtol * ||y||
        r_k is the residual vector at iteration k.

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

    """

    import copy

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

    assert y.shape[0] == num_w
    assert y.shape[1] == num_o

    def _init_random_array(*shape):
        return numpy.random.rand(*shape) + 1J * numpy.random.rand(*shape)

    # Init x_tensors
    x_shapes = [(D, num_rep)] + freq_dim * [(D, linear_dim)] + [(D, num_o)]
    x_sizes = [numpy.prod(shape) for shape in x_shapes]
    if random_init:
        assert x0 is None
        numpy.random.seed(seed)
        x_tensors = [_init_random_array(*shape) for shape in x_shapes]
    elif not x0 is None:
        x_tensors = copy.deepcopy(x0)
    else:
        x_tensors = [numpy.zeros(shape, dtype=complex) for shape in x_shapes]
    if is_enabled_MPI:
        for i in range(len(x_tensors)):
            x_tensors[i] = comm.bcast(x_tensors[i], root=0)

    if is_enabled_MPI:
        norm_y = numpy.sqrt(comm.allreduce(numpy.linalg.norm(y)**2))
        num_w_tot = comm.allreduce(num_w)
    else:
        norm_y = numpy.linalg.norm(y)
        num_w_tot = num_w

    def _to_x_tensors(x):
        tmp = x.reshape((2, -1))
        x1d = tmp[0, :] + 1J * tmp[1, :]
        offset = 0
        x_tensors = []
        for i in range(len(x_shapes)):
            block = x1d[offset:offset + x_sizes[i]]
            x_tensors.append(block.reshape(x_shapes[i]))
            offset += x_sizes[i]
        return x_tensors

    def _from_x_tensors(x_tensors):
        x1d = numpy.hstack([numpy.ravel(x) for x in x_tensors])
        return numpy.hstack((x1d.real, x1d.imag))

    def _se_local(x_tensors):
        y_pre = predict(prj, x_tensors)
        return squared_L2_norm(y - y_pre)

    def _snorm_local(x_tensors):
        return numpy.sum(numpy.array([squared_L2_norm(t) for t in x_tensors]))

    def loss_local(x):
        x_tensors = _to_x_tensors(x)
        return _se_local(x_tensors) + alpha * _snorm_local(x_tensors) / num_proc

    def se(x):
        x_tensors = _to_x_tensors(x)
        if is_enabled_MPI:
            return comm.allreduce(_se_local(x_tensors))
        else:
            return _se_local(x_tensors)

    def loss(x):
        if is_enabled_MPI:
            return comm.allreduce(loss_local(x))
        else:
            return loss_local(x)

    def update_r_tensor(x_r, xs_l, x_orb):
        """
        Build a least squares model for optimizing core tensor
        """
        A_op = linear_operator_r(num_w*num_o, D*num_rep, prj, x_r, xs_l, x_orb)
        x_r[:,:] = __ridge_complex_lsqr(num_w*num_o, D * num_rep, A_op, y, alpha, comm=comm).reshape((D, num_rep))

    def update_l_tensor(pos, x_r, xs_l, x_orb):
        assert pos >= 0

        t1 = time.time()

        mask = numpy.arange(freq_dim) != pos
        prj_masked = list(compress(prj, mask))
        xs_l_masked = list(compress(xs_l, mask))

        A_op = linear_operator_l(num_w*num_o, D*linear_dim, prj_masked, prj[pos], x_r, xs_l_masked, x_orb)

        # At this point, A_lsm is shape of (num_w, num_o, D, Nl)
        t2 = time.time()
        xs_l[pos][:,:] = __ridge_complex_lsqr(num_w*num_o, D*linear_dim, A_op, y, alpha,
                                                    x0=xs_l[pos].ravel(),
                                                    comm=comm).reshape((D, linear_dim))
        t3 = time.time()
        if verbose >= 2:
            print("rest : time ", t2-t1, t3-t2)

    def update_orb_tensor(x_r, xs_l, x_orb):

        # (wrl, dl) -> (wrd)
        # (wrm, dm) -> (wrd)
        # (wrn, dn) -> (wrd)
        tmp_wrd = numpy.full((num_w, num_rep, D), complex(1.0))
        for i in range(freq_dim):
            tmp_wrd *= numpy.einsum('wrl, dl -> wrd', prj[i], xs_l[i], optimize=True)
        A_lsm = numpy.einsum('wrd,dr -> w d', tmp_wrd, x_r, optimize=True)

        # A_lsm_c: (D, num_w)
        A_lsm_c = A_lsm.conjugate().transpose()

        def matvec(x):
            return numpy.dot(A_lsm, x.reshape((D, num_o))).ravel()

        def rmatvec(y):
            # Return (D, num_o)
            y = y.reshape((num_w, num_o))
            return numpy.dot(A_lsm_c, y).ravel()

        N1, N2 = num_w*num_o, D*num_o
        A_lsm_op = LinearOperator((N1, N2), matvec=matvec, rmatvec=rmatvec)
        x_orb[:, :] = __ridge_complex_lsqr(N1, N2, A_lsm_op, y,
                                                 alpha, comm=comm).reshape((D, num_o))


    def als(x):
        x_tensors = _to_x_tensors(x)

        x_r = x_tensors[0]
        xs_l = x_tensors[1:-1]
        x_orb = x_tensors[-1]

        # Optimize r tensor
        t1 = time.time()
        update_r_tensor(x_r, xs_l, x_orb)

        t2 = time.time()
        # Optimize the other tensors
        for pos in range(freq_dim):
            update_l_tensor(pos, x_r, xs_l, x_orb)

        t3 = time.time()

        update_orb_tensor(x_r, xs_l, x_orb)

        t4 = time.time()

        if rank == 0:
            print(' timings: ', t2-t1, t3-t2, t4-t3)

        return _from_x_tensors(x_tensors)

    x0 = _from_x_tensors(x_tensors)

    x_hist = []
    loss_hist = []
    rse_hist = []

    def append_x(x):
        assert len(x_hist) == len(loss_hist)
        if len(x_hist) > 2:
            del x_hist[:-2]
            del loss_hist[:-2]
            del rse_hist[:-2]
        x_hist.append(x)
        loss_hist.append(loss(x))
        rse_hist.append(numpy.sqrt(se(x)))

    append_x(x0)
    append_x(als(x0))

    beta = 1.0
    for epoch in range(nite):
        if verbose > 0 and rank == 0:
            print('epoch= ', epoch, ' loss= ', loss_hist[-1], ' rse= ', rse_hist[-1], ' diff_rse= ', numpy.abs(rse_hist[-1]-rse_hist[-2]), ' alpha = ', alpha)
        sys.stdout.flush()

        if nesterov:
            if epoch >= 1 and loss_hist[-1] > loss_hist[-2]:
                x_hist[-1] = x_hist[-2].copy()
                loss_hist[-1] = loss_hist[-2]
                beta = 0.0
            else:
                beta = 1.0
            if rank == 0:
                print(" beta = ", beta)
        else:
            beta = 0.0

        append_x(als(x_hist[-1] + beta * (x_hist[-1] - x_hist[-2])))

        if numpy.abs(rse_hist[-1]-rse_hist[-2]) < rtol * numpy.abs(rse_hist[-1]):
            break

    return _to_x_tensors(x_hist[-1])


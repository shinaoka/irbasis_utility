from __future__ import print_function

import sys
import numpy
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import OptimizeResult

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

class OvercompleteGFModel(object):
    """
    minimize |y - A * x|_2^2 + alpha * |x|_2^2
  
    A(n, r, l1, l2, ...) = U_1(n, r, l1) * U_2(n, r, l2) * ...
  
    The shape of A is (num_w, num_rep, linear_dim, linear_dim, ...)
    freq_di is the number of frequency indices.

    x(d, r, l1, l2, ..., o) = \sum_d X_0(d,r) * X_1(d,l1) * X_2(d,l2) * ... * X_{freq_dim-1}(d, l_{freq_dim-1})
        * X_{freq_dim}(d, o).

    o is an index for orbital components.

    """
    def __init__(self, num_w, num_rep, freq_dim, num_o, linear_dim, tensors_A, y, alpha, D):
        """

        Parameters
        ----------
        num_w : int
            Number of sampling points in Matsubara frequency domain

        num_rep : int
            Number of representations (meshes).
            For three-frequency objects, num_rep = 16.
            For particle-hole views, num_rep = 12.

        freq_dim : int
            Dimension of frequency axes.
            For three-frequency objects, freq_dim = 3.
            For particle-hole views, freq_dim = 2.

        linear_dim : int
            The linear dimension of IR

        tensors_A : list of ndarray objects
            Tensor-decomposed form of projector from IR to Matsubara frequency domain

        y : complex 1D array
            Data to be fitted

        alpha : float
            Initial value of regularization parameter

        D : int
            Rank of approximation (>=1)
        """

        self.y = numpy.array(y, dtype=complex)
        assert y.shape[0] == num_w
        assert y.shape[1] == num_o
        self.tensors_A = [numpy.array(t, dtype=complex) for t in tensors_A]
        self.alpha = alpha
        self.num_w = num_w
        self.num_rep = num_rep
        self.linear_dim = linear_dim
        self.num_o = num_o
        self.freq_dim = freq_dim
        assert self.freq_dim == 2 or self.freq_dim == 3
        self.D = D

        def create_tensor(N, M):
            return numpy.zeros((N, M), dtype=complex)

        self.x_r = create_tensor(D, num_rep)
        self.xs_l = [create_tensor(D, linear_dim) for i in range(freq_dim)]
        self.x_orb = create_tensor(D, num_o)

    def x_tensors(self):
        return [self.x_r] + self.xs_l + [self.x_orb]

    def full_tensor_x(self, x_tensors = None):
        if x_tensors is None:
            x_tensors = self.x_tensors()

        if self.freq_dim == 2:
            return numpy.einsum('dr, dl,dm, do->r lm o', *x_tensors, optimize=True)
        elif self.freq_dim == 3:
            return numpy.einsum('dr, dl,dm,dn do->r lmn o', *x_tensors, optimize=True)


    def predict_y(self, x_tensors=None):
        """
        Predict y from self.x_tensors
        """
        if x_tensors is None:
            x_tensors = self.x_tensors()

        return predict(self.tensors_A, x_tensors)

    def loss(self, x_tensors=None):
        """
        Compute mean squared error + L2 regularization term
        """
        if x_tensors is None:
            x_tensors = self.x_tensors()

        y_pre = self.predict_y(x_tensors)
        assert self.y.shape == y_pre.shape

        r = squared_L2_norm(self.y - y_pre)
        for t in x_tensors:
            r += self.alpha * squared_L2_norm(t)
        return r

    def squared_norm(self, x_tensors=None):
        """
        Compute mean squared error + L2 regularization term
        """
        if x_tensors is None:
            x_tensors = self.x_tensors()

        return numpy.sum([squared_L2_norm(t) for t in x_tensors])

    def update_alpha(self, squared_error, target_ratio=1e-8):
        """
        Update alpha so that L2 regularization term/residual term ~ target_ratio
        """
        x_tensors = self.x_tensors()

        reg = numpy.sum([squared_L2_norm(t) for t in x_tensors])

        #print("debug ", target_ratio, squared_error, reg)
        self.alpha = target_ratio * squared_error/reg

        return self.alpha

    def se(self, x_tensors=None):
        """
        Compute squared error
        """
        if x_tensors is None:
            x_tensors = self.x_tensors()

        y_pre = self.predict_y(x_tensors)
        assert self.y.shape == y_pre.shape

        return squared_L2_norm(self.y - y_pre)

    def mse(self, x_tensors=None):
        """
        Compute mean squared error
        """

        return self.se(x_tensors)/self.y.size

def linear_operator_r(N1, N2, tensors_A, x_r, xs_l, x_orb):
    num_w, R, linear_dim = tensors_A[0].shape
    D = x_r.shape[0]
    num_o = x_orb.shape[1]

    freq_dim = len(tensors_A)
    # O(Nw D R Nl)
    # 3* 10^5 * 100 * 10 * 30 = 10^10
    tmp_wrd = numpy.full((num_w, R, D), complex(1.0))
    for i in range(freq_dim):
        tmp_wrd *= numpy.einsum('wrl, dl -> wrd', tensors_A[i], xs_l[i], optimize=True)

    def matvec(x):
        x = x.reshape((D, R))
        # (wrd) * (dr) => (wd): O(Nw D R)
        tmp_wd_ = numpy.einsum('wrd, dr->wd', tmp_wrd, x, optimize=True)
        # (wd) * (do) => (wo): O(Nw D No)
        tmp_wo_ = numpy.einsum('wd, do->wo', tmp_wd_, x_orb, optimize=True).ravel()
        return tmp_wo_

    def rmatvec(y):
        y_c = y.reshape((num_w, num_o)).conjugate()
        # O(Nw D No)
        tmp_wd_ = numpy.einsum('do, wo -> wd', x_orb, y_c, optimize=True)
        # O(Nw D R)
        x_c = numpy.einsum('wrd, wd -> dr', tmp_wrd, tmp_wd_, optimize=True).ravel()
        return x_c.conjugate()

    return LinearOperator((N1, N2), matvec=matvec, rmatvec=rmatvec)

def linear_operator_l(N1, N2, tensors_A_masked, tensors_A_pos, x_r, xs_l_masked, x_orb):
    freq_dim = len(tensors_A_masked)+1
    num_w, R, linear_dim = tensors_A_masked[0].shape
    D = x_r.shape[0]
    num_o = x_orb.shape[1]

    # O(Nw D R Nl)
    tmp_wrd1 = numpy.full((num_w, R, D), complex(1.0))
    for i in range(freq_dim-1):
        tmp_wrd1 *= numpy.einsum('wrl, dl -> wrd', tensors_A_masked[i], xs_l_masked[i], optimize=True)
    tmp_wrd1 = numpy.einsum('wrd, dr->wrd', tmp_wrd1, x_r, optimize=True)
    tmp_wdn = numpy.einsum('wrd, wrn->wdn', tmp_wrd1, tensors_A_pos, optimize=True)
   
    def matvec(x):
        # x                 dn
        # tensors_A_pos     wrn
        #   ===> wrd
        # x_orb             do
        x = x.reshape((D, linear_dim))
        # O(Nw D Nl)
        tmp_wd = numpy.einsum('dn, wdn -> wd', x, tmp_wdn, optimize=True)
        # O(Nw D No)
        tmp_wo = numpy.einsum('wd, do -> wo', tmp_wd, x_orb, optimize=True).ravel()
        return tmp_wo

    def rmatvec(y):
        # (wdn), (do), (wo) -> (dn)
        # (do), (wo) -> (wd) : O(Nw D No)
        # (wd), (wdn) -> (dn) : O(Nw D Nl)
        y = y.reshape((num_w, num_o))
        # O(Nw D No)
        tmp_wd = numpy.einsum('do, wo -> wd', numpy.conj(x_orb), y, optimize=True)
        # O(Nw D Nl)
        tmp_dn = numpy.einsum('wd, wdn -> dn', tmp_wd, numpy.conj(tmp_wdn), optimize=True)
        return tmp_dn.ravel()

    return LinearOperator((N1, N2), matvec=matvec, rmatvec=rmatvec)

def __ridge_complex_lsqr(N1, N2, A, y, alpha, num_data=1, verbose=0, x0=None, atol=None, comm=None):
    from .lsqr import lsqr
    if is_enabled_MPI:
        if comm is None:
            raise RuntimeError("comm is None")
    r = lsqr(A, y.reshape((N1, num_data)), damp=numpy.sqrt(alpha), x0=x0, atol_r1norm=atol, comm=comm)
    return r[0]

def __normalize_tensor(tensor):
    norm = numpy.linalg.norm(tensor)
    return tensor/norm, norm

def optimize_als(model, nite, rtol = 1e-5, verbose=0, random_init=True, optimize_alpha=-1, print_interval=20, comm=None, seed=1):
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

    optimize_alpha: flaot
        If a positive value is given, the value of the regularization parameter alpha is automatically determined.
        (regularization term)/(squared norm of the residual) ~ optimize_alpha.

    Returns
    -------

    """

    rank = 0
    if comm is None and is_enabled_MPI:
        comm = MPI.COMM_WORLD

    if is_enabled_MPI:
        rank = comm.Get_rank()

    num_rep = model.num_rep
    D = model.D
    num_w = model.num_w
    freq_dim = model.freq_dim
    linear_dim = model.linear_dim
    num_o = model.num_o
    tensors_A = model.tensors_A
    y = model.y

    def update_r_tensor():
        """
        Build a least squares model for optimizing core tensor
        """
        A_op = linear_operator_r(num_w*num_o, D*num_rep, tensors_A, model.x_r, model.xs_l, model.x_orb)
        model.x_r[:,:] = __ridge_complex_lsqr(num_w*num_o, D * num_rep, A_op, y, model.alpha, atol=atol_lsqr, comm=comm).reshape((D, num_rep))

    def update_l_tensor(pos):
        assert pos >= 0

        t1 = time.time()

        mask = numpy.arange(freq_dim) != pos
        tensors_A_masked = list(compress(tensors_A, mask))
        xs_l_masked = list(compress(model.xs_l, mask))

        A_op = linear_operator_l(num_w*num_o, D*linear_dim, tensors_A_masked, tensors_A[pos], model.x_r, xs_l_masked, model.x_orb)

        # At this point, A_lsm is shape of (num_w, num_o, D, Nl)
        t2 = time.time()
        model.xs_l[pos][:,:] = __ridge_complex_lsqr(num_w*num_o, D*linear_dim, A_op, y, model.alpha,
                                                    x0=model.xs_l[pos].ravel(),
                                                    atol=atol_lsqr, comm=comm).reshape((D, linear_dim))
        t3 = time.time()
        if verbose >= 2:
            print("rest : time ", t2-t1, t3-t2)

    def update_orb_tensor():

        t1 = time.time()

        # (wrl, dl) -> (wrd)
        # (wrm, dm) -> (wrd)
        # (wrn, dn) -> (wrd)
        tmp_wrd = numpy.full((num_w, num_rep, D), complex(1.0))
        for i in range(freq_dim):
            tmp_wrd *= numpy.einsum('wrl, dl -> wrd', tensors_A[i], model.xs_l[i], optimize=True)
        A_lsm = numpy.einsum('wrd,dr -> w d', tmp_wrd, model.x_r, optimize=True)

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
        model.x_orb[:, :] = __ridge_complex_lsqr(N1, N2, A_lsm_op, y,
                                                 model.alpha, atol=atol_lsqr, comm=comm).reshape((D, num_o))

        t3 = time.time()
        if verbose >= 2:
            print("rest : time ", t2-t1, t3-t2)

    if random_init:
        numpy.random.seed(seed)
        model.x_r = numpy.random.rand(*model.x_r.shape) + 1J * numpy.random.rand(*model.x_r.shape)
        for i in range(len(model.xs_l)):
            model.xs_l[i] = numpy.random.rand(*model.xs_l[i].shape) + 1J*numpy.random.rand(*model.xs_l[i].shape)
        model.x_orb = numpy.random.rand(*model.x_orb.shape) + 1J* numpy.random.rand(*model.x_orb.shape)

    if is_enabled_MPI:
        model.x_r = comm.bcast(model.x_r, root=0)
        for i in range(len(model.xs_l)):
            model.xs_l[i] = comm.bcast(model.xs_l[i], root=0)
        model.x_orb = comm.bcast(model.x_orb, root=0)

    assert len(model.xs_l) == freq_dim
    if numpy.prod([numpy.linalg.norm(x) for x in model.x_tensors()]) == 0:
        raise RuntimeError("Some of tensors are zero at rank " + str(rank))

    def compute_residual():
        if is_enabled_MPI:
            se = comm.allreduce(model.se())
            mse = se/(comm.allreduce(num_w) * num_o)
        else:
            se = model.se()
            mse = se/(num_w * num_o)
        snorm = model.squared_norm()
        return se, snorm, mse

    losss = []
    rmses = []
    squared_errors = []

    se0, snorm0, mse0 = compute_residual()
    squared_errors.append(se0)
    rmses.append(numpy.sqrt(mse0))
    losss.append(se0 + model.alpha * snorm0)

    if is_enabled_MPI:
        norm_y = numpy.sqrt(comm.allreduce(numpy.linalg.norm(model.y)**2))
        num_w_tot = comm.allreduce(num_w)
    else:
        norm_y = numpy.linalg.norm(model.y)
        num_w_tot = num_w

    atol_lsqr = 0.1 * rtol * norm_y

    for epoch in range(nite):
        sys.stdout.flush()

        #print("debug A ", [numpy.linalg.norm(x) for x in model.x_tensors()])
        # Optimize r tensor
        t1 = time.time()
        update_r_tensor()

        #print("debug B ", [numpy.linalg.norm(x) for x in model.x_tensors()])

        t2 = time.time()
        # Optimize the other tensors
        for pos in range(model.freq_dim):
            update_l_tensor(pos)
            #print("debug C ", pos, [numpy.linalg.norm(x) for x in model.x_tensors()])

        t3 = time.time()

        update_orb_tensor()
        #print("debug D ", [numpy.linalg.norm(x) for x in model.x_tensors()])

        t4 = time.time()

        se, snorm, mse = compute_residual()
        squared_errors.append(se)
        rmses.append(numpy.sqrt(mse))
        losss.append(se + model.alpha * snorm)

        if epoch%print_interval == 0:
            if verbose > 0 and rank == 0:
                print("epoch = ", epoch, " loss = ", losss[-1], " rmse = ", rmses[-1], " alpha = ", model.alpha, " snorm=", snorm)

        if len(rmses) > 2:
            if numpy.abs(rmses[-2] - rmses[-1]) < rtol * (norm_y/numpy.sqrt(num_w_tot)):
                break

        if optimize_alpha > 0:
            se = comm.allreduce(model.se())
            model.update_alpha(se, optimize_alpha)

        t5 = time.time()

        if rank == 0:
            print(t2-t1, t3-t2, t4-t3, t5-t4)

        sys.stdout.flush()

    info = {}
    info['losss'] = losss
    info['rmses'] = rmses

    return info

def minimize_ls(fun, x0, jac, options, callback):
    maxiter = options['maxiter']
    disp = options['disp']

    amp_fac = 2.0

    #eta = 1e-5
    #gamma = 0.0

    xk = x0.copy()
    dx = 0.0 * xk
    
    for iter in range(maxiter):
        fun_xk = fun(xk)

        grad = jac(xk)
        eta = 1.0
        while True:
            x_new = xk - eta * grad
            fun_x_new = fun(x_new)
            #print("debug ", fun_x_new, fun_xk, eta)
            if fun_x_new < fun_xk:
                break
            eta *= 0.1

        if disp:
            print("Iter {} :  func= {} eta= {}".format(iter, fun_xk, eta))
        callback(xk)

        xk = x_new.copy()

        #if iter > 0:
            #if fun_xk > fun_xkm1:
                #eta /= amp_fac
                #xk = xkm1.copy()
                #fun_xk = fun_xkm1
            #else:
                #eta *= amp_fac

        #xkm1 = xk.copy()
        #fun_xkm1 = fun_xk
        #xk += - eta * jac(xk)

    res = OptimizeResult()
    res.x = xk
    return res

def optimize_grad(model, nite, method='l-bfgs', verbose=0, random_init=True, optimize_alpha=-1, print_interval=20, comm=None, seed=1, learning_rate=None):
    """
    L-BFGS

    Parameters
    ----------
    model:  object of OvercompleteGFModel
        Model to be optimized

    nite: int
        Number of max iterations.

    verbose: int
        0, 1, 2

    optimize_alpha: flaot
        If a positive value is given, the value of the regularization parameter alpha is automatically determined.
        (regularization term)/(squared norm of the residual) ~ optimize_alpha.

    Returns
    -------

    """

    import autograd.numpy as numpy  # Thinly-wrapped numpy
    from autograd import grad  # The only autograd function you may ever need
    from scipy.optimize import minimize

    rank = 0
    if comm is None and is_enabled_MPI:
        comm = MPI.COMM_WORLD

    if is_enabled_MPI:
        rank = comm.Get_rank()

    num_rep = model.num_rep
    D = model.D
    num_w = model.num_w
    freq_dim = model.freq_dim
    linear_dim = model.linear_dim
    num_o = model.num_o
    tensors_A = model.tensors_A
    y = model.y
    alpha = model.alpha

    x_shapes = [(D,num_rep)] + [(D,linear_dim)] * freq_dim + [(D,num_o)]
    x_sizes = [numpy.prod(shape) for shape in x_shapes]

    if random_init:
        numpy.random.seed(seed)
        model.x_r = numpy.random.rand(*model.x_r.shape) + 1J * numpy.random.rand(*model.x_r.shape)
        for i in range(len(model.xs_l)):
            model.xs_l[i] = numpy.random.rand(*model.xs_l[i].shape) + 1J*numpy.random.rand(*model.xs_l[i].shape)
        model.x_orb = numpy.random.rand(*model.x_orb.shape) + 1J* numpy.random.rand(*model.x_orb.shape)
        if is_enabled_MPI:
            model.x_r = comm.bcast(model.x_r, root=0)
            for i in range(len(model.xs_l)):
                model.xs_l[i] = comm.bcast(model.xs_l[i], root=0)
            model.x_orb = comm.bcast(model.x_orb, root=0)
    #else:
        #model.x_r = x0_tensors[0]
        #model.xsl_l = x0_tensors[1:-1]
        #model.x_orb = x0_tensors[-1]

    alpha = 1e-10

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
        return numpy.hstack([x1d.real, x1d.imag])

    def _predict(prj, x_tensors):
        xs_l = x_tensors[1:-1]
        freq_dim = len(prj)
        t1 = time.time()

        tmp_wrd = numpy.full((num_w, num_rep, D), complex(1.0))
        # O(Nw R D Nl)
        for i in range(freq_dim):
            tmp_wrd *= numpy.einsum('wrl, dl -> wrd', prj[i], xs_l[i], optimize=True)
        t2 = time.time()
        # O(Nw R D)
        tmp_wd = numpy.einsum('wrd, dr->wd', tmp_wrd, x_tensors[0], optimize=True)
        t3 = time.time()
        # O(Nw D No)
        tmp_wo = numpy.einsum('wd, do->wo', tmp_wd, x_tensors[-1], optimize=True).reshape((num_w, num_o))
        t4 = time.time()
        # print(t2-t1, t3-t2, t4-t3)
        return tmp_wo

    def _squared_L2_norm(x):
        x = x.ravel()
        return numpy.real(numpy.dot(numpy.conj(x.transpose()), x))

    def _compute_se(x_tensors):
        y_pre = _predict(tensors_A, x_tensors)
        return _squared_L2_norm(y - y_pre)

    def _compute_snorm(x_tensors):
        return numpy.sum([_squared_L2_norm(t) for t in x_tensors])

    def se_local(x):
        return _compute_se(_to_x_tensors(x))

    def snorm(x):
        return _compute_snorm(_to_x_tensors(x))

    grad_se_local = grad(se_local)
    grad_snorm = grad(snorm)

    def func(x):
        if is_enabled_MPI:
            se = comm.allreduce(se_local(x))
        else:
            se = se_local(x)
        return se + alpha * snorm(x)

    def grad_func(x):
        if is_enabled_MPI:
            grad_se = comm.allreduce(grad_se_local(x))
        else:
            grad_se = grad_se_local(x)
        return grad_se + alpha * grad_snorm(x)

    def callback(xk):
        if optimize_alpha > 0:
            reg = snorm(xk)
            if is_enabled_MPI:
                se = comm.allreduce(se_local(xk))
            else:
                se = se_local(xk)
            alpha = optimize_alpha * se/reg
            if rank == 0:
                print("alpha = ", alpha)
        sys.stdout.flush()

    x0_tensors = [model.x_r] + model.xs_l + [model.x_orb]
    x0 = _from_x_tensors(x0_tensors)

    # Sanity check
    x0_tensors_reconst = _to_x_tensors(x0)
    for i in range(len(x0_tensors)):
        numpy.allclose(x0_tensors[i], x0_tensors_reconst[i])

    disp = False
    if rank == 0:
        disp = True
    if method == 'l-bfgs':
        res = minimize(fun=func, x0=x0, jac=grad_func, method="L-BFGS-B", options={'maxiter' : nite, 'disp': disp}, callback=callback)
    elif method=='ls':
        res = minimize_ls(fun=func, x0=x0, jac=grad_func, options={'maxiter' : nite, 'learning_rate' : learning_rate, 'disp': disp}, callback=callback)
    else:
        raise RuntimeError("Unknown method")

    x_tensors = _to_x_tensors(res.x)
    model.x_r = x_tensors[0]
    model.xs_l = x_tensors[1:-1]
    model.x_orb = x_tensors[-1]
    model.alpha = alpha

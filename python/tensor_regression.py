from __future__ import print_function

from .regression import ridge_complex

import numpy
import scipy
from scipy.linalg import LinAlgError
from scipy.sparse.linalg import lsmr, LinearOperator

from itertools import compress, product
import time

is_enabled_MPI = False
is_master_node = True

def enable_MPI():
    global is_enabled_MPI
    global MPI
    global comm
    global is_master_node
    is_enabled_MPI = True
    print("Enabling MPI in tensor regression")
    _temp = __import__('mpi4py', globals(), locals(), ['MPI'], 0)
    MPI = _temp.MPI
    comm = MPI.COMM_WORLD
    is_master_node = comm.Get_rank() == 0

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

def cp_to_full_tensor(x_tensors):
    """
    Construct a full tensor representation
    
    sum_d X_0(d,l0) * X_1(d,l1) * X_2(d,l2) * ...
    
    We contract the tensors from left to right as follows:
        X_0(d, l0) -> tildeT(d, l0)
        tildeT(d,l0) * X_1(d,l1) -> tildeT(d, l0, l1)
    """
    dim = len(x_tensors)
    dims = [x.shape[1] for x in x_tensors]
    D = x_tensors[0].shape[0]
    
    tildeT = x_tensors[0]
    for i in range(1, dim):
        tildeT_reshaped = numpy.reshape(tildeT, (D,-1))
        tildeT = numpy.einsum('dI,di->dIi', tildeT_reshaped, x_tensors[i])
    full_tensor = numpy.sum(numpy.reshape(tildeT, (D,) + tuple(dims)), axis=0)
    return full_tensor


class OvercompleteGFModel(object):
    """
    minimize |y - A * x|_2^2 + alpha * |x|_2^2
  
    A(n, r, l1, l2, ...) = U_1(n, r, l1) * U_2(n, r, l2) * ...
  
    The shape of A is (num_w, num_rep, linear_dim, linear_dim, ...)
    freq_di is the number of frequency indices.

    x(d, r, l1, l2, ...) = \sum_d X_0(d,r) * X_1(d,l1) * X_2(d,l2) * ... * X_{freq_dim-1}(d, l_{freq_dim-1})
        * X_{freq_dim}(d, i)
        * X_{freq_dim}(d, j)
        * X_{freq_dim}(d, k)
        * X_{freq_dim}(d, l)

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
            For particle-hole vies, num_rep = 12.

        freq_dim : int
            Dimension of frequency axes.
            For three-frequency objects, freq_dim = 3.
            For particle-hole vies, freq_dim = 2.

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
            rand = numpy.random.rand(N, M) + 1J * numpy.random.rand(N, M)
            return rand
    
        self.x_r = create_tensor(D, num_rep)
        self.xs_l = [create_tensor(D, linear_dim) for i in range(freq_dim)]
        self.x_orb = numpy.random.rand(D, num_o) + 1J * numpy.random.rand(D, num_o)

        if is_enabled_MPI:
            self.x_r = comm.bcast(self.x_r, root=0)
            self.xs_l = comm.bcast(self.xs_l, root=0)
            self.x_orb = comm.bcast(self.x_orb, root=0)

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
    
        sum_d sum_{r,l1,l2,...} A(n, r, l1, l2, ...) * X_0(d,r) * X_1(d,l1) * X_2(d,l2) * ...
    
        We used A(n, r, l1, l2, ...) = U_1(n, r, l1) * U_2(n, r, l2) * ...
    
        We contract the tensors as follows:
            U_1(n, r, l1) * X_1(d, l1) -> UX_1(n, r, d)
            U_2(n, r, l2) * X_2(d, l2) -> UX_2(n, r, d)
        """
        if x_tensors is None:
            x_tensors = self.x_tensors()

        if self.freq_dim == 2:
            return numpy.einsum('wrl,wrm, dr,dl,dm, do->wo', *(self.tensors_A + x_tensors), optimize=True)
        elif self.freq_dim == 3:
            return numpy.einsum('wrl,wrm,wrn, dr,dl,dm,dn, do->wo', *(self.tensors_A + x_tensors), optimize=True)

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

    def update_alpha(self, target_ratio=1e-8):
        """
        Update alpha so that L2 regularization term/residual term ~ target_ratio
        """
        x_tensors = self.x_tensors()

        y_pre = self.predict_y(x_tensors)

        res = squared_L2_norm(self.y - y_pre)
        reg = numpy.sum([squared_L2_norm(t) for t in x_tensors])

        self.alpha = target_ratio * res/reg

        return self.alpha

    def mse(self, x_tensors=None):
        """
        Compute mean squared error
        """
        if x_tensors is None:
            x_tensors = self.x_tensors()

        y_pre = self.predict_y(x_tensors)
        assert self.y.shape == y_pre.shape
        return (squared_L2_norm(self.y - y_pre))/y_pre.size

def linear_operator_r(N1, N2, tensors_A, x_r, xs_l, x_orb):
    num_w, R, linear_dim = tensors_A[0].shape
    D = x_r.shape[0]
    num_o = x_orb.shape[1]

    freq_dim = len(tensors_A)
    if freq_dim == 2:
        tmp_wrd = numpy.einsum('wrl,wrm, dl,dm -> wrd', *(tensors_A + xs_l), optimize=True)
    elif freq_dim == 3:
        tmp_wrd = numpy.einsum('wrl,wrm,wrn, dl,dm,dn -> wrd', *(tensors_A + xs_l), optimize=True)

    def matvec(x):
        x = x.reshape((D, R))
        return numpy.einsum('wrd, do, dr->wo', tmp_wrd, x_orb, x, optimize=True).reshape(-1)

    def rmatvec(y):
        y_c = y.reshape((num_w, num_o)).conjugate()
        x_c = numpy.einsum('wrd, do, wo->dr', tmp_wrd, x_orb, y_c, optimize=True).reshape(-1)
        return x_c.conjugate()

    return LinearOperator((N1, N2), matvec=matvec, rmatvec=rmatvec)

def linear_operator_l(N1, N2, tensors_A_masked, tensors_A_pos, x_r, xs_l_masked, x_orb):
    # tensors_A_masked  [wrl]
    # x_r               dr
    # xs_l_masked       [dl]
    #   ===> wrd
    freq_dim = len(tensors_A_masked)+1
    num_w, R, linear_dim = tensors_A_masked[0].shape
    D = x_r.shape[0]
    num_o = x_orb.shape[1]


    if is_enabled_MPI:
        sizes, offsets = mpi_split(num_w, comm.size)
        rank = comm.Get_rank()
        start, end = offsets[rank], offsets[rank] + sizes[rank]

        #print("split ", sizes, offsets, start, end)
        tensors_A_masked = [t[start:end, :, :] for t in tensors_A_masked]
        tensors_A_pos = tensors_A_pos[start:end, :, :]


    if freq_dim == 2:
        tmp_wrd1 = numpy.einsum('wrl, dr, dl-> wrd', *(tensors_A_masked + [x_r] + xs_l_masked), optimize=True)
    elif freq_dim == 3:
        tmp_wrd1 = numpy.einsum('wrl,wrm, dr, dl,dm-> wrd', *(tensors_A_masked + [x_r] + xs_l_masked), optimize=True)
    else:
        raise RuntimeError("freq_dim must be either 2 or 3!")

    def matvec(x):
        # x                 dn
        # tensors_A_pos     wrn
        #   ===> wrd
        # x_orb             do
        x = x.reshape((D, linear_dim))
        tmp_wrd2 = numpy.einsum('dn, wrn -> wrd', x, tensors_A_pos, optimize=True)
        tmp_wd = numpy.einsum('wrd, wrd -> wd', tmp_wrd1, tmp_wrd2, optimize=True)
        tmp_wo = numpy.einsum('wd, do -> wo', tmp_wd, x_orb, optimize=True).reshape(-1)
        if is_enabled_MPI:
            tmp_wo_all = numpy.empty(num_w * num_o, dtype=complex)
            comm.Allgatherv(tmp_wo.ravel(), [tmp_wo_all, sizes * num_o, offsets * num_o, MPI.DOUBLE_COMPLEX])
            return tmp_wo_all
        else:
            return tmp_wo

    # tmp_wrd1          wrd
    # tensors_A_pos     wrn
    #    ===> wdn
    tmp_wdn = numpy.einsum('wrd, wrn -> wdn', tmp_wrd1, tensors_A_pos, optimize=True).conjugate()
    def rmatvec(y):
        # Note: do not forget to take complex conjugate of A
        # y                 wo
        #    ===> dno
        # x_orb             do
        #    ===> dn
        y = y.reshape((num_w, num_o))
        if is_enabled_MPI:
            tmp_dno_local = numpy.einsum('wdn, wo -> dno', tmp_wdn, y[start:end, :], optimize=True)
            tmp_dno = numpy.zeros(tmp_dno_local.size, dtype=complex)
            comm.Allreduce([tmp_dno_local.ravel(), MPI.DOUBLE_COMPLEX], [tmp_dno, MPI.DOUBLE_COMPLEX], op=MPI.SUM)
            #print("local ", rank, numpy.sum(tmp_dno_local))
            tmp_dno = tmp_dno.reshape(tmp_dno_local.shape)
            #print("sum ", rank, numpy.sum(tmp_dno))
        else:
            tmp_dno = numpy.einsum('wdn, wo -> dno', tmp_wdn, y, optimize=True)
        return numpy.einsum('dno, do -> dn', tmp_dno, numpy.conj(x_orb), optimize=True).reshape(-1)

    return LinearOperator((N1, N2), matvec=matvec, rmatvec=rmatvec)

def __ridge_complex_lsmr(N1, N2, A, y, alpha, num_data=1, verbose=0):
    if isinstance(A, numpy.ndarray):
        r = lsmr(A.reshape((N1,N2)), y.reshape((N1, num_data)), damp=numpy.sqrt(alpha))
    else:
        r = lsmr(A, y.reshape((N1, num_data)), damp=numpy.sqrt(alpha))
    return r[0]

def __normalize_tensor(tensor):
    norm = numpy.linalg.norm(tensor)
    return tensor/norm, norm

def optimize_als(model, nite, tol_rmse = 1e-5, verbose=0, min_norm=1e-8, optimize_alpha=-1, print_interval=20):
    """
    Alternating least squares

    Parameters
    ----------
    model:  object of OvercompleteGFModel
        Model to be optimized

    nite: int
        Number of max iterations.

    tol_rmse: float
        Stopping condition of optimization. rmse denotes "root mean squared error".

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
        t1 = time.time()


        t2 = time.time()
        A_op = linear_operator_r(num_w*num_o, D*num_rep, tensors_A, model.x_r, model.xs_l, model.x_orb)
        model.x_r[:,:] = __ridge_complex_lsmr(num_w*num_o, D * num_rep, A_op, y, model.alpha).reshape((D, num_rep))
        t3 = time.time()
        if verbose >= 2:
            print("r_tensor : time ", t2-t1, t3-t2)

    def update_l_tensor(pos):
        assert pos >= 0

        t1 = time.time()

        mask = numpy.arange(freq_dim) != pos
        tensors_A_masked = list(compress(tensors_A, mask))
        xs_l_masked = list(compress(model.xs_l, mask))

        A_op = linear_operator_l(num_w*num_o, D*linear_dim, tensors_A_masked, tensors_A[pos], model.x_r, xs_l_masked, model.x_orb)

        # At this point, A_lsm is shape of (num_w, num_o, D, Nl)
        t2 = time.time()
        model.xs_l[pos][:,:] = __ridge_complex_lsmr(num_w*num_o, D*linear_dim, A_op, y, model.alpha).reshape((D, linear_dim))
        t3 = time.time()
        if verbose >= 2:
            print("rest : time ", t2-t1, t3-t2)

    def update_orb_tensor():

        t1 = time.time()

        if freq_dim == 2:
            A_lsm = numpy.einsum('wrl,wrm, dr, dl,dm -> w d',
                               *(tensors_A + [model.x_r] + model.xs_l), optimize=True)
        elif freq_dim == 3:
            A_lsm = numpy.einsum('wrl,wrm,wrn, dr, dl,dm,dn -> w d',
                                 *(tensors_A + [model.x_r] + model.xs_l), optimize=True)

        # At this point, A_lsm is shape of (num_w, D)
        for o in range(num_o):
            model.x_orb[:, o] = __ridge_complex_lsmr(num_w, D, A_lsm, y[:, o], model.alpha)

        t3 = time.time()
        if verbose >= 2:
            print("rest : time ", t2-t1, t3-t2)

    losss = []
    epochs = range(nite)
    rmses = []
    for epoch in epochs:
        # Optimize r tensor
        t1 = time.time()
        update_r_tensor()

        t2 = time.time()
        # Optimize the other tensors
        for pos in range(model.freq_dim):
            update_l_tensor(pos)

        t3 = time.time()

        update_orb_tensor()

        t4 = time.time()

        if is_master_node:
            print(t2-t1, t3-t2, t4-t3)

        #print("epoch = ", epoch, " loss = ", model.loss(), numpy.sqrt(model.mse()))
        if epoch%print_interval == 0:
            loss = model.loss()
            losss.append(loss)
            rmses.append(numpy.sqrt(model.mse()))
            if verbose > 0 and is_master_node:
                print("epoch = ", epoch, " loss = ", losss[-1], " rmse = ", rmses[-1], " alpha = ", model.alpha)
                #for i, x in enumerate(model.x_tensors()):
                    #print("norm of x ", i, numpy.linalg.norm(x))

        if len(rmses) > 2:
            diff_rmse = numpy.abs(rmses[-2] - rmses[-1])
            if diff_rmse < tol_rmse:
                break

        if optimize_alpha > 0:
            model.update_alpha(optimize_alpha)

    info = {}
    info['losss'] = losss
    info['rmses'] = rmses

    return info


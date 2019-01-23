from __future__ import print_function

from .regression import ridge_complex

import numpy
from scipy.linalg import LinAlgError
import scipy

from itertools import compress, product
import time

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
    def __init__(self, num_w, num_rep, freq_dim, num_orb, linear_dim, tensors_A, y, alpha, D):
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
        assert y.shape[1] == num_orb
        self.tensors_A = [numpy.array(t, dtype=complex) for t in tensors_A]
        self.alpha = alpha
        self.num_w = num_w
        self.num_rep = num_rep
        self.linear_dim = linear_dim
        self.num_orb = num_orb
        self.freq_dim = freq_dim
        self.D = D
    
        def create_tensor(N, M):
            rand = numpy.random.rand(N, M) + 1J * numpy.random.rand(N, M)
            return rand
    
        self.x_r = create_tensor(D, num_rep)
        self.xs_l = [create_tensor(D, linear_dim) for i in range(freq_dim)]
        self.x_orb = numpy.random.rand(D, num_orb, num_orb, num_orb, num_orb) + 1J * numpy.random.rand(D, num_orb, num_orb, num_orb, num_orb)

    def x_tensors(self):
        return [self.x_r] + self.xs_l + [self.x_orb]

    def full_tensor_x(self, x_tensors=None):
        if x_tensors is None:
            return cp_to_full_tensor(self.x_tensors())
        else:
            return cp_to_full_tensor(x_tensors)

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
            return numpy.einsum('wrl,wrm, dr,dl,dm, defgh->wefgh', *(self.tensors_A + x_tensors), optimize=True)
        elif self.freq_dim == 3:
            return numpy.einsum('wrl,wrm,wrn, dr,dl,dm,dn, defgh->wefgh', *(self.tensors_A + x_tensors), optimize=True)

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


def __ridge_complex(N1, N2, A, y, alpha, x_old, solver='svd', precond=None, verbose=0):
    A_numpy = A.reshape((N1, N2))
    y_numpy = y.reshape((N1,))
    x_old_numpy = x_old.reshape((N2,))

    try:
        if solver == 'svd':
            try:
                x_numpy = ridge_complex(A_numpy, y_numpy, alpha, solver='svd')
            except LinAlgError:
                if verbose > 0:
                    print("svd did not converge, falling back to lsqr...")
                x_numpy = ridge_complex(A_numpy, y_numpy, alpha, solver='lsqr')

        elif solver == 'lsqr':
            x_numpy = ridge_complex(A_numpy, y_numpy, alpha, solver='lsqr', x0=x_old_numpy, precond=precond, verbose=verbose)
        else:
            raise RuntimeError("Unsupported solver: " + solver)
    except:
        print("A: ", A_numpy)
        print("y: ", y_numpy)
        print("alpha: ", alpha)
        raise RuntimeError("Error in ridge_complex")

    new_loss =  numpy.linalg.norm(y_numpy - numpy.dot(A_numpy, x_numpy))**2 + alpha * numpy.linalg.norm(x_numpy)**2
    old_loss = numpy.linalg.norm(y_numpy - numpy.dot(A_numpy, x_old_numpy))**2 + alpha * numpy.linalg.norm(x_old_numpy)**2
    loss_diff = new_loss - old_loss

    if verbose > 0 and loss_diff > 0 and numpy.abs(loss_diff) > 1e-8 * numpy.abs(new_loss):
        U, S, V = scipy.linalg.svd(A_numpy)
        print("Warning loss_diff > 0!: loss_diff = ", loss_diff)
        print("    condA", S[0]/S[-1], S[0], S[-1])
        print("    ", numpy.linalg.norm(y_numpy - numpy.dot(A_numpy, x_numpy))**2)
        print("    ", numpy.linalg.norm(y_numpy - numpy.dot(A_numpy, x_old_numpy))**2)
        print("    ", alpha * numpy.linalg.norm(x_numpy)**2)
        print("    ", alpha * numpy.linalg.norm(x_old_numpy)**2)

    return x_numpy, loss_diff

def __normalize_tensor(tensor):
    norm = numpy.linalg.norm(tensor)
    return tensor/norm, norm

def __sketch(tensors_A, y, sketch_size):
    num_w = y.shape[0]

    if sketch_size >= num_w:
        return [tensors_A, y]

    idx = numpy.sort(numpy.random.choice(numpy.arange(num_w), sketch_size, replace=False))
    return [t[idx,:,:] for t in tensors_A], y[idx,:,:,:,:]


def optimize_als(model, nite, tol_rmse = 1e-5, solver='svd', verbose=0, min_norm=1e-8, optimize_alpha=-1, sketch_size_fact=1E+8, print_interval=20):
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

    solver: string ('svd' or 'lsqr')
        Workhorse for solving a reduced least squares problem.
        Recommended to use 'svd'.
        If 'svd' fails to converge, it falls back to 'lsqr'.

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
    num_orb = model.num_orb
    tensors_A = model.tensors_A
    y = model.y

    def update_r_tensor():
        """
        Build a least squares model for optimizing core tensor
        """
        t1 = time.time()

        num_w_sk = min(num_w, int(sketch_size_fact * D * num_rep))
        tensors_A_sk, y_sk = __sketch(tensors_A, y, num_w_sk)

        if freq_dim == 2:
            A_lsm = numpy.einsum('wrl,wrm, dl,dm, defgh->w efgh dr',
                                 *(tensors_A_sk + model.xs_l + [model.x_orb]), optimize=True)
        elif freq_dim == 3:
            A_lsm = numpy.einsum('wrl,wrm,wrn, dl,dm,dn, defgh -> w efgh dr',
                                 *(tensors_A_sk + model.xs_l + [model.x_orb]), optimize=True)

        t2 = time.time()
        new_core_tensor, diff = __ridge_complex(num_w_sk*num_orb**4, D * num_rep, A_lsm, y_sk, model.alpha, model.x_r, solver)
        t3 = time.time()
        if verbose >= 2:
            print("core : time ", t2-t1, t3-t2)
        model.x_r = numpy.reshape(new_core_tensor, [D, num_rep])

        return diff

    def update_l_tensor(pos):
        assert pos >= 0

        t1 = time.time()
        num_w_sk = min(num_w, int(sketch_size_fact * D * linear_dim))
        tensors_A_sk, y_sk = __sketch(tensors_A, y, num_w_sk)

        mask = numpy.arange(freq_dim) != pos
        tensors_A_masked = list(compress(tensors_A_sk, mask))
        xs_l_masked = list(compress(model.xs_l, mask))

        if freq_dim == 2:
            tmp = numpy.einsum('wrl, dr, dl, defgh -> wefgh rd',
                               *(tensors_A_masked + [model.x_r] + xs_l_masked + [model.x_orb]), optimize=True)
        elif freq_dim == 3:
            tmp = numpy.einsum('wrl,wrm, dr, dl,dm, defgh -> wefgh rd',
                               *(tensors_A_masked + [model.x_r] + xs_l_masked + [model.x_orb]), optimize=True)
        A_lsm = numpy.einsum('wefgh rd, wrl->wefgh dl', tmp, tensors_A_sk[pos], optimize=True)

        # At this point, A_lsm is shape of (num_w, num_orb, num_orb, num_orb, num_orb, D, Nl)
        t2 = time.time()
        new_tensor, diff = __ridge_complex(num_w_sk*num_orb**4, D*linear_dim, A_lsm, y_sk, model.alpha, model.xs_l[pos], solver)
        t3 = time.time()
        if verbose >= 2:
            print("rest : time ", t2-t1, t3-t2)

        model.xs_l[pos] = numpy.reshape(new_tensor, [D, linear_dim])

        return diff

    def update_orb_tensor():
        diff = 0.0

        t1 = time.time()
        num_w_sk = min(num_w, int(sketch_size_fact * D * num_orb))
        tensors_A_sk, y_sk = __sketch(tensors_A, y, num_w_sk)

        if freq_dim == 2:
            A_lsm = numpy.einsum('wrl,wrm, dr, dl,dm -> w d',
                               *(tensors_A_sk + [model.x_r] + model.xs_l), optimize=True)
        elif freq_dim == 3:
            A_lsm = numpy.einsum('wrl,wrm,wrn, dr, dl,dm,dn -> w d',
                                 *(tensors_A_sk + [model.x_r] + model.xs_l), optimize=True)

        # At this point, A_lsm is shape of (num_w, D)
        # TODO: VECTERIZE
        for i, j, k, l in product(range(num_orb), repeat=4):
            new_tensor, diff_tmp = __ridge_complex(num_w_sk, D, A_lsm, y[:, i, j, k, l],
                                                   model.alpha, model.x_orb[:, i, j, k, l], solver)
            model.x_orb[:, i, j, k, l] = new_tensor
            diff += diff_tmp

        t3 = time.time()
        if verbose >= 2:
            print("rest : time ", t2-t1, t3-t2)

        return diff

    losss = []
    epochs = range(nite)
    loss = model.loss()
    rmses = []
    for epoch in epochs:
        # Optimize r tensor
        t1 = time.time()
        update_r_tensor()

        assert not loss is None

        t2 = time.time()
        # Optimize the other tensors
        for pos in range(model.freq_dim):
            update_l_tensor(pos)

            assert not loss is None
        t3 = time.time()

        update_orb_tensor()

        assert not loss is None
        t4 = time.time()

        #print(t2-t1, t3-t2, t4-t3)

        #print("epoch = ", epoch, " loss = ", loss, model.loss(), numpy.sqrt(model.mse()))
        if epoch%print_interval == 0:
            loss = model.loss()
            losss.append(loss)
            rmses.append(numpy.sqrt(model.mse()))
            if verbose > 0:
                print("epoch = ", epoch, " loss = ", losss[-1], " rmse = ", rmses[-1], " alpha = ", model.alpha)
                for i, x in enumerate(model.x_tensors()):
                    print("norm of x ", i, numpy.linalg.norm(x))

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


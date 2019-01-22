from __future__ import print_function

from .regression import ridge_complex

import numpy
from scipy.linalg import LinAlgError
import scipy

from itertools import compress
import time

def full_A_tensor(tensors_A):
    """
    Decomposition of A tensors
    A(n, r, l1, l2, ...) = U_1(n, r, l1) * U_2(n, r, l2) * ...
    
    The full tensor can be constructed as follows:
       U_1(n, r, l1)                    -> tildeA(n, r, l1)
       tildeA(n, r, l1) * U_2(n, r, l2) -> tildeA(n, r, l1, l2)
    """
    Nw = tensors_A[0].shape[0]
    Nr = tensors_A[0].shape[1]
    dim = len(tensors_A)
    
    tildeA = tensors_A[0]
    for i in range(1, dim):
        tildeA_reshaped = numpy.reshape(tildeA, (Nw, Nr, -1))
        tildeA = numpy.einsum('nrl,nrm->nrlm', tildeA_reshaped, tensors_A[i])
    
    full_tensor_dims = tuple([Nw, Nr] + [t.shape[-1] for t in tensors_A])
    return numpy.reshape(tildeA, full_tensor_dims)


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
  
    The shape of A is (Nw, Nr, linear_dim, linear_dim, ...)
    freq_di is the number of frequency indices.

    x(d, r, l1, l2, ...) = \sum_d X_0(d,r) * X_1(d,l1) * X_2(d,l2) * ... * X_{freq_dim-1}(d, l_{freq_dim-1})
        * X_{freq_dim}(d, i)
        * X_{freq_dim}(d, j)
        * X_{freq_dim}(d, k)
        * X_{freq_dim}(d, l)

    """
    def __init__(self, Nw, Nr, freq_dim, linear_dim, tensors_A, y, alpha, D):
        """

        Parameters
        ----------
        Nw : int
            Number of sampling points in Matsubara frequency domain

        Nr : int
            Number of representations (meshes).
            For three-frequency objects, Nr = 16.
            For particle-hole vies, Nr = 12.

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
        #self.right_dims = (Nr,) + (linear_dim,) * freq_dim
        #self.right_dim = freq_dim + 1
    
        self.y = numpy.array(y, dtype=complex)
        self.tensors_A = [numpy.array(t, dtype=complex) for t in tensors_A]
        self.alpha = alpha
        self.Nw = Nw
        self.Nr = Nr
        self.linear_dim = linear_dim
        self.freq_dim = freq_dim
        self.D = D
    
        def create_tensor(N, M):
            rand = numpy.random.rand(N, M) + 1J * numpy.random.rand(N, M)
            return rand
    
        self.x_tensor_r = create_tensor(D, Nr)
        self.x_tensors_l = [create_tensor(D, linear_dim) for i in range(freq_dim)]

    def x_tensors(self):
        return [self.x_tensor_r] + self.x_tensors_l

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
            return numpy.einsum('wrl,wrm, dr,dl,dm->w', *(self.tensors_A + x_tensors), optimize=True)
        elif self.freq_dim == 3:
            return numpy.einsum('wrl,wrm,wrn, dr,dl,dm,dn->w', *(self.tensors_A + x_tensors), optimize=True)

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
        return r/self.Nw

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
        return (squared_L2_norm(self.y - y_pre))/self.Nw


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

    return x_numpy, loss_diff/N1

def __normalize_tensor(tensor):
    norm = numpy.linalg.norm(tensor)
    return tensor/norm, norm

def optimize_als(model, nite, tol_rmse = 1e-5, solver='svd', verbose=0, min_norm=1e-8, optimize_alpha=-1):
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
    Nr = model.Nr
    D = model.D
    Nw = model.Nw
    freq_dim = model.freq_dim
    linear_dim = model.linear_dim
    tensors_A = model.tensors_A
    x_tensor_r = model.x_tensor_r
    x_tensors_l = model.x_tensors_l
    y = model.y

    def update_core_tensor():
        """
        Build a least squares model for optimizing core tensor
        """
        t1 = time.time()
        if freq_dim == 2:
            A_lsm = numpy.einsum('wrl,wrm, dl,dm->wdr', *(tensors_A + x_tensors_l), optimize=True)
        elif freq_dim == 3:
            A_lsm = numpy.einsum('wrl,wrm,wrn, dl,dm,dn->wdr', *(tensors_A + x_tensors_l), optimize=True)

        # Transpose A_lsm from (Nw, R, D) to (Nw, D, R)
        t2 = time.time()
        new_core_tensor, diff = __ridge_complex(Nw, D * Nr, A_lsm, y, model.alpha, x_tensor_r, solver)
        t3 = time.time()
        if verbose >= 2:
            print("core : time ", t2-t1, t3-t2)
        model.x_tensor_r = numpy.reshape(new_core_tensor, [D, Nr])

        return diff

    def update_l_tensor(pos):
        assert pos >= 0

        t1 = time.time()
        mask = numpy.arange(freq_dim) != pos
        tensors_A_masked = list(compress(tensors_A, mask))
        x_tensors_l_masked = list(compress(x_tensors_l, mask))

        if freq_dim == 2:
            tmp = numpy.einsum('wrl, dr, dl->wrd', *(tensors_A_masked + [x_tensor_r] + x_tensors_l_masked), optimize=True)
        elif freq_dim == 3:
            tmp = numpy.einsum('wrl,wrm, dr, dl,dm->wrd', *(tensors_A_masked + [x_tensor_r] + x_tensors_l_masked), optimize=True)
        A_lsm = numpy.einsum('wrd,wrl->wdl', tmp, tensors_A[pos], optimize=True)

        # At this point, A_lsm is shape of (Nw, D, Nl)
        t2 = time.time()
        new_tensor, diff = __ridge_complex(Nw, D*linear_dim, A_lsm, y, model.alpha, x_tensors_l[pos], solver)
        t3 = time.time()
        if verbose >= 2:
            print("rest : time ", t2-t1, t3-t2)

        model.x_tensors_l[pos] = numpy.reshape(new_tensor, [D, linear_dim])

        return diff

    losss = []
    epochs = range(nite)
    loss = model.loss()
    rmses = []
    for epoch in epochs:
        # Optimize core tensor
        loss += update_core_tensor()

        #if numpy.linalg.norm(model.x_tensor_r) < min_norm:
            #if verbose > 2:
                #for i, x in enumerate(model.x_tensors):
                    #print("norm of x ", i, numpy.linalg.norm(x))
            #info = {}
            #info['losss'] = losss
            #return info

        assert not loss is None

        # Optimize the other tensors
        for pos in range(model.freq_dim):
            loss += update_l_tensor(pos)

            assert not loss is None

            #if numpy.linalg.norm(model.x_tensors[pos]) < min_norm:
                #info = {}
                #info['losss'] = losss
                #return info

        losss.append(loss)

        if epoch%20 == 0:
            loss = model.loss()
            rmses.append(numpy.sqrt(model.mse()))
            print("epoch = ", epoch, " loss = ", losss[-1], " rmse = ", rmses[-1], " alpha = ", model.alpha)
            if verbose > 0:
                print("epoch = ", epoch, " loss = ", losss[-1], " rmse = ", rmses[-1], " alpha = ", model.alpha)
                for i, x in enumerate(model.x_tensors):
                    print("norm of x ", i, numpy.linalg.norm(x))

        if len(rmses) > 2:
            diff_rmse = numpy.abs(rmses[-2] - rmses[-1])
            if diff_rmse < tol_rmse:
                break

        if optimize_alpha > 0:
            model.update_alpha(optimize_alpha)

    info = {}
    info['losss'] = losss

    return info


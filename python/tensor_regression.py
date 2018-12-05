from __future__ import print_function

from .regression import ridge_complex

import numpy
from scipy.linalg import LinAlgError
import scipy
from itertools import *
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
    #assert full_tensor.shape == numpy.TensorShape(dims)
    return full_tensor


class OvercompleteGFModel(object):
    """
    minimize |y - A * x|_2^2 + alpha * |x|_2^2
  
    A(n, r, l1, l2, ...) = U_1(n, r, l1) * U_2(n, r, l2) * ...
  
    A_dim: (Nw, Nr, l1, l2, ...)
    freq_dim is the number of l1, l2, ....
    
    x(d, r, l1, l2, ...) = \sum_d X_0(d,r) * X_1(d,l1) * X_2(d,l2) * ...
    """
    def __init__(self, Nw, Nr, freq_dim, linear_dim, tensors_A, y, alpha, D):
        self.right_dims = (Nr,) + (linear_dim,) * freq_dim
        self.right_dim = freq_dim + 1
    
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
    
        self.x_tensors = [create_tensor(D, self.right_dims[i]) for i in range(self.right_dim)]
        self.coeff = 1.0

    def var_list(self):
        """
        Return a list of model parameters
        """
        return self.x_tensors + [self.coeff]

    def full_tensor_x(self, x_tensors_plus_coeff=None):
        if x_tensors_plus_coeff == None:
            return self.coeff * cp_to_full_tensor(self.x_tensors)
        else:
            return x_tensors_plus_coeff[-1] * cp_to_full_tensor(x_tensors_plus_coeff[:-1])

    def predict_y(self, x_tensors_plus_coeff=None):
        """
        Predict y from self.x_tensors
    
        sum_d sum_{r,l1,l2,...} A(n, r, l1, l2, ...) * X_0(d,r) * X_1(d,l1) * X_2(d,l2) * ...
    
        We used A(n, r, l1, l2, ...) = U_1(n, r, l1) * U_2(n, r, l2) * ...
    
        We contract the tensors as follows:
            U_1(n, r, l1) * X_1(d, l1) -> UX_1(n, r, d)
            U_2(n, r, l2) * X_2(d, l2) -> UX_2(n, r, d)
        """
        if x_tensors_plus_coeff == None:
            x_tensors = self.x_tensors
            coeff = self.coeff
        else:
            x_tensors =x_tensors_plus_coeff[:-1]
            coeff = x_tensors_plus_coeff[-1]

        ones = numpy.full((self.Nw,),1)
        result = numpy.einsum('dr,n->nrd', x_tensors[0], ones)
        for i in range(1, self.right_dim):
            UX = numpy.einsum('nrl,dl->nrd', self.tensors_A[i-1], x_tensors[i])
            result *= UX
        # At this point, "result" is shape of (Nw, Nr, D).
        return coeff * numpy.sum(result, axis = (1, 2))

    def loss(self, x_tensors_plus_coeff=None):
        """
        Compute mean squared error + L2 regularization term
        """
        if x_tensors_plus_coeff == None:
            x_tensors = self.x_tensors
            coeff = self.coeff
        else:
            x_tensors =x_tensors_plus_coeff[:-1]
            coeff = x_tensors_plus_coeff[-1]

        y_pre = self.predict_y(x_tensors + [coeff])
        assert self.y.shape == y_pre.shape

        r = squared_L2_norm(self.y - y_pre)
        tmp = coeff**(2./len(x_tensors))
        for t in x_tensors:
            r += tmp * self.alpha * squared_L2_norm(t)
        return r/self.Nw

    def update_alpha(self, target_ratio=1e-8):
        """
        Update alpha so that L2 regularization term/residual term ~ target_ratio
        """
        x_tensors = self.x_tensors
        coeff = self.coeff
        assert coeff == 1.0

        y_pre = self.predict_y(x_tensors + [coeff])

        res = squared_L2_norm(self.y - y_pre)
        reg = numpy.sum([squared_L2_norm(t) for t in x_tensors])

        self.alpha = target_ratio * res/reg

        return self.alpha

    def mse(self, x_tensors_plus_coeff=None):
        """
        Compute mean squared error
        """
        if x_tensors_plus_coeff == None:
            x_tensors = self.x_tensors
            coeff = self.coeff
        else:
            x_tensors = x_tensors_plus_coeff[:-1]
            coeff = x_tensors_plus_coeff[-1]

        y_pre = self.predict_y(x_tensors + [coeff])
        assert self.y.shape == y_pre.shape
        return (squared_L2_norm(self.y - y_pre))/self.Nw


def ridge_complex_tf(N1, N2, A, y, alpha, x_old, solver='svd', precond=None, verbose=0):
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

    if loss_diff > 0 and numpy.abs(loss_diff) > 1e-8 * numpy.abs(new_loss):
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

def optimize_als(model, nite, tol_rmse = 1e-5, solver='svd', verbose=0, precond=None, min_norm=1e-8, optimize_alpha=-1):
    Nr = model.Nr
    D = model.D
    Nw = model.Nw
    freq_dim = model.freq_dim
    linear_dim = model.linear_dim
    tensors_A = model.tensors_A
    x_tensors = model.x_tensors
    coeff = model.coeff
    y = model.y

    # UXs is defined as follows.
    #for i in range(freq_dim):
    #    UXs[i, :, :, :] = numpy.einsum('nrl,dl->nrd', tensors_A[i], x_tensors[i+1])
    UXs = numpy.empty((freq_dim, Nw, Nr, D), dtype=complex)
    for i in range(freq_dim):
        UXs[i, :, :, :] = numpy.einsum('nrl,dl->nrd', tensors_A[i], x_tensors[i+1])

    def update_core_tensor():
        """
        Build a least squares model for optimizing core tensor
        """
        t1 = time.time()
        A_lsm = coeff * numpy.prod(UXs, axis=0)

        # Reshape A_lsm as (Nw, R, D) to (Nw, D, R)
        A_lsm = numpy.transpose(A_lsm, [0, 2, 1])
        t2 = time.time()
        new_core_tensor, diff = ridge_complex_tf(Nw, D * Nr, A_lsm, y, model.alpha, x_tensors[0], solver)
        t3 = time.time()
        if verbose >= 2:
            print("core : time ", t2-t1, t3-t2)
        model.x_tensors[0] = numpy.reshape(new_core_tensor, [D, Nr])

        return diff

    def update_l_tensor(pos):
        assert pos > 0

        t1 = time.time()
        mask = numpy.arange(freq_dim) != pos-1
        UX_prod = numpy.prod(UXs[mask, :, :, :], axis=0)
        UX_prod = numpy.einsum('nrd,dr->nrd', UX_prod, x_tensors[0])

        # This is slow part.
        A_lsm = coeff * numpy.sum(numpy.einsum('nrd,nrl->nrdl', UX_prod, tensors_A[pos-1]), axis=1)

        # At this point, A_lsm is shape of (Nw, D, Nl)
        t2 = time.time()
        new_tensor, diff = ridge_complex_tf(Nw, D*linear_dim, A_lsm, y, model.alpha, x_tensors[pos], solver)
        t3 = time.time()
        if verbose >= 2:
            print("rest : time ", t2-t1, t3-t2)

        model.x_tensors[pos] = numpy.reshape(new_tensor, [D, linear_dim])

        UXs[pos-1, :, :, :] = numpy.tensordot(tensors_A[pos-1], x_tensors[pos].transpose((1,0)), axes=(2,0))

        return diff

    losss = []
    epochs = range(nite)
    loss = model.loss()
    rmses = []
    for epoch in epochs:
        # Optimize core tensor
        loss += update_core_tensor()

        if numpy.linalg.norm(model.x_tensors[0]) < min_norm:
            if verbose > 2:
                for i, x in enumerate(model.x_tensors):
                    print("norm of x ", i, numpy.linalg.norm(x))
            info = {}
            info['losss'] = losss
            return info

        assert not loss is None

        # Optimize the other tensors
        for pos in range(1, model.freq_dim+1):
            loss += update_l_tensor(pos)

            assert not loss is None

            if numpy.linalg.norm(model.x_tensors[pos]) < min_norm:
                info = {}
                info['losss'] = losss
                return info

        losss.append(loss)

        if epoch%20 == 0:
            loss = model.loss()
            rmses.append(numpy.sqrt(model.mse()))
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


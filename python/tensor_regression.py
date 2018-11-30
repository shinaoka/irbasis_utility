from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from .regression import ridge_complex

tfe.enable_eager_execution()

import numpy as np
from itertools import *

real_dtype = tf.float64
cmplx_dtype = tf.complex128


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
        tildeA_reshaped = tf.reshape(tildeA, (Nw, Nr, -1))
        tildeA = tf.einsum('nrl,nrm->nrlm', tildeA_reshaped, tensors_A[i])
    
    full_tensor_dims = tuple([Nw, Nr] + [t.shape[-1] for t in tensors_A])
    return tf.reshape(tildeA, full_tensor_dims)


def squared_L2_norm(x):
    """
    Squared L2 norm
    """
    if x.dtype in [tf.float64, tf.float32, tf.float16]:
        return tf.reduce_sum(tf.multiply(x, x))
    elif x.dtype in [tf.complex128, tf.complex64]:
        return tf.reduce_sum(tf.real(tf.multiply(tf.conj(x), x)))
    else:
        raise RuntimeError("Unknown type: " + x.dtype)

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
        tildeT_reshaped = tf.reshape(tildeT, (D,-1))
        tildeT = tf.einsum('dI,di->dIi', tildeT_reshaped, x_tensors[i])
    full_tensor = tf.reduce_sum(tf.reshape(tildeT, (D,) + tuple(dims)), axis=0)
    assert full_tensor.shape == tf.TensorShape(dims)
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
    
        # Check shapes
        assert y.shape == tf.TensorShape([Nw])
        assert len(tensors_A) == freq_dim
        for t in tensors_A:
            assert t.shape == tf.TensorShape([Nw, Nr, linear_dim])
    
        self.y = tf.constant(y, dtype=cmplx_dtype)
        self.tensors_A = [tf.constant(t, dtype=cmplx_dtype) for t in tensors_A]
        self.alpha = alpha
        self.Nw = Nw
        self.Nr = Nr
        self.linear_dim = linear_dim
        self.freq_dim = freq_dim
        self.D = D
    
        def create_tensor(N, M):
            rand = np.random.rand(N, M) + 1J * np.random.rand(N, M)
            return tf.Variable(rand, dtype=cmplx_dtype)
    
        self.x_tensors = [create_tensor(D, self.right_dims[i]) for i in range(self.right_dim)]
        self.coeff = tf.Variable(1.0, dtype=real_dtype)

    def var_list(self):
        """
        Return a list of model parameters
        """
        return [self.x_tensors + self.coeff]

    def full_tensor_x(self, x_tensors_plus_coeff=None):
        if x_tensors_plus_coeff == None:
            return tf.cast(self.coeff, dtype=cmplx_dtype) * cp_to_full_tensor(self.x_tensors)
        else:
            return tf.cast(x_tensors_plus_coeff[-1], dtype=cmplx_dtype) * cp_to_full_tensor(x_tensors_plus_coeff[:-1])

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

        ones = tf.constant(np.full((self.Nw,),1), dtype=cmplx_dtype)
        result = tf.einsum('dr,n->nrd', x_tensors[0], ones)
        for i in range(1, self.right_dim):
            UX = tf.einsum('nrl,dl->nrd', self.tensors_A[i-1], x_tensors[i])
            result = tf.multiply(result, UX)
        # At this point, "result" is shape of (Nw, Nr, D).
        return tf.cast(coeff, dtype=cmplx_dtype) * tf.reduce_sum(result, axis = [1, 2])

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


def ridge_complex_tf(N1, N2, A, y, alpha, x_old, solver='svd', precond=None):
    # TODO: remove CPU code
    A_numpy = A.numpy().reshape((N1, N2))
    y_numpy = y.numpy().reshape((N1,))
    x_old_numpy = x_old.numpy().reshape((N2,))

    if solver == 'svd':
        x_numpy = ridge_complex(A_numpy, y_numpy, alpha, solver='svd')
    elif solver == 'lsqr':
        x_numpy = ridge_complex(A_numpy, y_numpy, alpha, solver='lsqr', precond=precond)
    else:
        raise RuntimeError("Unsupported solver: " + solver)

    x = tf.constant(x_numpy, dtype=cmplx_dtype)

    loss_diff =  np.linalg.norm(y_numpy - np.dot(A_numpy, x_numpy))**2 \
                 - np.linalg.norm(y_numpy - np.dot(A_numpy, x_old_numpy))**2 \
                 + alpha * np.linalg.norm(x_numpy)**2 \
                 - alpha * np.linalg.norm(x_old_numpy)**2

    if loss_diff > 0:
        print(np.linalg.norm(y_numpy - np.dot(A_numpy, x_numpy))**2)
        print(np.linalg.norm(y_numpy - np.dot(A_numpy, x_old_numpy))**2)
        print(alpha * np.linalg.norm(x_numpy)**2)
        print(alpha * np.linalg.norm(x_old_numpy)**2)
    assert loss_diff <= 0

    return x, loss_diff/N1

def __normalize_tensor(tensor):
    norm = tf.real(tf.norm(tensor))
    return tensor/tf.cast(norm, dtype=cmplx_dtype), norm

def optimize_als(model, nite, tol_rmse = 1e-5, solver='svd', verbose=0, precond=None):
    Nr = model.Nr
    D = model.D

    def update_core_tensor():
        """
        Build a least squares model for optimizing core tensor
        """
        # TODO: remove np.full()
        A_lsm = tf.fill([model.Nw, model.Nr, model.D], tf.cast(1.0, dtype=cmplx_dtype))
        for i in range(model.freq_dim):
            UX = tf.einsum('nrl,dl->nrd', model.tensors_A[i], model.x_tensors[i+1])
            A_lsm = tf.multiply(A_lsm , UX)

        # Reshape A_lsm as (Nw, R, D) to (Nw, D, R)
        A_lsm = tf.transpose(A_lsm, [0, 2, 1])
        A_lsm = tf.cast(model.coeff, dtype=cmplx_dtype) * A_lsm
        # This should be
        new_core_tensor, diff = ridge_complex_tf(model.Nw, model.D * model.Nr, A_lsm, model.y, model.alpha, model.x_tensors[0], solver)
        new_core_tensor = tf.reshape(new_core_tensor, [model.D, model.Nr])

        #new_core_tensor, norm = __normalize_tensor(new_core_tensor)
        #model.coeff.assign(model.coeff * norm)

        tf.assign(model.x_tensors[0], new_core_tensor)

        return diff

    def update_l_tensor(pos):
        assert pos > 0

        # TODO: remove np.full()
        UX_prod = tf.constant(np.full((model.Nw, model.Nr, model.D),1), dtype=cmplx_dtype)
        for i in range(model.freq_dim):
            if i + 1 == pos:
                continue
            UX = tf.einsum('nrl,dl->nrd', model.tensors_A[i], model.x_tensors[i+1])
            UX_prod = tf.multiply(UX_prod , UX)
        # Core tensor
        UX_prod = tf.einsum('nrd,dr->nrd', UX_prod, model.x_tensors[0])
        A_lsm = tf.reduce_sum(tf.einsum('nrd,nrl->nrdl', UX_prod, model.tensors_A[pos-1]), axis=1)
        A_lsm = tf.cast(model.coeff, dtype=cmplx_dtype) * A_lsm
        # At this point, A_lsm is shape of (Nw, D, Nl)
        new_tensor, diff = ridge_complex_tf(model.Nw, model.D*model.linear_dim, A_lsm, model.y, model.alpha, model.x_tensors[pos], solver)
        new_tensor = tf.reshape(new_tensor, [model.D, model.linear_dim])

        #new_tensor, norm = __normalize_tensor(new_tensor)
        #model.coeff.assign(model.coeff * norm)

        tf.assign(model.x_tensors[pos], new_tensor)

        return diff

    losss = []
    diff_losss = []
    epochs = range(nite)
    loss = model.loss().numpy()
    for epoch in epochs:
        # Optimize core tensor
        loss += update_core_tensor()

        #for x in model.x_tensors:
            #print("norm of x ", tf.norm(x))

        assert not loss is None
        assert loss >= 0

        # Optimize the other tensors
        for pos in range(1, model.freq_dim+1):
            loss += update_l_tensor(pos)

            assert not loss is None
            assert loss >= 0

        losss.append(loss)

        #print("epoch = ", epoch, " loss = ", losss[-1], " rmse = ", np.sqrt(model.mse()), model.coeff.numpy())
        if verbose > 0 and epoch%20 == 0:
            print("epoch = ", epoch, " loss = ", losss[-1], " rmse = ", np.sqrt(model.mse()), model.coeff.numpy())
            for i, x in enumerate(model.x_tensors):
                print("norm of x ", i, tf.norm(x).numpy())

        if len(losss) > 2:
            diff_losss.append(np.abs(losss[-2] - losss[-1]))
            if losss[-1] < tol_rmse**2 or np.abs(losss[-2] - losss[-1]) < tol_rmse**2:
                break

    info = {}
    info['losss'] = losss

    return info


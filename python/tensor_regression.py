from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

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
        self.freq_dim = freq_dim
        self.D = D
    
        def create_tensor(N, M):
            rand = np.random.rand(N, M) + 1J * np.random.rand(N, M)
            return tf.Variable(rand, dtype=cmplx_dtype)
    
        self.x_tensors = [create_tensor(D, self.right_dims[i]) for i in range(self.right_dim)]

    def var_list(self):
        """
        Return a list of model parameters
        """
        return self.x_tensors

    def full_tensor_x(self, x_tensors=None):
        if x_tensors == None:
            x_tensors = self.x_tensors
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
        if x_tensors == None:
            x_tensors = self.x_tensors
            
        ones = tf.constant(np.full((self.Nw,),1), dtype=cmplx_dtype)
        result = tf.einsum('dr,n->nrd', x_tensors[0], ones)
        for i in range(1, self.right_dim):
            UX = tf.einsum('nrl,dl->nrd', self.tensors_A[i-1], x_tensors[i])
            result = tf.multiply(result, UX)
        # At this point, "result" is shape of (Nw, Nr, D).
        return tf.reduce_sum(result, axis = [1, 2])

    def loss(self, x_tensors=None):
        """
        Compute mean squared error + L2 regularization term
        """
        if x_tensors == None:
            x_tensors = self.x_tensors
        y_pre = self.predict_y(x_tensors)
        assert self.y.shape == y_pre.shape
        return (squared_L2_norm(self.y - y_pre) + self.alpha * squared_L2_norm(self.full_tensor_x(x_tensors)))/self.Nw

    def mse(self, x_tensors=None):
        """
        Compute mean squared error
        """
        if x_tensors == None:
            x_tensors = self.x_tensors
        y_pre = self.predict_y(x_tensors)
        assert self.y.shape == y_pre.shape
        return (squared_L2_norm(self.y - y_pre))/self.Nw


def optimize(model, nite, learning_rate = 0.001, tol_rmse = 1e-5, verbose=0):
    def loss_f():
        loss = model.loss()
        losss.append(loss)
        return loss
    
    def evaluate_loss(model, grads, stepsize):
        var_list = [tf.Variable(v) for v in model.var_list()]
        for i in range(len(var_list)):
            var_list[i].assign_sub(stepsize * grads[i])
        return model.loss(var_list)

    current_learning_rate = learning_rate
    learning_rate_fact = 10.0
        
    losss = []
    diff_losss = []
    epochs = range(nite)
    for epoch in epochs:
        # Compute gradients
        with tf.GradientTape() as tape:
            loss = loss_f()
        grads = tape.gradient(loss, model.var_list())
        if verbose > 0 and epoch%10 == 0:
            print("epoch = ", epoch, " loss = ", loss)
    
        # Simple line search algorithm
        #stepsizes = [0.01*learning_rate, 0.1*learning_rate, learning_rate, 10*learning_rate, 100*learning_rate, 1000*learning_rate]
        stepsizes = [current_learning_rate]
        losss_ls = {}
        while True:
            for s in stepsizes:
                if not s in losss_ls:
                    losss_ls[s] = evaluate_loss(model, grads, s).numpy()
            opt_stepsize = min(losss_ls, key = lambda x: losss_ls.get(x))
            opt_loss = losss_ls[opt_stepsize]
            #print(loss, opt_loss, opt_stepsize, losss_ls)

            if opt_loss > loss:
                # step sizes are too big.
                stepsizes.append(np.amin(stepsizes) / learning_rate_fact)
            elif opt_stepsize == np.amax(stepsizes):
                # There is a chance to use a larger step size
                stepsizes.append(np.amax(stepsizes) * learning_rate_fact)
            else:
                break

        # Update x
        #print("debug ", epoch, loss.numpy(), opt_stepsize)
        current_learning_rate = opt_stepsize
        var_list = model.var_list()
        for i in range(len(var_list)):
            var_list[i].assign_sub(opt_stepsize * grads[i])
    
        if len(losss) > 2:
            diff_losss.append(np.abs(losss[-2] - losss[-1]))
            if losss[-1] < tol_rmse**2 or np.abs(losss[-2] - losss[-1]) < tol_rmse**2:
                break

    info = {}
    info['losss'] = losss

    return info

from __future__ import print_function

import numpy
from .tensor_network import Tensor, TensorNetwork, conj_a_b, differenciate

class AutoALS:
    """
    Automated alternating least squares fitting of tensor network
    """
    def __init__(self, y, tilde_y, target_tensors, verbose=False):
        """

        :param y: TensorNetwork
            Tensor network to be fitted (constant during optimization)
        :param tilde_y: TensorNetwork
            Tensor network for fitting y
        :param target_tensors: list of Tensor
            Name of tensors in tilde_y to be optimized
        """

        assert isinstance(y, TensorNetwork)
        assert isinstance(tilde_y, TensorNetwork)

        if tilde_y.external_subscripts != y.external_subscripts:
            ##print("AAA", tilde_y.external_subscripts)
            #print("BBB", y.external_subscripts)
            raise RuntimeError("Subscripts for external indices of y and tilde_y do not match")

        # No tensor must be conjugate.
        if numpy.any([t.is_conj for t in y.tensors]):
            raise RuntimeError("Some tensor in y is conjugate.")
        if numpy.any([t.is_conj for t in tilde_y.tensors]):
            raise RuntimeError("Some tensor in tilde_y is conjugate.")
        if numpy.any([t.is_conj for t in target_tensors]):
            raise RuntimeError("No target tensor can be conjugate.")

        if numpy.sum([t for t in y.tensors if t in target_tensors]) > 0:
            raise RuntimeError("y must not contain a target tensor.")
        if numpy.sum([t for t in tilde_y.tensors if t in target_tensors]) == 0:
            raise RuntimeError("tilde_y contains no target tensor.")

        # Tensor network representation of <tilde_y | y>
        self._ctildey_y = conj_a_b(tilde_y, y)

        # Tensor network representation of <tilde_y | tilde_y>
        self._ctildey_tildey = conj_a_b(tilde_y, tilde_y)

        # ALS fitting matrix
        self._num_tensors_opt = len(target_tensors)
        self._target_tensors = target_tensors
        self._ys = [differenciate(self._ctildey_y, t.conjugate()) for t in target_tensors]
        self._As = [differenciate(self._ctildey_tildey, [t.conjugate(), t]) for t in target_tensors]

    def fit(self, niter, tensors_value):
        """
        Perform ALS fitting

        :param niter: int
            Number of iterations
        :param tensor_values: dict of (str, ndarray)
            Values of tensors. Those of target tensors will be updated.
        """

        for iter in range(niter):
           for idx_t in range(self._num_tensors_opt):
               print("iter", iter, idx_t)
               mat_A = self._As[idx_t](tensors_value)
               vec_y = self._ys[idx_t](tensors_value).ravel()
               N = len(vec_y)
               tensor_name = self._target_tensors[idx_t].name
               tensors_value[tensor_name][:] = numpy.linalg.solve(mat_A.reshape((N, N)), vec_y).reshape(tensors_value[tensor_name].shape)

from __future__ import print_function

import numpy
from .tensor_network import Tensor, TensorNetwork, conj_a_b, differenciate, from_int_to_char_subscripts, _unique_order_preserved
from scipy.sparse.linalg import LinearOperator, lgmres

def _create_linear_operator(op_array, op_subs, left_subs, right_subs, dims):
    op_subs_str = ''.join(op_subs)
    left_subs_str = ''.join(left_subs)
    right_subs_str = ''.join(right_subs)

    matvec_str = '{},{}->{}'.format(op_subs_str, right_subs_str, left_subs_str)
    rmatvec_str = '{},{}->{}'.format(op_subs_str, left_subs_str, right_subs_str)

    def matvec(v):
        return numpy.einsum(matvec_str, op_array, v.reshape(dims), optimize=True)

    def rmatvec(v):
        return numpy.einsum(rmatvec_str, op_array, v.reshape(dims).conjugate(), optimize=True).conjugate()

    N = numpy.prod(dims)
    return LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec)


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
        self._ys = []
        for t in target_tensors:
            diff = differenciate(self._ctildey_y, t.conjugate())
            diff.find_contraction_path()
            self._ys.append(diff)

        self._opAs = []
        for t in target_tensors:
            diff = differenciate(self._ctildey_tildey, [t.conjugate(), t])
            diff.find_contraction_path()
            tc_subs = self._ctildey_tildey.tensor_subscripts(t.conjugate())
            t_subs = self._ctildey_tildey.tensor_subscripts(t)

            A_subs = diff.external_subscripts
            char_subs = from_int_to_char_subscripts([A_subs, tc_subs, t_subs])
            self._opAs.append(lambda x: _create_linear_operator(diff.evaluate(x), *char_subs, t.shape))

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
               vec_y = self._ys[idx_t].evaluate(tensors_value).ravel()
               N = len(vec_y)
               tensor_name = self._target_tensors[idx_t].name
               opA = self._opAs[idx_t](tensors_value)
               # TODO: set appropriate tol
               r = lgmres(opA, vec_y)
               tensors_value[tensor_name][:] = r[0].reshape(tensors_value[tensor_name].shape)

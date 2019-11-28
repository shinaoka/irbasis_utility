from __future__ import print_function

import numpy
from .tensor_network import Tensor, TensorNetwork, conj_a_b, differenciate, from_int_to_char_subscripts, _unique_order_preserved
from scipy.sparse.linalg import LinearOperator, lgmres

class LSSolver(object):
    """
    Solve linear system for one tensor
    TODO: mpi parallelization
    """
    def __init__(self, target_tensor, ctildey_y, ctildey_tildey):
        # Tensor network for y
        self._y_tn = differenciate(ctildey_y, target_tensor.conjugate())
        self._y_tn.find_contraction_path()

        # Tensor network for A
        self._A_tn = differenciate(ctildey_tildey, [target_tensor.conjugate(), target_tensor])
        self._A_tn.find_contraction_path()
        tc_subs = ctildey_tildey.tensor_subscripts(target_tensor.conjugate())
        t_subs = ctildey_tildey.tensor_subscripts(target_tensor)

        A_subs = self._A_tn.external_subscripts
        op_subs, left_subs, right_subs = from_int_to_char_subscripts([A_subs, tc_subs, t_subs])

        op_subs_str = ''.join(op_subs)
        left_subs_str = ''.join(left_subs)
        right_subs_str = ''.join(right_subs)

        self._matvec_str = '{},{}->{}'.format(op_subs_str, right_subs_str, left_subs_str)
        self._rmatvec_str = '{},{}->{}'.format(op_subs_str, left_subs_str, right_subs_str)
        self._dims = target_tensor.shape

    def solve(self, tensors_value):
        op_array = self._A_tn.evaluate(tensors_value)

        matvec = lambda v: numpy.einsum(self._matvec_str, op_array, v.reshape(self._dims), optimize=True)
        rmatvec = lambda v : numpy.einsum(self._rmatvec_str, op_array, v.reshape(self._dims).conjugate(), optimize=True).conjugate()

        N = numpy.prod(self._dims)
        opA = LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec)

        vec_y = self._y_tn.evaluate(tensors_value).ravel()
        r = lgmres(opA, vec_y)

        return r[0].reshape(self._dims)


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

        if numpy.count_nonzero([t in target_tensors for t in y.tensors]) > 0:
            raise RuntimeError("y must not contain a target tensor.")
        if numpy.count_nonzero([t in target_tensors for t in tilde_y.tensors]) == 0:
            raise RuntimeError("tilde_y contains no target tensor.")

        # Tensor network representation of <tilde_y | y>
        self._ctildey_y = conj_a_b(tilde_y, y)
        self._ctildey_y.find_contraction_path()

        # Tensor network representation of <tilde_y | tilde_y>
        self._ctildey_tildey = conj_a_b(tilde_y, tilde_y)
        self._ctildey_tildey.find_contraction_path()

        # Residual
        # <tilde_y - y|tilde_y - y> = <tilde_y|tilde_y> - <tilde_y|y> - <y|tilde_y> + <y|y>
        self._cy_y = conj_a_b(y, y)
        self._cy_y.find_contraction_path()

        self._num_tensors_opt = len(target_tensors)
        self._target_tensors = target_tensors

        self._solvers = []
        for t in target_tensors:
            self._solvers.append(LSSolver(t, self._ctildey_y, self._ctildey_tildey))


    def squared_error(self, tensors_value):
        """
        Compute squared error

        :param tensor_value: dict of (str, ndarray)
           Values of tensors (input)
        :return:
           Squared error
        """
        return self._ctildey_tildey.evaluate(tensors_value) + self._cy_y.evaluate(tensors_value)  - 2 * self._ctildey_y.evaluate(tensors_value).real


    def fit(self, niter, tensors_value):
        """
        Perform ALS fitting

        :param niter: int
            Number of iterations
        :param tensor_value: dict of (str, ndarray)
            Values of tensors. Those of target tensors will be updated.
        """

        for iter in range(niter):
           for idx_t in range(self._num_tensors_opt):
               tensor_name = self._target_tensors[idx_t].name
               tensors_value[tensor_name][:] = self._solvers[idx_t].solve(tensors_value)

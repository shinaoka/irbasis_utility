from __future__ import print_function

import numpy
from .tensor_network import Tensor, TensorNetwork, conj_a_b, differenciate, from_int_to_char_subscripts
from scipy.sparse.linalg import LinearOperator, lgmres

def _mpi_split(work_size, comm_size):
    base = work_size // comm_size
    leftover = int(work_size % comm_size)

    sizes = numpy.ones(comm_size, dtype=int) * base
    sizes[:leftover] += 1

    offsets = numpy.zeros(comm_size, dtype=int)
    offsets[1:] = numpy.cumsum(sizes)[:-1]

    return sizes, offsets

class LSSolver(object):
    """
    Solve linear system for one tensor
    """
    def __init__(self, target_tensor, ctildey_y, ctildey_tildey, comm, distributed, parallel_solver):
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

        self._comm = comm
        self._distributed = distributed
        self._parallel_solver = parallel_solver

        self._op_is_matrix = (left_subs + right_subs == op_subs)

    def solve(self, tensors_value):
        op_array = self._A_tn.evaluate(tensors_value)
        vec_y = self._y_tn.evaluate(tensors_value).ravel()
        if self._distributed:
            op_array[:] = self._comm.allreduce(op_array)
            vec_y[:] = self._comm.allreduce(vec_y)

        N = numpy.prod(self._dims)
        if self._parallel_solver and self._op_is_matrix:
            from mpi4py import MPI

            mpi_type = MPI.COMPLEX if numpy.iscomplexobj(op_array) else MPI.DOUBLE

            rank = self._comm.Get_rank()
            sizes, offsets = _mpi_split(N, self._comm.Get_size())
            start, end = offsets[rank], offsets[rank] + sizes[rank]
            A = op_array.reshape((N, N))[start:end, :]
            conjA = (op_array.reshape((N, N))[:, start:end]).conjugate().transpose()
            if numpy.amin(sizes) == 0:
                raise RuntimeError("sizes contains 0!")

            def matvec(v):
                recv = numpy.empty(N, dtype=A.dtype)
                self._comm.Allgatherv(numpy.dot(A,v).ravel(), [recv, sizes, offsets, mpi_type])
                return recv

            def rmatvec(v):
                recv = numpy.empty(N, dtype=A.dtype)
                self._comm.Allgatherv(numpy.dot(conjA,v).ravel(), [recv, sizes, offsets, mpi_type])
                return recv

            opA = LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec)
            r = lgmres(opA, vec_y)
            return r[0].reshape(self._dims)
        else:
            matvec = lambda v: numpy.einsum(self._matvec_str, op_array, v.reshape(self._dims), optimize=True)
            rmatvec = lambda v : numpy.einsum(self._rmatvec_str, op_array, v.reshape(self._dims).conjugate(), optimize=True).conjugate()
            opA = LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec)
            r = lgmres(opA, vec_y)
            return r[0].reshape(self._dims)


class AutoALS:
    """
    Automated alternating least squares fitting of tensor network
    """
    def __init__(self, y, tilde_y, target_tensors, verbose=False, comm=None, distributed_subscript=None):
        """

        :param y: TensorNetwork
            Tensor network to be fitted (constant during optimization)
        :param tilde_y: TensorNetwork
            Tensor network for fitting y
        :param target_tensors: list of Tensor
            Name of tensors in tilde_y to be optimized
        :param comm:
            MPI communicator.
            If comm is not None, MPI is enabled.
            If MPI is enabled, ONE of the external indices of y and tilde_y is distributed on different processes.
            That index can have different sizes on different processes.
        :param distributed_subscript: Integer
            Subscript of the distributed index
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

        self._comm = comm

        # Check shape
        if not distributed_subscript is None:
            self._distributed = True
            if self._comm is None:
                raise RuntimeError("Enable MPI!")
            # Check all dimensions but distributed subscript
            if not distributed_subscript in y.external_subscripts:
                raise RuntimeError("Set correct distributed_subscript!")
            idx = y.external_subscripts.index(distributed_subscript)
            shapes = self._comm.allgather(numpy.delete(numpy.array(y.shape), idx))
            if not numpy.all(shapes[0] == shapes):
                raise RuntimeError("Shape mismatch of y tensor on different nodes!")
        else:
            self._distributed = False

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
            self._solvers.append(LSSolver(t, self._ctildey_y, self._ctildey_tildey, self._comm, self._distributed, not self._comm is None))
            #self._solvers.append(LSSolver(t, self._ctildey_y, self._ctildey_tildey, self._comm, self._distributed, False))


    def squared_error(self, tensors_value):
        """
        Compute squared error

        :param tensor_value: dict of (str, ndarray)
           Values of tensors (input)
        :return:
           Squared error
        """

        if self._comm is None:
            return self._ctildey_tildey.evaluate(tensors_value) + self._cy_y.evaluate(tensors_value)  - 2 * self._ctildey_y.evaluate(tensors_value).real
        else:
            return self._comm.allreduce(self._ctildey_tildey.evaluate(tensors_value) + self._cy_y.evaluate(tensors_value)  - 2 * self._ctildey_y.evaluate(tensors_value).real)

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

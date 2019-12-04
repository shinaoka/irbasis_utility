from __future__ import print_function

import numpy
from .tensor_network import TensorNetwork, conj_a_b, differenciate, from_int_to_char_subscripts
from scipy.sparse.linalg import LinearOperator, cg

def _mpi_split(work_size, comm_size):
    base = work_size // comm_size
    leftover = int(work_size % comm_size)

    sizes = numpy.ones(comm_size, dtype=int) * base
    sizes[:leftover] += 1

    offsets = numpy.zeros(comm_size, dtype=int)
    offsets[1:] = numpy.cumsum(sizes)[:-1]

    return sizes, offsets

class LeastSquaresOpGenerator(object):
    """
    Coefficient matrix of a least squares problem
    """
    def __init__(self, target_tensor, term, comm, distributed, parallel_solver):
        assert not target_tensor.is_conj
        # Tensor network for A
        self._A_tn = differenciate(term, [target_tensor.conjugate(), target_tensor])
        self._A_tn.find_contraction_path()
        tc_subs = term.tensor_subscripts(target_tensor.conjugate())
        t_subs = term.tensor_subscripts(target_tensor)

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

    def construct(self, tensors_value):
        op_array = self._A_tn.evaluate(tensors_value)
        if self._distributed:
            op_array[:] = self._comm.allreduce(op_array)

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

            return LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec)
        else:
            matvec = lambda v: numpy.einsum(self._matvec_str, op_array, v.reshape(self._dims), optimize=True).ravel()
            rmatvec = lambda v : numpy.einsum(self._rmatvec_str, op_array, v.reshape(self._dims).conjugate(), optimize=True).conjugate().ravel()
            return LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec)


def _sum_ops_flipped_sign(ops):
    # Compute sum of linear operators of the same shape and flip the sign
    matvec = lambda v: -sum([coeff * o.matvec(v) for coeff, o in ops])
    rmatvec = lambda v: -sum([numpy.conj(coeff) * o.rmatvec(v) for coeff, o in ops])
    N1, N2 = ops[0][1].shape
    return LinearOperator((N1, N2), matvec=matvec, rmatvec=rmatvec)


class LinearOperatorGenerator(object):
    """
    Differentiate sum of terms representing scalar tensors and construct a generator of LinearOperator
    """
    def __init__(self, target_tensor, terms, comm, distributed, parallel_solver):
        self._generators = []
        for coeff, term in terms:
            if term.has(target_tensor) and term.has(target_tensor.conjugate()):
                genA = LeastSquaresOpGenerator(target_tensor, term, comm, distributed, parallel_solver)
                self._generators.append((coeff, genA))

    def construct(self, tensors_value):
        # Note sign is flipped
        return _sum_ops_flipped_sign([(coeff, genA.construct(tensors_value)) for coeff, genA in self._generators])


class VectorGenerator(object):
    """
    Differentiate sum of terms representing scalar tensors and construct a generator of vector to be fitted
    """
    def __init__(self, target_tensor, terms, comm, distributed):
        self._y_tn = []
        for coeff, term in terms:
            if (not term.has(target_tensor)) and term.has(target_tensor.conjugate()):
                diff = differenciate(term, target_tensor.conjugate())
                diff.find_contraction_path()
                self._y_tn.append((coeff, diff))
        self._comm = comm
        self._distributed = distributed

    def construct(self, tensors_value):
        # Evaluate vector to be fitted
        vec_y = sum([coeff * t.evaluate(tensors_value) for coeff, t in self._y_tn]).ravel()
        if self._distributed:
            vec_y[:] = self._comm.allreduce(vec_y)
        return vec_y


def _eval_terms(terms, tensors_value):
    return numpy.sum([coeff * t.evaluate(tensors_value) for coeff, t in terms])


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
        :param distributed_subscript: Integer
            Subscript of the distributed index
            If this is not None, ONE of the external indices of y and tilde_y is distributed on different MPI nodes.
            That index can have different sizes on different processes.
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

        def dot_product(a, b):
            ab = conj_a_b(a[1], b[1])
            ab.find_contraction_path()
            return (numpy.conj(a[0])*b[0], ab)

        # Tensor networks representing squared errors
        all_terms = []
        self._diagonal_terms = []
        self._half_offdiag_terms = []

        ket_tensors = [(1,tilde_y), (-1,y)]
        for i, lt in enumerate(ket_tensors):
            for j, rt in enumerate(ket_tensors):
                dp = dot_product(lt, rt)
                if i < j:
                    self._half_offdiag_terms.append(dp)
                elif i==j:
                    self._diagonal_terms.append(dp)
                all_terms.append(dp)

        self._A_generators = {}
        self._y_generators = {}
        for t in target_tensors:
            self._A_generators[t.name] = LinearOperatorGenerator(t, all_terms, comm, self._distributed, not self._comm is None)
            self._y_generators[t.name] = VectorGenerator(t, all_terms, comm, self._distributed)

        self._num_tensors_opt = len(target_tensors)
        self._target_tensors = target_tensors


    def squared_error(self, tensors_value):
        """
        Compute squared error

        :param tensor_value: dict of (str, ndarray)
           Values of tensors (input)
        :return:
           Squared error
        """

        local_se = _eval_terms(self._diagonal_terms, tensors_value) + 2 * numpy.real(_eval_terms(self._half_offdiag_terms, tensors_value))
        if self._comm is None:
            return local_se
        else:
            return self._comm.allreduce(local_se)

    def fit(self, niter, tensors_value):
        """
        Perform ALS fitting

        :param niter: int
            Number of iterations
        :param tensor_value: dict of (str, ndarray)
            Values of tensors. Those of target tensors will be updated.
        """

        for iter in range(niter):
            for target_tensor in self._target_tensors:
                name = target_tensor.name
                opA = self._A_generators[name].construct(tensors_value)
                vec_y = self._y_generators[name].construct(tensors_value)
                # Note: A is a hermitian.
                r = cg(opA, vec_y)
                tensors_value[name][:] = r[0].reshape(tensors_value[name].shape)
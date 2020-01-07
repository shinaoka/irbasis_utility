from __future__ import print_function

import numpy
from collections import ChainMap
from copy import deepcopy
from .tensor_network import Tensor, TensorNetwork, conj_a_b, differentiate, from_int_to_char_subscripts
from scipy.sparse.linalg import LinearOperator, lgmres, aslinearoperator, LinearOperator
import sys
import time
from threadpoolctl import threadpool_info, threadpool_limits

def _mpi_split(work_size, comm_size):
    base = work_size // comm_size
    leftover = int(work_size % comm_size)

    sizes = numpy.ones(comm_size, dtype=int) * base
    sizes[:leftover] += 1

    offsets = numpy.zeros(comm_size, dtype=int)
    offsets[1:] = numpy.cumsum(sizes)[:-1]

    return sizes, offsets

def _is_list_of_instance(object, classinfo):
    """
    Return True if the object is a list of instances of the classinfo argument.
    """
    return isinstance(object, list) and numpy.all([isinstance(o, classinfo) for o in object])

def _trans_axes(src_subs, dst_subs):
    return tuple([src_subs.index(s) for s in dst_subs])

def _trans_axes_diag(src_subs, left_subs):
    return tuple([src_subs.index(s) for s in left_subs])

class MatrixLinearOperator(LinearOperator):
    # From https://github.com/scipy/scipy/blob/v1.4.1/scipy/sparse/linalg/interface.py
    def __init__(self, A):
        super(MatrixLinearOperator, self).__init__(A.dtype, A.shape)
        self.A = A
        self.__adj = None
        self.args = (A,)

    def _matmat(self, X):
        return self.A.dot(X)

    def _adjoint(self):
        if self.__adj is None:
            self.__adj = _AdjointMatrixLinearOperator(self)
        return self.__adj


class _AdjointMatrixLinearOperator(MatrixLinearOperator):
    def __init__(self, adjoint):
        self.A = adjoint.A.T.conj()
        self.__adjoint = adjoint
        self.args = (adjoint,)
        self.shape = adjoint.shape[1], adjoint.shape[0]

    @property
    def dtype(self):
        return self.__adjoint.dtype

    def _adjoint(self):
        return self.__adjoint


class DiagonalLinearOperator(LinearOperator):
    """
    Thin wrapper of scipy.sparse.linalg.LinearOperator

    :param is_diagonal: Bool
        Whether operator is diagonal or not.

    """
    def __init__(self, shape, diagonals):
        super(DiagonalLinearOperator, self).__init__(diagonals.dtype, shape)
        self._diagonals = diagonals

    def _matvec(self, x):
        return self._diagonals * x

    def _rmatvec(self, x):
        return numpy.conj(self._diagonals) * x

    @property
    def diagonals(self):
        return self._diagonals


class LeastSquaresOpGenerator(object):
    """
    Coefficient matrix of a least squares problem
    """
    def __init__(self, target_tensor, term, comm, distributed, verbose=False, mem_limit=1E+19):
        """

        :param target_tensor: Tensor
            "term" is differentiated by "target_tensor".
        :param term: TensorNetwork
             The tensor network to be differentiated.
        :param comm:
             MPI communicator
        :param distributed: bool
             If True, the linear operator is summed over all MPI nodes.
        """
        assert not target_tensor.is_conj

        self._target_tensor = target_tensor
        self._comm = comm
        self._rank = 0 if self._comm is None else self._comm.Get_rank()
        self._distributed = distributed

        # Tensor network for A
        self._A_tn = differentiate(term, [target_tensor.conjugate(), target_tensor])
        self._A_tn.find_contraction_path(verbose, mem_limit)
        tc_subs = term.tensor_subscripts(target_tensor.conjugate())
        t_subs = term.tensor_subscripts(target_tensor)

        A_subs = self._A_tn.external_subscripts
        op_subs, left_subs, right_subs = from_int_to_char_subscripts([A_subs, tc_subs, t_subs])

        # Dense matrix 
        self._op_is_matrix = len(left_subs) + len(right_subs) == len(op_subs)

        # Diagonal
        self._op_is_diagonal = set(left_subs) == set(right_subs)

        if self._op_is_matrix and self._op_is_diagonal:
            # For instance, operator is a sclar, treat this operator as a diagonal one.
            self._op_is_matrix = False

        if self._op_is_matrix:
            # As dense matrix
            # Transpose axes of matrix
            self._trans_axes = _trans_axes(op_subs, left_subs + right_subs)
        elif self._op_is_diagonal:
            assert left_subs == right_subs
            self._trans_axes_diag = _trans_axes_diag(op_subs, left_subs)
        else:
            op_subs_str = ''.join(op_subs)
            left_subs_str = ''.join(left_subs)
            right_subs_str = ''.join(right_subs)
            self._matvec_str = '{},{}->{}'.format(op_subs_str, right_subs_str, left_subs_str)
            self._rmatvec_str = '{},{}->{}'.format(op_subs_str, left_subs_str, right_subs_str)

        self._dims = target_tensor.shape
        self._size = numpy.prod(self._dims)

    @property
    def size(self):
        return self._size

    def construct(self, tensors_value):
        if self._comm is not None:
            from mpi4py import MPI

        t1 = time.time()
        op_array = self._A_tn.evaluate(tensors_value)
        t2 = time.time()
        if self._distributed:
            recv = numpy.empty_like(op_array)
            self._comm.Reduce(op_array, recv, MPI.SUM, root=0)
            if self._rank == 0:
                op_array = recv
            else:
                op_array[...] = 0.0
        t3 = time.time()
        if self._rank == 0:
            print('     contraction= ', t2-t1, ', Reduce= ', t3-t2)
        N = self._size

        if self._op_is_matrix:
            op_array = op_array.transpose(self._trans_axes)
            A = op_array.reshape((N, N))
            return MatrixLinearOperator(A)
        elif self._op_is_diagonal:
            return DiagonalLinearOperator((N, N), diagonals=op_array.transpose(self._trans_axes_diag))
        else:
            matvec = lambda v: numpy.einsum(self._matvec_str, op_array, v.reshape(self._dims), optimize=True).ravel()
            rmatvec = lambda v : numpy.einsum(self._rmatvec_str, op_array, v.reshape(self._dims).conjugate(), optimize=True).conjugate().ravel()
            return LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec)

def _sum_ops(ops):
    # Sum of linear operators
    # If each operator is a matrix or diagonal and at least one operator is a matrix, evaluate the sum as a dense matrix.
    # If all operators are diagonal, return diagonal elements.
    # Otherwise, construct a linear representing the sum.
    num_matrix_op = numpy.sum([isinstance(o, MatrixLinearOperator) for _, o in ops])
    is_all_diagonal = numpy.all([isinstance(o, DiagonalLinearOperator) for _, o in ops])

    def op_dtype():
        dtype = numpy.dtype(numpy.float64)
        for coeff, o in ops:
            dtype = numpy.promote_types(dtype, numpy.min_scalar_type(coeff))
            dtype = numpy.promote_types(dtype, o.dtype)
        return dtype

    if is_all_diagonal:
        dtype = op_dtype()
        N = ops[0][1].shape[0]
        diagonals = numpy.zeros((N,), dtype=dtype)
        for coeff, o in ops:
            diagonals += coeff * o.diagonals
        return diagonals
    elif num_matrix_op > 0 and numpy.all([isinstance(o, MatrixLinearOperator) or isinstance(o, DiagonalLinearOperator) for _, o in ops]):
        dtype = op_dtype()
        N = ops[0][1].shape[0]
        A = numpy.zeros((N, N), dtype=dtype)
        for coeff, o in ops:
            if isinstance(o, MatrixLinearOperator):
                A += coeff * o.A
            elif isinstance(o, DiagonalLinearOperator):
                A += coeff * numpy.diag(o.diagonals)
            else:
               raise RuntimeError("Something got wrong!")
        return A
    else:
        matvec = lambda v: sum([coeff * o.matvec(v) for coeff, o in ops])
        rmatvec = lambda v: sum([numpy.conj(coeff) * o.rmatvec(v) for coeff, o in ops])
        N1, N2 = ops[0][1].shape
        return LinearOperator((N1, N2), matvec=matvec, rmatvec=rmatvec)

def _identity_operator(N):
    return DiagonalLinearOperator((N, N), diagonals=numpy.ones(N))

def _is_matrix_operator(op):
    return isinstance(op, MatrixLinearOperator)

class LinearOperatorGenerator(object):
    """
    Differentiate sum of terms representing scalar tensors and construct a generator of LinearOperator
    """
    def __init__(self, target_tensor, terms, comm, distributed, reg_L2, verbose, mem_limit):
        self._generators = []
        for coeff, term in terms:
            if term.has(target_tensor) and term.has(target_tensor.conjugate()):
                genA = LeastSquaresOpGenerator(target_tensor, term, comm, distributed, verbose, mem_limit)
                self._generators.append((coeff, genA))
        self._reg_L2 = reg_L2

    def construct(self, tensors_value):
        ops = [(coeff, genA.construct(tensors_value)) for coeff, genA in self._generators]
        if self._reg_L2 != 0:
            N = self._generators[0][1].size
            ops.append((self._reg_L2, _identity_operator(N)))
        return _sum_ops(ops)

class VectorGenerator(object):
    """
    Differentiate sum of terms representing scalar tensors and construct a generator of vector to be fitted
    """
    def __init__(self, target_tensor, terms, comm, distributed):
        self._y_tn = []
        self._trans_axes = []
        for coeff, term in terms:
            if (not term.has(target_tensor)) and term.has(target_tensor.conjugate()):
                diff = differentiate(term, target_tensor.conjugate())
                diff.find_contraction_path()
                t_subs = term.tensor_subscripts(target_tensor.conjugate())
                trans_axes = _trans_axes(diff.external_subscripts, t_subs)
                self._y_tn.append((coeff, diff, trans_axes))
        self._comm = comm
        self._distributed = distributed

    def construct(self, tensors_value):
        # Evaluate vector to be fitted
        vec_y = sum([coeff * t.evaluate(tensors_value).transpose(axes) for coeff, t, axes in self._y_tn]).ravel()
        if self._distributed:
            vec_y[:] = self._comm.allreduce(vec_y)
        return vec_y


def _eval_terms(terms, tensors_value):
    return numpy.sum([coeff * t.evaluate(tensors_value) for coeff, t in terms])


class AutoALS:
    """
    Automated alternating least squares fitting of tensor network
    """
    def __init__(self, Y, tilde_Y, target_tensors, reg_L2=0.0, comm=None, distributed_subscript=None, mem_limit=1E+19, num_threads=1):
        """

        :param Y: a list of TensorNetwork
            Tensor networks to be fitted (constant during optimization)
        :param tilde_Y: a list of TensorNetwork
            Fitting tensor networks
        :param target_tensors: a list of Tensor
            Name of tensors in tilde_y to be optimized
        :param reg_L2: float
            L2 regularization parameter
        :param comm:
            MPI communicator.
            If comm is not None, MPI is enabled.
        :param distributed_subscript: Integer
            Subscript of the distributed index
            If this is not None, ONE of the external indices of y and tilde_y is distributed on different MPI nodes.
            That index can have different sizes on different processes.
        :param mem_limit: Integer
            mememory limit for einsum_path. 1E+8 * 16 Byte = 1.6 GB
        :param num_threads: Integer
            Number of threads set for blas backend of numpy when solving linear systems.
        """

        if isinstance(Y, TensorNetwork):
            Y = [Y]
        if isinstance(tilde_Y, TensorNetwork):
            tilde_Y = [tilde_Y]

        assert _is_list_of_instance(Y, TensorNetwork)
        assert _is_list_of_instance(tilde_Y, TensorNetwork)
        assert _is_list_of_instance(target_tensors, Tensor)

        if len(set([y.external_subscripts for y in Y + tilde_Y])) > 1:
            raise RuntimeError("Subscripts for external indices are not identical.")

        self._mem_limit = 1E+20 if mem_limit is None else mem_limit

        # No tensor must be conjugate.
        for y in Y:
            if numpy.any([t.is_conj for t in y.tensors]):
                raise RuntimeError("Some tensor in Y is conjugate.")
            if numpy.count_nonzero([t in target_tensors for t in y.tensors]) > 0:
                raise RuntimeError("y must not contain a target tensor.")
        for tilde_y in tilde_Y:
            if numpy.any([t.is_conj for t in tilde_y.tensors]):
                raise RuntimeError("Some tensor in tilde_Y is conjugate.")
            if numpy.count_nonzero([t in target_tensors for t in tilde_y.tensors]) == 0:
                raise RuntimeError("tilde_y contains no target tensor. target_tensors={}, tilde_y".format(target_tensors, tilde_y))

        if numpy.any([t.is_conj for t in target_tensors]):
            raise RuntimeError("No target tensor can be conjugate.")

        self._comm = comm
        self._rank = 0 if self._comm is None else self._comm.Get_rank()
        if not isinstance(distributed_subscript, int) and not distributed_subscript is None:
            raise RuntimeError("Wrong type of distributed_subscript!")
        self._distributed = not distributed_subscript is None

        def dot_product(a, b):
            ab = conj_a_b(a[1], b[1])
            ab.find_contraction_path()
            return (numpy.conj(a[0])*b[0], ab)

        # Tensor networks representing squared errors
        all_terms = []
        self._diagonal_terms = []
        self._half_offdiag_terms = []

        ket_tensors = [(1,tilde_y) for tilde_y in tilde_Y] + [(-1,y) for y in Y]
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
            self._A_generators[t.name] = LinearOperatorGenerator(t, all_terms, comm, self._distributed, reg_L2, self._rank==0, self._mem_limit)
            self._y_generators[t.name] = VectorGenerator(t, all_terms, comm, self._distributed)

        self._target_tensors = target_tensors
        self._target_tensor_names = set([t.name for t in self._target_tensors])
        self._reg_L2 = reg_L2

        self._num_threads = num_threads

    def squared_error(self, tensors_value):
        """
        Compute squared error

        :param tensor_value: dict of (str, ndarray)
           Values of tensors (input)
        :return:
           Squared error
        """

        local_se = numpy.real(_eval_terms(self._diagonal_terms, tensors_value)) + 2 * numpy.real(_eval_terms(self._half_offdiag_terms, tensors_value))
        if self._comm is None:
            return local_se
        else:
            return self._comm.allreduce(local_se)

    def cost(self, tensors_value):
        """
        Evaluate cost function (squared error + L2 regularization)

        :param tensor_value: dict of (str, ndarray)
           Values of tensors (input)
        :return:
           Squared error
        """

        se = self.squared_error(tensors_value)
        norm2 = numpy.sum([numpy.linalg.norm(tensors_value[name])**2 for name in self._target_tensor_names])
        return se + self._reg_L2 * norm2

    def _one_sweep(self, params, constants, verbose=False):
        # One sweep of ALS
        # Shallow copy
        params_new = deepcopy(params)
        tensors_value = ChainMap(params_new, constants)
        for target_tensor in self._target_tensors:
            # Disable multithreading in blas
            with threadpool_limits(limits=1, user_api='blas'):
                name = target_tensor.name
                t1 = time.time()
                opA = self._A_generators[name].construct(tensors_value)
                t2 = time.time()
                vec_y = self._y_generators[name].construct(tensors_value)
                t3 = time.time()

            # Enable multithreading in blas
            with threadpool_limits(limits=self._num_threads, user_api='blas'):
                # Solve the linear system
                #  Use numpy.linalg.solve() for a dense matrix
                if self._rank == 0:
                    # Solve Ax + y = 0 for x on master node
                    if isinstance(opA, numpy.ndarray) and opA.ndim == 1:
                        r = -vec_y/opA
                    elif isinstance(opA, numpy.ndarray) and opA.ndim == 2:
                        r = numpy.linalg.solve(opA, -vec_y)
                    else:
                        x0 = tensors_value[name].ravel()
                        if numpy.linalg.norm(vec_y - opA.matvec(x0)) > numpy.linalg.norm(vec_y):
                            x0 = None
                        r = lgmres(opA, -vec_y, tol=1e-10, atol=0, x0=x0)[0]
                    tensors_value[name][:] = r.reshape(tensors_value[name].shape)
                t4 = time.time()
                if self._comm is not None:
                    self._comm.Barrier()
                    tensors_value[name][:] = self._comm.bcast(tensors_value[name], root=0)
                if self._rank == 0:
                    print(' timings: contruction of opA ', t2-t1, ' contruction of vecy= ', t3-t2, ' solve= ', t4-t3)

        return params_new


    def fit(self, niter, tensors_value, rtol=1e-8, nesterov=True, verbose=False):
        """
        Perform ALS fitting

        :param niter: int
            Number of iterations
        :param tensor_value: dict of (str, ndarray)
            Values of tensors. Those of target tensors will be updated.
        :param rtol: float
            Relative torelance for loss
        :param nesterov: bool
            Use Nesterov's acceleration
        """
        sys.stdout.flush()

        rank = 0 if self._comm is None else self._comm.Get_rank()

        target_tensor_names = set([t.name for t in self._target_tensors])

        # Deep copy
        x0 = {k: v.copy() for k, v in tensors_value.items() if k in target_tensor_names}

        # Shallow copy
        constants = {k: v for k, v in tensors_value.items() if not k in target_tensor_names}

        x_hist = []
        loss_hist = []

        # Record history
        def append_x(x):
            assert len(x_hist) == len(loss_hist)
            if len(x_hist) > 2:
                del x_hist[:-2]
                del loss_hist[:-2]
            x_hist.append(x)
            loss_hist.append(self.cost(ChainMap(x, constants)))

        append_x(x0)
        append_x(self._one_sweep(x0, constants))

        beta = 1.0
        t_start = time.time()
        for iter in range(niter):
            if rank==0 and verbose:
                print('iter= {} loss= {} walltime= {}'.format(iter, loss_hist[-1], time.time()-t_start))
                sys.stdout.flush()

            if nesterov and iter >= 10:
                if iter >= 1 and loss_hist[-1] > loss_hist[-2]:
                    x_hist[-1] = deepcopy(x_hist[-2])
                    loss_hist[-1] = loss_hist[-2]
                    beta = 0.0
                else:
                    beta = 1.0
                if rank == 0:
                    print(" beta = ", beta)
            else:
                beta = 0.0

            if beta == 0.0:
                append_x(self._one_sweep(x_hist[-1], constants))
            else:
                x_tmp = {}
                for name in target_tensor_names:
                    x_tmp[name] = x_hist[-1][name] + beta * (x_hist[-1][name] - x_hist[-2][name])
                append_x(self._one_sweep(x_tmp, constants))


            if numpy.abs(loss_hist[-1]-loss_hist[-2]) < rtol * numpy.abs(loss_hist[-1]):
                break

        tensors_value.update(x_hist[-1])
        #print(loss_hist)

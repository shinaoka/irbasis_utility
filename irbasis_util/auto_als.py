from __future__ import print_function

import numpy
from collections import ChainMap
from copy import deepcopy
from .tensor_network import Tensor, TensorNetwork, conj_a_b, differentiate, from_int_to_char_subscripts
from scipy.sparse.linalg import LinearOperator, lgmres, aslinearoperator
import sys
import time
import ctypes

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
        self._distributed = distributed

        # Tensor network for A
        self._A_tn = differentiate(term, [target_tensor.conjugate(), target_tensor])
        self._A_tn.find_contraction_path(verbose, mem_limit)
        tc_subs = term.tensor_subscripts(target_tensor.conjugate())
        t_subs = term.tensor_subscripts(target_tensor)

        A_subs = self._A_tn.external_subscripts
        op_subs, left_subs, right_subs = from_int_to_char_subscripts([A_subs, tc_subs, t_subs])

        self._op_is_matrix = len(left_subs) + len(right_subs) == len(op_subs)

        if self._op_is_matrix:
            #Transpose axes of matrix
            self._trans_axes = _trans_axes(op_subs, left_subs + right_subs)
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

        #t1 = time.time()
        op_array = self._A_tn.evaluate(tensors_value)
        #t2 = time.time()
        if self._distributed:
            recv = numpy.empty_like(op_array)
            self._comm.Allreduce(op_array, recv, MPI.SUM)
            op_array = recv
        #t3 = time.time()
        #print('const: ', t2-t1, t3-t2, op_array.shape)
        N = self._size

        if self._op_is_matrix:
            op_array = op_array.transpose(self._trans_axes)
            A = op_array.reshape((N, N))
            return aslinearoperator(A)
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

def _identity_operator(N):
    matvec = lambda v: v
    rmatvec = lambda v: v
    return LinearOperator((N, N), matvec=matvec, rmatvec=rmatvec)


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
        # Note sign is flipped
        ops = [(coeff, genA.construct(tensors_value)) for coeff, genA in self._generators]
        if self._reg_L2 != 0:
            N = self._generators[0][1].size
            ops.append((self._reg_L2, _identity_operator(N)))
        return _sum_ops_flipped_sign(ops)

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
    def __init__(self, Y, tilde_Y, target_tensors, reg_L2=0.0, comm=None, distributed_subscript=None, mem_limit=1E+19, num_threads=None):
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
            Number of threads set for MKL backend of numpy when solving linear systems.
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

        # TODO: Add support for OpenBlas
        if num_threads is not None:
            self._mkl_rt = ctypes.CDLL('libmkl_rt.so')
            self._dynamic_threading = True
            self._num_threads = num_threads
        else:
            self._mkl_rt = None
            self._dynamic_threading = False
            self._num_threads = None

    def _set_num_threads(self, num_threads):
        assert self._dynamic_threading
        self._mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(num_threads)))

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
            # Disable multithreading in numpy
            if self._dynamic_threading:
                self._set_num_threads(1)

            name = target_tensor.name
            t1 = time.time()
            opA = self._A_generators[name].construct(tensors_value)
            t2 = time.time()
            vec_y = self._y_generators[name].construct(tensors_value)
            t3 = time.time()

            # Enable multithreading in numpy
            if self._dynamic_threading:
                self._set_num_threads(self._num_threads)
                if verbose:
                    print('Setting num_threads = {} for numpy.'.format(self._num_threads))

            # Solve the linear system
            # FIXME: Is A a hermitian?
            x0 = tensors_value[name].ravel()
            if numpy.linalg.norm(vec_y - opA.matvec(x0)) > numpy.linalg.norm(vec_y):
                x0 = None
            if self._rank == 0:
                # Solve only on master node
                r = lgmres(opA, vec_y, tol=1e-10, atol=0, x0=x0)
                tensors_value[name][:] = r[0].reshape(tensors_value[name].shape)
            t4 = time.time()
            if self._comm is not None:
                self._comm.Barrier()
                tensors_value[name][:] = self._comm.bcast(tensors_value[name], root=0)

            # Disable multithreading in numpy
            if self._dynamic_threading:
                self._set_num_threads(1)

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

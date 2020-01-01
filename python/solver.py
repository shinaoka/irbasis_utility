from __future__ import print_function

from numba import njit

from .four_point import from_PH_convention, to_PH_convention, FourPoint, convert_projector
from .tensor_network import Tensor, TensorNetwork
from .tensor_regression_auto_als import fit_impl, fit
from .gf import LocalGf2CP

import numpy
import copy

def mpi_split(work_size, comm_size):
    base = work_size // comm_size
    leftover = int(work_size % comm_size)

    sizes = numpy.ones(comm_size, dtype=int) * base
    sizes[:leftover] += 1

    offsets = numpy.zeros(comm_size, dtype=int)
    offsets[1:] = numpy.cumsum(sizes)[:-1]

    return sizes, offsets



class FourPointBasisTransform:
    def __init__(self, beta, wmax, scut=1e-4, Nl_max=None, comm=None, sp_F=None):
        """

        :param beta:
        :param wmax:
        :param scut: float
            Cutoff value for singular values
        :param Nl_max: int
            Max basis size for Green's function like object
        :param comm:
        :param sp_F: list of (int, int, int, int)
            Sampling points in fermionic representation
            If MPI is used, sampling points must be distributed over nodes.
        """
        self._Lambda = beta * wmax

        Nl_max_G = Nl_max if Nl_max is not None else None
        Nl_max_vertex = Nl_max+1 if Nl_max is not None else None

        # Basis for G2, Xloc (objects decaying at high frequency)
        self._basis_G2 = FourPoint(self._Lambda, beta, scut, Nl_max=Nl_max_G, augmented=True, vertex=False)

        # Basis for vertex
        self._basis_vertex = FourPoint(self._Lambda, beta, scut, Nl_max=Nl_max_vertex, augmented=True, vertex=True)

        assert self._basis_vertex.Nl == self._basis_G2.Nl + 1

        self._comm = comm
        if self._comm is None:
            MPI_size = 1
            rank = 0
        else:
            MPI_size = self._comm.size
            rank = self._comm.rank


        # Use sampling points for G2 (rather than vertex)
        # in fermionic frequency convention
        if sp_F is None:
            sp_all_F = numpy.array(self._basis_G2.sampling_points_matsubara(self._basis_G2.Nl-1))
            n_sp_all = sp_all_F.shape[0]
            sizes, offsets = mpi_split(n_sp_all, MPI_size)
            start, end = offsets[rank], offsets[rank] + sizes[rank]
            self._sp_local_F = sp_all_F[start:end, :]
            del sp_all_F
        else:
            assert isinstance(sp_F[0], tuple)
            assert len(sp_F[0]) == 4
            self._sp_local_F = copy.deepcopy(sp_F)
        self._n_sp_local = len(self._sp_local_F)

        # Cache data
        self._cache_attr = ['_prj_G2', '_prj_vertex', '_prj_multiply_PH']
        for name in self._cache_attr:
            setattr(self, name, None)

        self._rank = rank

        self._generate_projectors_for_fit()

    @property
    def n_sp_local(self):
        """
        Number of sampling on this MPI node
        """
        return self._n_sp_local

    @property
    def Lambda(self):
        return self._Lambda

    @property
    def basis_G2(self):
        return self._basis_G2

    @property
    def basis_vertex(self):
        return self._basis_vertex

    def _generate_projectors_for_fit(self):
        """
        Generate projectors for fitting.
        The results will be cached.
        """

        if self._prj_vertex is None:
            self._prj_vertex = self._basis_vertex.projector_to_matsubara_vec(self._sp_local_F)
            for imat in range(3):
                self._prj_vertex[imat] = self._prj_vertex[imat].reshape((self._n_sp_local, 16, self._basis_vertex.Nl))

            self._prj_G2 = [self._prj_vertex[imat][:, :, 1:] for imat in range(3)]


    def fit_LocalGf2CP(self, g, data_sp, niter, random_init=True, rtol=1e-8, alpha=1e-8, verbose=1, seed=100):
        assert data_sp.shape[0] == self._n_sp_local

        # Find non-zero orbital components
        tmp = numpy.sqrt(numpy.sum(numpy.abs(data_sp)**2, axis=0)).ravel()
        orb_idx = tmp / numpy.amax(tmp) > 1e-8

        x0 = None if random_init else g.tensors

        prj = {True: self._prj_vertex, False: self._prj_G2}[g.vertex]
        assert prj is not None
        D = g.D

        tensors = fit(data_sp[:, orb_idx], prj, D, niter, rtol, verbose=verbose,
                 random_init=random_init, comm=self._comm, seed=seed, alpha=alpha, nesterov=True, x0=x0)

        for i in range(4):
            g.tensors[i][:] = tensors[i][:]
        g.tensors[-1][:] = 0.0
        g.tensors[-1][:, orb_idx] = tensors[-1]


    def _get_prj_multiply_PH_L(self, vertex):
        pass

    def _get_prj_multiply_PH_R(self, vertex):
        pass

    def multiply_LocalGf2CP_PH(self, g_left, g_right, n_max_inner, D_new, nite, rtol, alpha):
        assert self._prj_vertex is not None

        # Sampling points for multiplication
        sp_left, sp_right = _construct_sampling_points_multiplication(range(-n_max_inner, n_max_inner), self._sp_local_F)

        num_w_outer = self._n_sp_local
        num_w_inner = 2*n_max_inner

        sp_left = sp_left.reshape((num_w_outer * num_w_inner, 4))
        sp_right = sp_right.reshape((num_w_outer * num_w_inner, 4))

        # Projectors
        basis_dict = {True: self.basis_vertex, False: self.basis_G2}
        print("Generating projectors for left object...")
        prj_left = basis_dict[g_left.vertex].projector_to_matsubara_vec(sp_left, reduced_memory=True)
        print("Generating projectors for right object...")
        prj_right = basis_dict[g_left.vertex].projector_to_matsubara_vec(sp_right, reduced_memory=True)
        vertex_result = g_left.vertex and g_right.vertex
        prj = {True: self._prj_vertex, False: self._prj_G2}[vertex_result]
        Nl_result = {True: self._basis_vertex.Nl, False: self._basis_G2.Nl}[vertex_result]

        assert g_left.No == g_right.No

        verbose = self._rank == 0

        xtensors = _multiply_LocalGf2CP_PH(self._n_sp_local, num_w_inner, g_left, g_right, prj, prj_left, prj_right,
                                     D_new, nite, rtol, alpha, self._comm, seed=1, verbose=verbose)

        return LocalGf2CP(self.Lambda, Nl_result, g_left.No, D_new, xtensors, vertex_result)


    def clear_large_data(self):
        """
        Clear large cache data
        """

        for name in self._cache_attr:
            setattr(self, name, None)


def _multiply_LocalGf2CP_PH(Nw, num_w_inner, g_left, g_right, prj, prj_multiply_PH_L, prj_multiply_PH_R,
                                     D_new, nite, rtol, alpha, comm, seed=1, verbose=False):
    """
    Compute the product of two tensors in PH channel
    """

    Nr = 16

    assert len(prj_multiply_PH_L)==2
    assert len(prj_multiply_PH_R)==2
    prj_multiply_PH_L = (prj_multiply_PH_L[0], prj_multiply_PH_L[1].reshape((3, Nw, num_w_inner, Nr)))
    prj_multiply_PH_R = (prj_multiply_PH_R[0], prj_multiply_PH_R[1].reshape((3, Nw, num_w_inner, Nr)))

    print('Computing product of two objects...', end='')
    y0, y1 = _innersum_LocalGf2CP_PH(Nw, num_w_inner, g_left, g_right, prj_multiply_PH_L, prj_multiply_PH_R)
    print('done!')

    Nw, _, _ = prj[0].shape
    No = g_left.tensors[-1].shape[1]
    D_L = g_left.D
    D_R = g_right.D

    tensors = [Tensor('y0', (D_L * D_R, Nw)), Tensor('y1', (D_L * D_R, No))]
    tensor_values = {'y0': y0, 'y1': y1}
    subs = [(10, 0), (10, 1)]
    Y_tn = TensorNetwork(tensors, subs, external_subscripts={0, 1})

    return fit_impl(Y_tn, tensor_values, prj, D_new, nite, rtol=rtol, verbose=verbose, random_init=True, x0=None, alpha=alpha, comm=comm,
        seed=seed, nesterov=True)

@njit
def _contract(Nw, num_w_inner, xr_L, xr_R, U_xls_L, U_xls_R, coords_trans_L, coords_trans_R, work_L, work_R,
              work_L2, work_R2, out):
    """
    Perform summation over inner frequencies
    Nr = 16. D_L and D_R

    :param Nw: int
        Number of outer frequencies
    :param num_w_inner: int
        Number of inner frequencies
    :param xr_L: (D_L, Nr)
        Tensor for "r" of left object
    :param xr_R: (D_R, Nr)
        Tensor for "r" of right object
    :param U_xls_L: (3, Nr, D_L)
        Product of tensors for IR and projectors (left object)
    :param U_xls_R: (3, Nr, D_R)
        Product of tensors for IR and projectors (right object)
    :param coords_trans_L: (Nw, 2*n_max_inner, 3, Nr)
        Translation from 2pt frequencies to 1pt frequencies
    :param coords_trans_R: (Nw, 2*n_max_inner, 3, Nr)
        Translation from 2pt frequencies to 1pt frequencies
    :param work_L: (Nr, D_L)
        Work array
    :param work_R: (Nr, D_R)
        Work array
    :param work_L2: (D_L,)
        Work array
    :param work_R2: (D_R,)
        Work array
    :param out: (Nw, D_L, D_R)
        Output
    """
    D_L, Nr = xr_L.shape
    D_R, Nr = xr_R.shape
    assert coords_trans_L.shape == (Nw, num_w_inner, 3, Nr)
    assert coords_trans_R.shape == (Nw, num_w_inner, 3, Nr)
    _, N_1pfreq_L, _ =  U_xls_L.shape
    _, N_1pfreq_R, _ =  U_xls_R.shape

    assert work_L.shape == (Nr, D_L)
    assert work_R.shape == (Nr, D_R)
    assert work_L2.shape == (D_L,)
    assert work_R2.shape == (D_R,)
    assert Nr == 16

    assert out.shape == (Nw, D_L, D_R)

    out[:, :, :] = 0.0
    for i_outer in range(Nw):
        for i_inner in range(num_w_inner):
            work_L[:,:] = xr_L.transpose()
            work_R[:,:] = xr_R.transpose()
            for imat in range(3):
                for r in range(Nr):
                    work_L[r, :] *= U_xls_L[imat, coords_trans_L[i_outer, i_inner, imat, r], :]
                    work_R[r, :] *= U_xls_R[imat, coords_trans_R[i_outer, i_inner, imat, r], :]
            work_L2[:] = numpy.sum(work_L, axis=0)
            work_R2[:] = numpy.sum(work_R, axis=0)
            out[i_outer, :, :] += numpy.outer(work_L2, work_R2)

def _innersum_LocalGf2CP_PH(Nw, num_w_inner, g_left, g_right, prj_multiply_PH_L, prj_multiply_PH_R):
    """
    Compute the product of two tensors in PH channel

    """

    # Number of representations
    Nr = 16

    # Bond dimensions
    D_L = g_left.D
    D_R = g_right.D

    # Frequency part
    x0 =_innersum_freq_PH(Nw, num_w_inner, g_left.tensors[0:4], g_right.tensors[0:4], prj_multiply_PH_L, prj_multiply_PH_R)

    # Orbital part
    No = g_left.tensors[-1].shape[1]
    sqrt_No = int(numpy.sqrt(No))
    assert sqrt_No**2 == No
    x1 = numpy.einsum('Dpq, Eqr->DEpr', g_left.tensors[-1].reshape((D_L,sqrt_No,sqrt_No)),
                      g_right.tensors[-1].reshape((D_R, sqrt_No, sqrt_No)), optimize=True).reshape((D_L*D_R, No))

    return x0, x1


def _innersum_freq_PH(nw_outer, num_w_inner, xfreqs_L, xfreqs_R, prj_multiply_PH_L, prj_multiply_PH_R):
    """
    Perform frequency summation for the internal line in PH channel
    """

    # Number of representations
    Nr = 16

    D_L, Nl_L = xfreqs_L[1].shape
    D_R, Nl_R = xfreqs_R[1].shape

    # Alias
    Unl_L = prj_multiply_PH_L[0]
    Unl_R = prj_multiply_PH_R[0]

    N_1ptfreq_L = Unl_L.shape[0]
    N_1ptfreq_R = Unl_R.shape[0]

    # Unl_L/R (3, N_1ptfreq, Nl)
    UnD_L = numpy.empty((3, N_1ptfreq_L, D_L), dtype=numpy.complex)
    UnD_R = numpy.empty((3, N_1ptfreq_R, D_R), dtype=numpy.complex)
    for imat in range(3):
        UnD_L[imat, :, :] = numpy.einsum('nl,Dl->nD', Unl_L, xfreqs_L[imat + 1], optimize=True)
        UnD_R[imat, :, :] = numpy.einsum('nl,Dl->nD', Unl_R, xfreqs_R[imat + 1], optimize=True)

    x0 = numpy.empty((nw_outer, D_L, D_R), dtype=numpy.complex)
    work_L = numpy.empty((Nr, D_L,), dtype=numpy.complex)
    work_R = numpy.empty((Nr, D_R,), dtype=numpy.complex)
    work_L2 = numpy.empty((D_L,), dtype=numpy.complex)
    work_R2 = numpy.empty((D_R,), dtype=numpy.complex)

    # Index conversion from 2pt frequency to 1pt frequency
    # coords_trans_L, coords_trans_R: (3, nw_outer*n_max_inner, Nr) => (nw_outer, n_max_inner, 3, Nr)
    coords_trans_L = numpy.array(prj_multiply_PH_L[1]).reshape((3, nw_outer, num_w_inner, Nr)).transpose((1, 2, 0, 3))
    coords_trans_R = numpy.array(prj_multiply_PH_R[1]).reshape((3, nw_outer, num_w_inner, Nr)).transpose((1, 2, 0, 3))

    _contract(nw_outer, num_w_inner, xfreqs_L[0], xfreqs_R[0],
              UnD_L, UnD_R,
              coords_trans_L, coords_trans_R,
              work_L, work_R,
              work_L2, work_R2, x0)

    return x0.transpose((1, 2, 0)).reshape((D_L * D_R, nw_outer))


def _innersum_freq_PH_ref(nw_outer, num_w_inner, xfreqs_L, xfreqs_R, prj_multiply_PH_L, prj_multiply_PH_R):
    """
    Does the same job as _innersum_freq_PH, but requires much more memory.
    """
    # Number of representations
    Nr = 16

    # Bond dimensions
    D_L, Nl_L = xfreqs_L[1].shape
    D_R, Nl_R = xfreqs_R[1].shape

    # Prepare projectors
    prj_multiply_PH_L = convert_projector(*prj_multiply_PH_L)
    prj_multiply_PH_R = convert_projector(*prj_multiply_PH_R)
    for i in range(3):
        prj_multiply_PH_L[i] = prj_multiply_PH_L[i].reshape((nw_outer, num_w_inner, Nr, Nl_L))
        prj_multiply_PH_R[i] = prj_multiply_PH_R[i].reshape((nw_outer, num_w_inner, Nr, Nl_R))

    # Define tensor network
    tensors = []
    subs = []
    tensor_values = {}

    subs_W_inner = 30
    subs_W = 2
    subs_r_L = 10
    subs_r_R = 20

    for i in [1, 2, 3]:
        tensors.append(Tensor('UL{}'.format(i), (nw_outer, num_w_inner, Nr, Nl_L)))
        subs.append((subs_W, subs_W_inner, subs_r_L, i + 10))
        tensor_values['UL{}'.format(i)] = prj_multiply_PH_L[i-1]

        tensors.append(Tensor('UR{}'.format(i), (nw_outer, num_w_inner, Nr, Nl_R)))
        subs.append((subs_W, subs_W_inner, subs_r_R, i + 20))
        tensor_values['UR{}'.format(i)] = prj_multiply_PH_R[i-1]

        tensors.append(Tensor('xL{}'.format(i), (D_L, Nl_L)))
        subs.append((0, i + 10))
        tensor_values['xL{}'.format(i)] = xfreqs_L[i]

        tensors.append(Tensor('xR{}'.format(i), (D_R, Nl_R)))
        subs.append((1, i + 20))
        tensor_values['xR{}'.format(i)] = xfreqs_R[i]

    tensors.append(Tensor('xL0', (D_L, Nr)))
    subs.append((0, subs_r_L))
    tensor_values['xL0'] = xfreqs_L[0]

    tensors.append(Tensor('xR0', (D_R, Nr)))
    subs.append((1, subs_r_R))
    tensor_values['xR0'] = xfreqs_R[0]

    tn = TensorNetwork(tensors, subs, (2, 0, 1))
    tn.find_contraction_path(verbose=True)

    return tn.evaluate(tensor_values).reshape((nw_outer, D_L * D_R)).transpose()

def _construct_sampling_points_multiplication(inner_w_range, sp_outer_F):
    """

    :param inner_w_range: range like
       Define the range of the internal frequency index
    :param sp_outer_F: array like
       Sampling points in fermionic convention
    :return:
       Sampling points for multiplication (left & right objects)
    """
    # Sampling points for multiplication in fermionic convention

    num_outer_w = sp_outer_F.shape[0]

    num_inner_w = len(inner_w_range)
    sp_PH_left = numpy.empty((num_outer_w, num_inner_w, 3), dtype=int)
    sp_PH_right = numpy.empty((num_outer_w, num_inner_w, 3), dtype=int)
    sp_PH = to_PH_convention(sp_outer_F)
    for i, (nf1, nf2, nb) in enumerate(sp_PH):
        for idx_inner, n in enumerate(inner_w_range):
            sp_PH_left[i, idx_inner, :] = (nf1, n, nb)
            sp_PH_right[i, idx_inner, :] = (n, nf2, nb)
    sp_left = from_PH_convention(sp_PH_left.reshape((-1, 3))).reshape((num_outer_w, num_inner_w, 4))
    sp_right = from_PH_convention(sp_PH_right.reshape((-1, 3))).reshape((num_outer_w, num_inner_w, 4))

    return sp_left, sp_right


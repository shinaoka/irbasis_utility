from __future__ import print_function

from .four_point import from_PH_convention, to_PH_convention, FourPoint
#from .tensor_regression_auto_als import fit, predict
#from .auto_als import AutoALS
from .tensor_network import Tensor, TensorNetwork
from .tensor_regression_auto_als import fit_impl, fit
from .gf import LocalGf2CP

import numpy
import gc
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

    @property
    def Lambda(self):
        return self._Lambda

    @property
    def basis_G2(self):
        return self._basis_G2

    @property
    def basis_vertex(self):
        return self._basis_vertex

    def generate_projectors_for_fit(self):
        """
        Generate projectors for fitting.
        The results will be cached.
        """

        if self._prj_vertex is None:
            self._prj_vertex = self._basis_vertex.projector_to_matsubara_vec(self._sp_local_F, decomposed_form=True)
            for imat in range(3):
                self._prj_vertex[imat] = self._prj_vertex[imat].reshape((self._n_sp_local, 16, self._basis_vertex.Nl))

            self._prj_G2 = [self._prj_vertex[imat][:, :, 1:] for imat in range(3)]


    def fit_LocalGf2CP(self, g, data_sp, niter, random_init=True, rtol=1e-8, alpha=1e-8, verbose=1, seed=100):
        assert data_sp.shape[0] == self._n_sp_local

        self.generate_projectors_for_fit()

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

    def multiply_LocalGf2CP_PH(self, g_left, g_right, niw_inner, D_new, nite, rtol, alpha):
        # Sampling points for multiplication
        nf_inner = numpy.arange(-niw_inner, niw_inner)
        sp_PH_left = numpy.empty((self._n_sp_local, 2*niw_inner, 3), dtype=int)
        sp_PH_right = numpy.empty((self._n_sp_local, 2*niw_inner, 3), dtype=int)
        sp_PH = to_PH_convention(self._sp_local_F)
        for i, (nf1, nf2, nb) in enumerate(sp_PH):
            for idx_inner, n in enumerate(nf_inner):
                sp_PH_left[i, idx_inner, :] = (nf1, n, nb)
                sp_PH_right[i, idx_inner, :] = (n, nf2, nb)
        sp_left = from_PH_convention(sp_PH_left.reshape((-1, 3)))
        sp_right = from_PH_convention(sp_PH_right.reshape((-1, 3)))

        # Projectors
        basis_dict = {True: self.basis_vertex, False: self.basis_G2}
        print("Generating projectors for left object...")
        prj_left = basis_dict[g_left.vertex].projector_to_matsubara_vec(sp_left).reshape((self._n_sp_local, 2*niw_inner, -1))
        print("Generating projectors for right object...")
        prj_right = basis_dict[g_left.vertex].projector_to_matsubara_vec(sp_right).reshape((self._n_sp_local, 2*niw_inner, -1))
        vertex_result = g_left.vertex and g_right.vertex
        prj = {True: self._prj_vertex, False: self._prj_G2}[vertex_result]
        Nl_result = {True: self._basis_vertex.Nl, False: self._basis_G2.Nl}[vertex_result]


        assert g_left.No == g_right.No

        verbose = self._rank == 0
        xtensors = _multiply_LocalGf2CP_PH_compress(g_left, g_right, prj, prj_left, prj_right,
                                     D_new, nite, rtol, alpha, self._comm, seed=1, verbose=verbose)

        return LocalGf2CP(self.Lambda, Nl_result, g_left.No, D_new, xtensors, vertex_result)


    def clear_large_data(self):
        """
        Clear large cache data
        """

        for name in self._cache_attr:
            setattr(self, name, None)

        gc.collect()


def _multiply_LocalGf2CP_PH_compress(g_left, g_right, prj, prj_multiply_PH_L, prj_multiply_PH_R,
                                     D_new, nite, rtol, alpha, comm, seed=1, verbose=False):
    """
    Compute the product of two tensors in PH channel
    """

    y0, y1 = _multiply_LocalGf2CP_PH(g_left, g_right, prj_multiply_PH_L, prj_multiply_PH_R, verbose=verbose)

    Nw, _, _, _ = prj_multiply_PH_L[0].shape
    No = g_left.tensors[-1].shape[1]
    D_L = g_left.D
    D_R = g_right.D

    tensors = [Tensor('y0', (D_L * D_R, Nw)), Tensor('y1', (D_L * D_R, No))]
    tensor_values = {'y0': y0, 'y1': y1}
    subs = [(10, 0), (10, 1)]
    Y_tn = TensorNetwork(tensors, subs, external_subscripts={0, 1})

    return fit_impl(Y_tn, tensor_values, prj, D_new, nite, rtol=rtol, verbose=0, random_init=True, x0=None, alpha=alpha, comm=comm,
        seed=seed, nesterov=True)



def _multiply_LocalGf2CP_PH(g_left, g_right, prj_multiply_PH_L, prj_multiply_PH_R,
                            verbose=False):
    """
    Compute the product of two tensors in PH channel

    :param g_left:
    :param g_right:
    :param prj_multiply_PH:
    :param verbose: bool
    :return: tuple of two ndarrays
        Two tensors for frequency and orbitals
    """

    # Number of frequencies for outer/internal lines
    Nw, Nw_inner, _, Nl_L = prj_multiply_PH_L[0].shape
    _, _, _, Nl_R = prj_multiply_PH_R[0].shape

    No = g_left.tensors[-1].shape[1]
    sqrt_No = int(numpy.sqrt(No))
    assert sqrt_No**2 == No

    assert Nl_L == g_left.Nl
    assert Nl_R == g_right.Nl

    # Bond dimensions
    D_L = g_left.D
    D_R = g_right.D

    # Number of representations
    Nr = 16

    # Define tensor network
    tensors = []
    subs = []
    tensor_values = {}

    subs_W_inner = 30
    subs_W = 2
    subs_r_L = 10
    subs_r_R = 20

    for i in [1, 2, 3]:
        tensors.append(Tensor('UL{}'.format(i), (Nw, Nw_inner, Nr, Nl_L)))
        subs.append((subs_W, subs_W_inner, subs_r_R, i + 10))
        tensor_values['UL{}'.format(i)] = prj_multiply_PH_L[i-1]

        tensors.append(Tensor('UR{}'.format(i), (Nw, Nw_inner, Nr, Nl_R)))
        subs.append((subs_W, subs_W_inner, subs_r_L, i + 20))
        tensor_values['UR{}'.format(i)] = prj_multiply_PH_R[i-1]

        tensors.append(Tensor('xL{}'.format(i), (D_L, Nl_L)))
        subs.append((0, i + 10))
        tensor_values['xL{}'.format(i)] = g_left.tensors[i]

        tensors.append(Tensor('xR{}'.format(i), (D_R, Nl_R)))
        subs.append((1, i + 20))
        tensor_values['xR{}'.format(i)] = g_right.tensors[i]

    tensors.append(Tensor('xL0', (D_L, Nr)))
    subs.append((0, 10))
    tensor_values['xL0'] = g_left.tensors[0]

    tensors.append(Tensor('xR0', (D_R, Nr)))
    subs.append((1, 20))
    tensor_values['xR0'] = g_right.tensors[0]

    tn = TensorNetwork(tensors, subs, (2, 0, 1))
    tn.find_contraction_path(verbose=verbose)

    # Evaluate tensor network
    x0 = tn.evaluate(tensor_values).reshape((Nw, D_L * D_R))
    x1 = numpy.einsum('Dpq, Eqr->DEpr', g_left.tensors[3].reshape((D_L,sqrt_No,sqrt_No)),
                      g_right.tensors[3].reshape((D_R, sqrt_No, sqrt_No)), optimize=True).reshape((D_L*D_R, No))

    return x0, x1






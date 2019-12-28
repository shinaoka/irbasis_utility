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
            self._prj_vertex = self._basis_vertex.projector_to_matsubara_vec(self._sp_local_F)
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
        assert self._prj_vertex is not None

        # Sampling points for multiplication
        Nw_inner = 2*niw_inner
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
        #prj_left = basis_dict[g_left.vertex].projector_to_matsubara_vec(sp_left).reshape((self._n_sp_local, 2*niw_inner, -1))
        prj_left = basis_dict[g_left.vertex].projector_to_matsubara_vec(sp_left, reduced_memory=True)
        print("Generating projectors for right object...")
        #prj_right = basis_dict[g_left.vertex].projector_to_matsubara_vec(sp_right).reshape((self._n_sp_local, 2*niw_inner, -1))
        prj_right = basis_dict[g_left.vertex].projector_to_matsubara_vec(sp_right, reduced_memory=True)
        vertex_result = g_left.vertex and g_right.vertex
        prj = {True: self._prj_vertex, False: self._prj_G2}[vertex_result]
        Nl_result = {True: self._basis_vertex.Nl, False: self._basis_G2.Nl}[vertex_result]

        assert g_left.No == g_right.No

        verbose = self._rank == 0
        xtensors = _multiply_LocalGf2CP_PH_compress(self._n_sp_local, Nw_inner, g_left, g_right, prj, prj_left, prj_right,
                                     D_new, nite, rtol, alpha, self._comm, seed=1, verbose=verbose)

        return LocalGf2CP(self.Lambda, Nl_result, g_left.No, D_new, xtensors, vertex_result)


    def clear_large_data(self):
        """
        Clear large cache data
        """

        for name in self._cache_attr:
            setattr(self, name, None)

        gc.collect()


def _multiply_LocalGf2CP_PH_compress(Nw, Nw_inner, g_left, g_right, prj, prj_multiply_PH_L, prj_multiply_PH_R,
                                     D_new, nite, rtol, alpha, comm, seed=1, verbose=False):
    """
    Compute the product of two tensors in PH channel
    """

    Nr = 16
    for i in range(1,4):
        prj_multiply_PH_L[i] = prj_multiply_PH_L[i].reshape((Nw, Nw_inner, Nr))
        prj_multiply_PH_R[i] = prj_multiply_PH_R[i].reshape((Nw, Nw_inner, Nr))

    y0, y1 = _multiply_LocalGf2CP_PH(Nw, Nw_inner, g_left, g_right, prj_multiply_PH_L, prj_multiply_PH_R)

    Nw, _, _ = prj[0].shape
    No = g_left.tensors[-1].shape[1]
    D_L = g_left.D
    D_R = g_right.D

    tensors = [Tensor('y0', (D_L * D_R, Nw)), Tensor('y1', (D_L * D_R, No))]
    tensor_values = {'y0': y0, 'y1': y1}
    subs = [(10, 0), (10, 1)]
    Y_tn = TensorNetwork(tensors, subs, external_subscripts={0, 1})

    return fit_impl(Y_tn, tensor_values, prj, D_new, nite, rtol=rtol, verbose=0, random_init=True, x0=None, alpha=alpha, comm=comm,
        seed=seed, nesterov=True)


def _contract_one_side(xr, U_xls, coords_trans, work, out):
    """
    Contract tensors for one side.
    """
    D, Nr = xr.shape
    assert coords_trans.shape == (3, Nr)
    _, N_1pfreq, _ =  U_xls.shape

    assert work.shape == (Nr, D)
    assert out.shape == (D,)

    work[:,:] = xr.transpose()
    for imat in range(3):
        for r in range(Nr):
            work[r, :] *= U_xls[imat, coords_trans[imat, r], :]
    numpy.sum(work, axis=0, out=out)



def _multiply_LocalGf2CP_PH(Nw, Nw_inner, g_left, g_right, prj_multiply_PH_L, prj_multiply_PH_R):
    """
    Compute the product of two tensors in PH channel

    :param g_left:
    :param g_right:
    :param prj_multiply_PH:
    :param verbose: bool
    :return: tuple of two ndarrays
        Two tensors for frequency and orbitals
    """

    # Number of representations
    Nr = 16

    Nl_L, Nl_R = g_left.Nl, g_right.Nl

    # Alias
    Unl_L = prj_multiply_PH_L[0]
    Unl_R = prj_multiply_PH_R[0]

    N_1ptfreq_L = Unl_L.shape[0]
    N_1ptfreq_R = Unl_R.shape[0]

    No = g_left.tensors[-1].shape[1]
    sqrt_No = int(numpy.sqrt(No))
    assert sqrt_No**2 == No

    assert Nl_L == g_left.Nl
    assert Nl_R == g_right.Nl

    # Bond dimensions
    D_L = g_left.D
    D_R = g_right.D

    # Unl_L/R (3, N_1ptfreq, Nl)
    UnD_L = numpy.empty((3, N_1ptfreq_L, D_L), dtype=numpy.complex)
    UnD_R = numpy.empty((3, N_1ptfreq_R, D_R), dtype=numpy.complex)
    for imat in range(3):
        UnD_L[imat, :, :] = numpy.einsum('nl,Dl->nD', Unl_L, g_left.tensors[imat+1][:, :])
        UnD_R[imat, :, :] = numpy.einsum('nl,Dl->nD', Unl_R, g_right.tensors[imat+1][:, :])

    x0 = numpy.empty((Nw, D_L, D_R), dtype=numpy.complex)
    wrk_L = numpy.empty((D_L,), dtype=numpy.complex)
    wrk_R = numpy.empty((D_R,), dtype=numpy.complex)
    wrk_L_2 = numpy.empty((Nr, D_L,), dtype=numpy.complex)
    wrk_R_2 = numpy.empty((Nr, D_R,), dtype=numpy.complex)

    # Index conversion from 2pt frequency to 1pt frequency
    # coords_trans_L, coords_trans_R: (3, Nw*Nw_inner, Nr) => (Nw, Nw_inner, 3, Nr)
    coords_trans_L = numpy.array(prj_multiply_PH_L[1:]).reshape((3, Nw, Nw_inner, Nr)).transpose((1, 2, 0, 3))
    coords_trans_R = numpy.array(prj_multiply_PH_R[1:]).reshape((3, Nw, Nw_inner, Nr)).transpose((1, 2, 0, 3))

    for i_outer in range(Nw):
        x0[i_outer, :, :] = 0.0
        for i_inner in range(Nw_inner):
            _contract_one_side(g_left.tensors[0], UnD_L, coords_trans_L[i_outer,i_inner,:,:], wrk_L_2, out=wrk_L)
            _contract_one_side(g_right.tensors[0], UnD_R, coords_trans_R[i_outer,i_inner,:,:], wrk_R_2, out=wrk_R)
            x0[i_outer, :, :] += numpy.outer(wrk_L, wrk_R)

    x1 = numpy.einsum('Dpq, Eqr->DEpr', g_left.tensors[-1].reshape((D_L,sqrt_No,sqrt_No)),
                      g_right.tensors[-1].reshape((D_R, sqrt_No, sqrt_No)), optimize=True).reshape((D_L*D_R, No))

    x0 = x0.transpose((1,2,0)).reshape((D_L*D_R, Nw))
    return x0, x1






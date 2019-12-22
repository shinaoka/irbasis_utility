from __future__ import print_function

#from .tensor_network import Tensor, TensorNetwork, conj_a_b, differenciate, from_int_to_char_subscripts
from .four_point import from_PH_convention, FourPoint
from .gf import LocalGf2CP
#from .auto_als import AutoALS
from .tensor_regression_auto_als import fit, predict

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
    def __init__(self, beta, wmax, scut=1e-4, comm=None, sp_F=None):
        """

        :param beta:
        :param wmax:
        :param scut:
        :param comm:
        :param sp_F: list of (int, int, int, int)
            Sampling points in fermionic representation
            If MPI is used, sampling points must be distributed over nodes.
        """
        Lambda = beta * wmax

        # Basis for G2, Xloc (objects decaying at high frequency)
        self._basis_G2 = FourPoint(Lambda, beta, scut, augmented=True, vertex=False)

        # Basis for vertex
        self._basis_vertex = FourPoint(Lambda, beta, scut, augmented=True, vertex=True)

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
        else:
            assert isinstance(sp_F[0], tuple)
            assert len(sp_F[0]) == 4
            self._sp_local_F = copy.deepcopy(sp_F)
        self._n_sp_local = len(self._sp_local_F)

        self._prj_G2 = None
        self._prj_vertex = None

    @property
    def basis_G2(self):
        return self._basis_G2


    @property
    def basis_vertex(self):
        return self._basis_vertex

    def generate_projectors_for_fit(self):
        """
        Generate projectors for fitting. The results will be cached.
        """

        if self._prj_G2 is None:
            self._prj_G2 = self._basis_G2.projector_to_matsubara_vec(self._sp_local_F, decomposed_form=True)
            for imat in range(3):
                self._prj_G2[imat] = self._prj_G2[imat].reshape((self._n_sp_local, 16, self._basis_G2.Nl))

        if self._prj_vertex is None:
            self._prj_vertex = self._basis_vertex.projector_to_matsubara_vec(self._sp_local_F, decomposed_form=True)
            for imat in range(3):
                self._prj_vertex[imat] = self._prj_vertex[imat].reshape((self._n_sp_local, 16, self._basis_vertex.Nl))

        #print("debugA", self._prj_G2)
        #print("debugB", self._prj_vertex)


    def fit_LocalGf2CP(self, G2, data_sp, niter, random_init=True, rtol=1e-8, alpha=1e-8, verbose=1, seed=100):
        assert data_sp.shape[0] == self._n_sp_local

        self.generate_projectors_for_fit()

        num_o = data_sp.shape[1]

        # Find non-zero orbital components
        tmp = numpy.sqrt(numpy.sum(numpy.abs(data_sp)**2, axis=0)).ravel()
        orb_idx = tmp / numpy.amax(tmp) > 1e-8
        #num_o_nonzero = numpy.sum(orb_idx)

        x0 = None if random_init else G2.tensors

        prj = {True: self._prj_vertex, False: self._prj_G2}[G2.vertex]
        assert prj is not None
        D = G2.D

        tensors = fit(data_sp[:, orb_idx], prj, D, niter, rtol, verbose=verbose,
                 random_init=random_init, comm=self._comm, seed=seed, alpha=alpha, nesterov=True, x0=x0)

        for i in range(4):
            G2.tensors[i][:] = tensors[i][:]
        G2.tensors[-1][:] = 0.0
        G2.tensors[-1][:, orb_idx] = tensors[-1]


    #def fit_vertex(self, vertex):

        #return xtensors


    def clear_large_data(self):
        """
        Clear large cache data
        """

        self._prj_G2 = None
        self._prj_vertex = None
        gc.collect()

def fit_G2loc_ph():
    pass

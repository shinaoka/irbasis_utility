from __future__ import print_function
import numpy
import scipy
import scipy.linalg
from irbasis import *
from itertools import product, permutations
from .internal import *
from .two_point_basis import *

from numba import njit, jit

# FFF representations (#1-#4)
idx_n1n2n3_FFF = list()
idx_n1n2n3_FFF.append(numpy.array((0,1,2))) # (iw_1, i_w2, i_w3)
idx_n1n2n3_FFF.append(numpy.array((0,1,3))) # (iw_1, i_w2, i_w4)
idx_n1n2n3_FFF.append(numpy.array((0,2,3))) # (iw_1, i_w3, i_w4)
idx_n1n2n3_FFF.append(numpy.array((1,2,3))) # (iw_2, i_w3, i_w4)
idx_n1n2n3_FFF = numpy.array(idx_n1n2n3_FFF)

# FBF representations (#5-#16)
idx_n1n2n4_FBF = numpy.array([numpy.array((p[0], p[1], p[3])) for p in permutations([0, 1, 2, 3]) if p[0] < p[3]])

@njit
def _construct_coords(n1_n2_n3_n4_vec, o_fb):
    """
    Construct conversion from 2pt frequencies to 1pt frequencies
    """
    nw = n1_n2_n3_n4_vec.shape[0]

    min_o = numpy.amin(o_fb)
    max_o = numpy.amax(o_fb)

    num_o_dense = max_o - min_o + 1
    map_o = numpy.empty(num_o_dense, dtype=numpy.int64)
    map_o[:] = -100000000
    for i in range(len(o_fb)):
        map_o[o_fb[i]-min_o] = i

    coords = numpy.empty((3, nw, 16, 3), dtype=numpy.int64)
    for i in range(nw):
        nvec = n1_n2_n3_n4_vec[i, :]

        # FFF representations
        for r in range(4):
            three_ferminonic_freqs = nvec[idx_n1n2n3_FFF[r]]
            for imat in range(3):
                coords[imat, i, r, 0] = i
                coords[imat, i, r, 1] = r
                coords[imat, i, r, 2] = map_o[2*three_ferminonic_freqs[imat]+1 - min_o]

        # FBF representations
        for r in range(12):
            n1_p, n2_p, n4_p = nvec[idx_n1n2n4_FBF[r]]
            coords[:, i, r + 4, 0] = i
            coords[:, i, r + 4, 1] = r + 4

            coords[0, i, r + 4, 2] = map_o[2*n1_p+1 - min_o] # F
            coords[1, i, r + 4, 2] = map_o[2*(n1_p + n2_p + 1) - min_o] # B
            coords[2, i, r + 4, 2] = map_o[2*(-n4_p - 1)+1 - min_o]  # F

    return coords

class FourPoint(object):
    def __init__(self, Lambda, beta, cutoff=1e-8, augmented=True, vertex=False, ortho=False, Nl_max=None):
        self._Lambda = Lambda
        self._beta = beta

        bf = irbasis.load('F', Lambda)
        bb = irbasis.load('B', Lambda)
        if augmented:
            if ortho:
                bb = augmented_ortho_basis_b(bb)
            else:
                bb = augmented_basis_b(bb)
        if vertex:
            bf = vertex_basis(bf)
            bb = vertex_basis(bb)

        self._Bf = Basis(bf, beta, cutoff, Nl_max)
        self._Bb = Basis(bb, beta, cutoff, Nl_max)

        # DG: the below is tantamount to using a larger cutoff
        # for one of the basis
        self._Nl = min(self._Bf.dim, self._Bb.dim)

    @property
    def beta(self):
        return self._beta

    @property
    def Nl(self):
        return self._Nl

    @property
    def basis_beta_f(self):
        return self._Bf

    @property
    def basis_beta_b(self):
        return self._Bb

    def normalized_S(self, decomposed_form = False):
        Nl = self._Nl
        sf = numpy.array([self._Bf.Sl(l) / self._Bf.Sl(0) for l in range(Nl)])
        sb = numpy.array([self._Bb.Sl(l) / self._Bb.Sl(0) for l in range(Nl)])
        svec1 = numpy.zeros((16, Nl))
        svec2 = numpy.zeros((16, Nl))
        svec3 = numpy.zeros((16, Nl))

        svec1[0:4, :], svec2[0:4, :], svec3[0:4, :] = sf[None, :], sf[None, :], sf[None, :]
        svec1[4:, :], svec2[4:, :], svec3[4:, :] = sf[None, :], sb[None, :], sf[None, :]

        if decomposed_form:
            return svec1, svec2, svec3
        else:
            return numpy.einsum('ri,rj,rk->rijk', svec1, svec2, svec3)


    def projector_to_matsubara_vec(self, n1_n2_n3_n4_vec, reduced_memory=False, decomposed_form=True):
        """

        Return a sparse representation of projector from IR to Matsubara frequencies

        :param n1_n2_n3_n4_vec: list of tuples or 2D ndarray of shape (Nw, 3)
            Sampling points in fermionic convention
        :return: a list of three ndarrays of shape (Nw, 16, Nl) [reduced_memory=False] or
            a list of four ndarrays of shapes of (Nf, Nl), (3, Nw, 16).
        """

        assert decomposed_form

        n1_n2_n3_n4_vec = numpy.asarray(n1_n2_n3_n4_vec)
        nw = n1_n2_n3_n4_vec.shape[0]

        Unl_fb, coords = self.projector_to_matsubara_decoupled(n1_n2_n3_n4_vec)

        if reduced_memory:
            return [Unl_fb] + [coords[imat][:, :, -1] for imat in range(3)]
        else:
            import sparse
            num_o = Unl_fb.shape[0]
            r = []
            for imat in range(3):
                coords_mat = coords[imat,:,:].reshape((nw*16, 3)).transpose()
                coo = sparse.COO(coords_mat, numpy.ones(nw*16, dtype=int), shape=(nw, 16, num_o))
                M = coo.reshape((nw*16, num_o)).to_scipy_sparse().dot(Unl_fb)
                r.append(M.reshape((nw, 16, self._Nl)))
            return r

    def projector_to_matsubara_decoupled(self, n1_n2_n3_n4_vec):
        n1_n2_n3_n4_vec = numpy.asarray(n1_n2_n3_n4_vec)

        assert numpy.amax(n1_n2_n3_n4_vec) <= numpy.iinfo(numpy.int64).max
        assert numpy.amin(n1_n2_n3_n4_vec) >= numpy.iinfo(numpy.int64).min

        nw = n1_n2_n3_n4_vec.shape[0]

        # All unique fermionic frequencies for the one-particle basis
        n_f = numpy.unique(numpy.hstack((n1_n2_n3_n4_vec, -n1_n2_n3_n4_vec - 1)))

        # All unique bosonic frequencies for the one-particle basis
        n_b = n1_n2_n3_n4_vec[:, :, None] + n1_n2_n3_n4_vec[:, None, :] + 1
        n_b = numpy.unique(n_b)

        o_f = 2*n_f + 1
        o_b = 2*n_b

        # Precompute values of one-particle basis functions
        Unl_f = self._Bf.compute_Unl(n_f)[:, 0:self._Nl]
        Unl_b = self._Bb.compute_Unl(n_b)[:, 0:self._Nl]

        Unl_fb = numpy.vstack((Unl_f, Unl_b))
        o_fb = numpy.hstack((o_f, o_b))

        coords = _construct_coords(n1_n2_n3_n4_vec, o_fb)

        return Unl_fb, coords

    def projector_to_matsubara(self, n1, n2, n3, n4, decomposed_form = False):
        """
        Return a projector from IR to a Matsubara frequency
        """
        if n1 + n2 + n3 + n4 + 2 != 0:
            raise RuntimeError("The sum rule for frequencies is violated!")

        nvec = numpy.array([n1, n2, n3, n4])

        M1 = numpy.zeros((16, self._Nl), dtype=complex)
        M2 = numpy.zeros((16, self._Nl), dtype=complex)
        M3 = numpy.zeros((16, self._Nl), dtype=complex)
        # FFF representations
        for r in range(4):
            n1_p, n2_p, n3_p = nvec[idx_n1n2n3_FFF[r]]
            M1[r, :] = self._get_Unl_f(n1_p)
            M2[r, :] = self._get_Unl_f(n2_p)
            M3[r, :] = self._get_Unl_f(n3_p)

        # FBF representations
        for r in range(12):
            n1_p, n2_p, n4_p = nvec[idx_n1n2n4_FBF[r]]
            M1[r + 4, :] = self._get_Unl_f(n1_p)
            M2[r + 4, :] = self._get_Unl_b(n1_p + n2_p + 1)
            M3[r + 4, :] = self._get_Unl_f(-n4_p - 1)

        if decomposed_form:
            return M1, M2, M3
        else:
            return numpy.einsum('ri,rj,rk->rijk', M1, M2, M3)

    def sampling_points_matsubara(self, whichl):
        """
        Return sampling points
        """
        sp_o_f = 2 * self._Bf.sampling_points_matsubara(whichl) + 1
        sp_o_b = 2 * self._Bb.sampling_points_matsubara(whichl)
        sp_o = []
        Nf = len(sp_o_f)
        Nb = len(sp_o_b)

        ovec = numpy.zeros((4), dtype=int)

        # FFF
        append_sp = lambda o1, o2, o3 : sp_o.append((o1, o2, o3, -o1-o2-o3))
        for i, j, k in product(range(Nf), repeat=3):
            ovec[:3] = sp_o_f[i], sp_o_f[j], sp_o_f[k]
            ovec[3] = - ovec[0] - ovec[1] - ovec[2]
            append_sp(ovec[0], ovec[1], ovec[2]) # No. 1
            append_sp(ovec[0], ovec[1], ovec[3]) # No. 2
            append_sp(ovec[0], ovec[3], ovec[1]) # No. 3
            append_sp(ovec[3], ovec[0], ovec[1]) # No. 4

        # FBF
        perms = [numpy.array(p) for p in permutations([0, 1, 2, 3]) if p[0] < p[1]]
        for i, j, k in product(range(Nf), range(Nb), range(Nf)):
            of1, ob, of2 = sp_o_f[i], sp_o_b[j], sp_o_f[k]
            ovec[0], ovec[1], ovec[3] = of1, ob - of1, -of2
            ovec[2] = - ovec[0] - ovec[1] - ovec[3]
            for p in perms:
                sp_o.append(tuple(ovec[p]))

        conv = lambda x: tuple(map(o_to_matsubara_idx_f, x))

        return sorted(list(map(conv, list(set(sp_o)))))

    def _get_Unl_f(self, n):
        return self._Bf.compute_Unl([n])[:,0:self._Nl].reshape((self._Nl))

    def _get_Unl_b(self, n):
        return self._Bb.compute_Unl([n])[:,0:self._Nl].reshape((self._Nl))

def to_PH_convention(n1n2n3n4):
    """
    To particle-hole convention
    """
    if isinstance(n1n2n3n4, tuple):
        n = -n1n2n3n4[1] - 1
        np = n1n2n3n4[2]
        m = n1n2n3n4[0] + n1n2n3n4[1] + 1
        return (n, np, m)
    elif isinstance(n1n2n3n4, numpy.ndarray):
        n_points = n1n2n3n4.shape[0]
        assert n1n2n3n4.shape[1]==4
        r = numpy.empty((n_points, 3), dtype=int)
        r[:, 0] = -n1n2n3n4[:, 1] - 1
        r[:, 1] = n1n2n3n4[:, 2]
        r[:, 2] = n1n2n3n4[:, 0] + n1n2n3n4[:, 1] + 1
        return r
    else:
        raise RuntimeError("Invalid n1n2n3n4")

def from_PH_convention(n_np_m):
    """
    From particle-hole convention
    """
    if isinstance(n_np_m, tuple):
        n, np, m = n_np_m
        return (n + m, -n - 1, np, -np - 1 - m)
    elif isinstance(n_np_m, numpy.ndarray):
        n_points = n_np_m.shape[0]
        assert n_np_m.shape[1]==3
        r = numpy.empty((n_points, 4), dtype=int)
        n, np, m = n_np_m[:, 0], n_np_m[:, 1], n_np_m[:, 2]
        r[:, 0] = n + m
        r[:, 1] = -n - 1
        r[:, 2] = np
        r[:, 3] = -np - 1 - m
        return r
    else:
        raise RuntimeError("Invalid n_np_m")

from __future__ import print_function
import numpy
import scipy
import scipy.linalg
from irbasis import *
from itertools import product, permutations
from .internal import *
from .two_point_basis import *

# FFF representations (#1-#4)
idx_n1n2n3_FFF = list()
idx_n1n2n3_FFF.append(numpy.array((0,1,2))) # (iw_1, i_w2, i_w3)
idx_n1n2n3_FFF.append(numpy.array((0,1,3))) # (iw_1, i_w2, i_w4)
idx_n1n2n3_FFF.append(numpy.array((0,2,3))) # (iw_1, i_w3, i_w4)
idx_n1n2n3_FFF.append(numpy.array((1,2,3))) # (iw_2, i_w3, i_w4)

# FBF representations (#5-#16)
idx_n1n2n4_FBF = [numpy.array((p[0], p[1], p[3])) for p in permutations([0, 1, 2, 3]) if p[0] < p[3]]


class FourPoint(object):
    def __init__(self, Lambda, beta, cutoff = 1e-8, augmented=True):
        self._Lambda = Lambda
        self._beta = beta
        self._Bf = Basis(irbasis.load('F', Lambda), beta, cutoff)
        if augmented:
            self._Bb = Basis(augmented_basis_b(irbasis.load('B', Lambda)), beta, cutoff)
        else:
            self._Bb = Basis(irbasis.load('B', Lambda), beta, cutoff)
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

    def projector_to_matsubara_vec(self, n1_n2_n3_n4_vec, decomposed_form = False):
        """
        Return a projector from IR to Matsubara frequencies
        """
        n_f = []
        n_b = []
        nw = len(n1_n2_n3_n4_vec)
        for i in range(nw):
            for n in n1_n2_n3_n4_vec[i]:
                n_f.append(n)
                n_f.append(-n - 1)
            for n, np in product(n1_n2_n3_n4_vec[i], repeat=2):
                n_b.append(n + np + 1)
        self._Bf._precompute_Unl(n_f)
        self._Bb._precompute_Unl(n_b)

        if decomposed_form:
            r = [numpy.zeros((nw, 16, self._Nl), dtype=complex) for i in range(3)]
            for i, (n1, n2, n3, n4) in enumerate(n1_n2_n3_n4_vec):
                M1, M2, M3 = self.projector_to_matsubara(n1, n2, n3, n4, decomposed_form)
                r[0][i, :, :] = M1
                r[1][i, :, :] = M2
                r[2][i, :, :] = M3
            return r
        else:
            r = []
            for n1, n2, n3, n4 in n1_n2_n3_n4_vec:
                r.append(self.projector_to_matsubara(n1, n2, n3, n4, decomposed_form))
            return r

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
        sp_o_f = 2 * sampling_points_matsubara(self._Bf, whichl) + 1
        sp_o_b = 2 * sampling_points_matsubara(self._Bb, whichl)
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
            append_sp(ovec[0], ovec[2], ovec[3]) # No. 3
            append_sp(ovec[1], ovec[2], ovec[3]) # No. 4

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
    n = -n1n2n3n4[1] - 1
    np = n1n2n3n4[2]
    m = n1n2n3n4[0] + n1n2n3n4[1] + 1
    return (n, np, m)

def from_PH_convention(n_np_m):
    """
    From particle-hole convention
    """
    n, np, m = n_np_m
    return (n + m, -n - 1, np, -np - 1 - m)

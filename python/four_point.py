from __future__ import print_function
import numpy
import scipy
import scipy.linalg
from irbasis import *
from itertools import product, permutations
from .internal import *
from .two_point_basis import *

def _sign(s):
    return 1 if ((s % 2) == 0) else -1


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

    def normalized_S(self):
        Nl = self._Nl
        svec = numpy.zeros((16, Nl, Nl, Nl))
        sf = numpy.array([self._Bf.Sl(l) / self._Bf.Sl(0) for l in range(Nl)])
        sb = numpy.array([self._Bb.Sl(l) / self._Bb.Sl(0) for l in range(Nl)])
        svec[0:4, :, :, :] = sf[:, None, None] * sf[None, :, None] * sf[None, None, :]
        svec[4:, :, :, :] = sf[:, None, None] * sb[None, :, None] * sf[None, None, :]
        return svec

    def projector_to_matsubara_vec(self, n1_n2_n3_n4_vec):
        """
        Return a projector from IR to Matsubara frequencies
        """
        n_f = []
        n_b = []
        for i in range(len(n1_n2_n3_n4_vec)):
            for n in n1_n2_n3_n4_vec[i]:
                n_f.append(n)
                n_f.append(-n-1)
            for n, np in product(n1_n2_n3_n4_vec[i], repeat=2):
                n_b.append(n + np + 1)
        self._Bf._precompute_Unl(n_f)
        self._Bb._precompute_Unl(n_b)

        r = []
        for n1, n2, n3, n4 in n1_n2_n3_n4_vec:
            r.append(self.projector_to_matsubara(n1, n2, n3, n4))
        return r

    def projector_to_matsubara(self, n1, n2, n3, n4):
        """
        Return a projector from IR to a Matsubara frequency
        """
        if n1 + n2 + n3 + n4 + 2 != 0:
            raise RuntimeError("The sum rule for frequencies is violated!")

        nvec = numpy.array([n1, n2, n3, n4])

        M = numpy.zeros((16, self._Nl, self._Nl, self._Nl), dtype=complex)
        # FFF representations
        for r in range(4):
            n1_p, n2_p, n3_p = nvec[idx_n1n2n3_FFF[r]]
            tensor_left = numpy.einsum('i,j->ij', self._get_Unl_f(n1_p), self._get_Unl_f(n2_p))
            M[r, :, :, :] = numpy.einsum('ij,k->ijk', tensor_left, self._get_Unl_f(n3_p))

        # FBF representations
        for r in range(12):
            n1_p, n2_p, n4_p = nvec[idx_n1n2n4_FBF[r]]
            tensor_FB = numpy.einsum('i,j->ij', self._get_Unl_f(n1_p), self._get_Unl_b(n1_p + n2_p + 1))
            M[r + 4, :, :, :] = numpy.einsum('ij,k->ijk', tensor_FB, self._get_Unl_f(-n4_p - 1))

        return M

    def sampling_points_matsubara(self, whichl):
        """
        Return sampling points
        """
        sp_o_f = 2*sampling_points_matsubara(self._Bf, whichl) + 1
        sp_o_b = 2*sampling_points_matsubara(self._Bb, whichl)
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

        conv = lambda x: \
            (o_to_matsubara_idx_f(x[0]),\
               o_to_matsubara_idx_f(x[1]),\
               o_to_matsubara_idx_f(x[2]),\
               o_to_matsubara_idx_f(x[3]) )

        return list(map(conv, list(set(sp_o))))

    def _get_Unl_f(self, n):
        return self._Bf.compute_Unl([n])[:,0:self._Nl].reshape((self._Nl))

    def _get_Unl_b(self, n):
        return self._Bb.compute_Unl([n])[:,0:self._Nl].reshape((self._Nl))

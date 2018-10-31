from __future__ import print_function
import numpy
import scipy
import scipy.linalg
from irbasis import *
from itertools import product
from .internal import *
from .two_point_basis import *

def _sign(s):
    return 1 if ((s % 2) == 0) else -1

#def _A_imp(u1, u2, Nl):
    #mat1 = numpy.array([u1(l) for l in range(Nl)])
    #mat2 = numpy.array([u2(l) for l in range(Nl)])
    #return numpy.einsum('i,j->ij', mat1, mat2)

class FourPointPHView(object):
    def __init__(self, boson_freq, Lambda, beta, cutoff = 1e-8, augmented=True):
        if not isinstance(boson_freq, int):
            raise RuntimeError("boson_freq should be an integer")

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
        self._nshift = 2
        self._m = boson_freq
        self._o = 2 * boson_freq

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
        svec = numpy.zeros((3, 2, 2, Nl, Nl))
        sf = numpy.array([self._Bf.Sl(l) / self._Bf.Sl(0) for l in range(Nl)])
        sb = numpy.array([self._Bb.Sl(l) / self._Bb.Sl(0) for l in range(Nl)])
        for s1, s2 in product(range(2), range(2)):
            svec[0, s1, s2, :, :] = sf[:, None] * sf[None, :]
            svec[1, s1, s2, :, :] = sb[:, None] * sf[None, :]
            svec[2, s1, s2, :, :] = sb[:, None] * sf[None, :]
        return svec

    def projector_to_matsubara_vec(self, n1_n2_vec):
        """
        Return a projector from IR to Matsubara frequencies
        """
        o_f = []
        o_b = []
        for i in range(len(n1_n2_vec)):
            o1 = 2*n1_n2_vec[i][0] + 1
            o2 = 2*n1_n2_vec[i][1] + 1
            o_f.append(o1)
            o_f.append(o2)
            o_f.append(o1 + self._o)
            o_f.append(o2 + self._o)
            o_b.append(o1 - o2)
            o_b.append(o1 + o2 + self._o)
            o_b.append(o2 - o1)
            o_b.append(o2 + o1 + self._o)
        self._Bf._precompute_Unl(list(map(o_to_matsubara_idx_f, o_f)))
        self._Bb._precompute_Unl(list(map(o_to_matsubara_idx_b, o_b)))

        r = []
        for i in range(len(n1_n2_vec)):
            r.append(self.projector_to_matsubara(n1_n2_vec[i][0], n1_n2_vec[i][1]))
        return r

    def projector_to_matsubara(self, n1, n2):
        """
        Return a projector from IR to a Matsubara frequency
        """
        o1, o2 = 2*n1 + 1, 2*n2 + 1
        M = numpy.zeros((3, self._nshift, self._nshift, self._Nl, self._Nl), dtype=complex)
        for s1, s2 in product(range(self._nshift), repeat=2):
            sign = -1 * _sign(s1)
            # Note: with this signature, einsum does not actually perform any summation
            # HS: may be better to write like A[:, None] * B[None, :] ?
            M[0, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_Usol(s1, o1),             self._get_Usol(s2, o2))
            M[1, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_Usol(s1, o1 + sign * o2), self._get_Usol(s2, o2))
            M[2, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_Usol(s1, o2 + sign * o1), self._get_Usol(s2, o1))
        return M

    def sampling_points_matsubara(self, whichl):
        """
        Return sampling points in two-fermion-frequency convention
        """
        sp_o_f = 2 * sampling_points_matsubara(self._Bf, whichl) + 1
        sp_o_b = 2 * sampling_points_matsubara(self._Bb, whichl)
        sp_o = []
        Nf = len(sp_o_f)
        Nb = len(sp_o_b)
        for s1, s2 in product(range(2), repeat=2):
            for i, j in product(range(Nf), repeat=2):
                # Fermion, Fermion
                sp_o.append((sp_o_f[i] - s1 * self._o, sp_o_f[j] - s2 * self._o))
            for i, j in product(range(Nb), range(Nf)):
                # Boson, Fermion
                o2 = sp_o_f[j] - s2 * self._o
                o1 = sp_o_b[i] - s1 * self._o + _sign(s1) * o2
                sp_o.append((o1, o2))
                sp_o.append((o2, o1))

        # Remove duplicate elements
        sp_o = list(set(sp_o))

        # From "o" convention to normal Matsubara convention
        return [(o_to_matsubara_idx_f(p[0]), o_to_matsubara_idx_f(p[1])) for p in sp_o]

    def _get_Usol(self, s, o):
        if o%2 == 0:
            # boson
            return self._get_Usnl_b(s, o_to_matsubara_idx_b(o))
        else:
            # fermion
            return self._get_Usnl_f(s, o_to_matsubara_idx_f(o))

    def _get_Usnl_f(self, s, n):
        return self._Bf.compute_Unl([n + s * self._m])[:,0:self._Nl].reshape((self._Nl))

    def _get_Usnl_b(self, s, n):
        return self._Bb.compute_Unl([n + s * self._m])[:,0:self._Nl].reshape((self._Nl))

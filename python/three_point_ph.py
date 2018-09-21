from __future__ import print_function
import numpy
import scipy
import scipy.linalg
from irbasis import *
from itertools import product
from .internal import *
from .two_point_basis import *

def _sign(s):
    return 1 if s%2 == 0 else -1

#def _A_imp(u1, u2, Nl):
    #mat1 = numpy.array([u1(l) for l in range(Nl)])
    #mat2 = numpy.array([u2(l) for l in range(Nl)])
    #return numpy.einsum('i,j->ij', mat1, mat2)

class ThreePointPHBasis(object):
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

        self._Nl = min(self._Bf.dim, self._Bb.dim)

        self._nshift = 2
        self._m = boson_freq

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
        svec = numpy.zeros((3,2,2,Nl,Nl))
        sf = numpy.array([self._Bf.Sl(l)/self._Bf.Sl(0) for l in range(self._Nl)])
        sb = numpy.array([self._Bb.Sl(l)/self._Bb.Sl(0) for l in range(self._Nl)])

        for s1, s2 in product(range(2), range(2)):
            svec[0, s1, s2, :, :] = sf[:, None] * sf[None, :]
            svec[1, s1, s2, :, :] = sb[:, None] * sf[None, :]
            svec[2, s1, s2, :, :] = sb[:, None] * sf[None, :]

        return svec

    def projector_to_matsubara_vec(self, n1_n2_vec):
        """
        Return a projector from IR to Matsubara frequencies
        """
        n_f = []
        n_b = []
        for i in range(len(n1_n2_vec)):
            n1 = n1_n2_vec[i][0]
            n2 = n1_n2_vec[i][1]
            n_f.append(n1)
            n_f.append(n2)
            n_f.append(n1 + self._m)
            n_f.append(n2 + self._m)
            n_f.append(n1 - n2)
            n_f.append(n1 + n2 + self._m)
            n_f.append(n2 - n1)
            n_f.append(n2 + n1 + self._m)
        self._Bf._precompute_Unl(n_f)
        self._Bb._precompute_Unl(n_b)

        r = []
        for i in range(len(n1_n2_vec)):
            r.append(self.projector_to_matsubara(n1_n2_vec[i][0], n1_n2_vec[i][1]))
        return r

    def projector_to_matsubara(self, n1, n2):
        """
        Return a projector from IR to a Matsubara frequency
        """
        M = numpy.zeros((3, self._nshift, self._nshift, self._Nl, self._Nl), dtype=complex)
        for s1, s2 in product(range(self._nshift), repeat=2):
            sign = -1 * _sign(s1)
            M[0, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_Usnl_f(s1, n1),             self._get_Usnl_f(s2, n2))
            M[1, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_Usnl_b(s1, n1 + sign * n2), self._get_Usnl_f(s2, n2))
            M[2, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_Usnl_b(s1, n2 + sign * n1), self._get_Usnl_f(s2, n1))
        return M

    def sampling_points_matsubara(self):
        """
        Return sampling points in two-fermion-frequency convention
        """
        Nl = self._Nl
        whichl = Nl - 1
        sp_f = sampling_points_matsubara(self._Bf, whichl)
        sp_b = sampling_points_matsubara(self._Bb, whichl)

        sp = []

        Nf = len(sp_f)
        Nb = len(sp_b)
        for s1, s2 in product(range(2), repeat=2):
            for i, j in product(range(Nf), repeat=2):
                # Fermion, Fermion
                sp.append((sp_f[i] - s1 * self._m, sp_f[j] - s2 * self._m))

            for i, j in product(range(Nb), range(Nf)):
                # Boson, Fermion
                n2 = sp_f[j] - s2 * self._m
                n1 = sp_b[i] - s1 * self._m + _sign(s1) * n2

                sp.append((n1, n2))
                sp.append((n2, n1))

        return list(set(sp))

    def _get_Usnl_f(self, s, n):
        return self._Bf.compute_Unl([n + s * self._m])[:,0:self._Nl].reshape((self._Nl))

    def _get_Usnl_b(self, s, n):
        return self._Bb.compute_Unl([n + s * self._m])[:,0:self._Nl].reshape((self._Nl))

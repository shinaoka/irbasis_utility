from __future__ import print_function
import numpy
import scipy
import scipy.linalg
from irbasis import *
from itertools import product
from internal import *
from two_point_basis import *

def _sign(s):
    return 1 if s%2 == 0 else -1

def _A_imp(u1, u2, Nl):
    mat1 = numpy.array([u1(l) for l in range(Nl)])
    mat2 = numpy.array([u2(l) for l in range(Nl)])
    return numpy.einsum('i,j->ij', mat1, mat2)

class ThreePointPHBasis(object):
    def __init__(self, boson_freq, Lambda, beta, Nl = -1, cutoff = 1e-8):
        if not isinstance(boson_freq, int):
            raise RuntimeError("boson_freq should be an integer")

        self._Lambda = Lambda
        self._beta = beta
        self._Bf = Basis(irbasis.load('F', Lambda), beta)
        #self._Bb = Basis(irbasis.load('B', Lambda), beta)
        self._Bb = Basis(augmented_basis_b(Lambda), beta)

        if Nl == -1:
            self._Nl = min(self._Bf.dim(), self._Bb.dim())
        else:
            self._Nl = Nl
        assert Nl <= self._Bf.dim()
        assert Nl <= self._Bb.dim()

        self._nshift = 2
        self._m = boson_freq

    @property
    def Nl(self):
        return self._Nl

    @property
    def basis_beta_f(self):
        return self._Bf

    @property
    def basis_beta_b(self):
        return self._Bb

    def projector_to_matsubara_vec(self, n1_vec, n2_vec):
        n_f = []
        n_b = []
        for i in range(len(n1_vec)):
            n1 = n1_vec[i]
            n2 = n2_vec[i]
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
        for i in range(len(n1_vec)):
            r.append(self.projector_to_matsubara(n1_vec[i], n2_vec[i]))
        return r

    def projector_to_matsubara(self, n1, n2):
        M = numpy.zeros((3, self._nshift, self._nshift, self._Nl, self._Nl), dtype=complex)
        for s1, s2 in product(range(self._nshift), repeat=2):
            sign = -1 * _sign(s1)
            M[0, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_Usnl_f(s1, n1),             self._get_Usnl_f(s2, n2))
            M[1, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_Usnl_b(s1, n1 + sign * n2), self._get_Usnl_f(s2, n2))
            M[2, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_Usnl_b(s1, n2 + sign * n1), self._get_Usnl_f(s2, n1))
        return M

    def _get_Usnl_f(self, s, n):
        return self._Bf.compute_Unl([n + s * self._m])[:,0:self._Nl].reshape((self._Nl))

    def _get_Usnl_b(self, s, n):
        return self._Bb.compute_Unl([n + s * self._m])[:,0:self._Nl].reshape((self._Nl))

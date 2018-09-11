import numpy
import scipy
import scipy.linalg
from irbasis import *
from itertools import product
from internal import *
from augmented_basis import *

def _sign(s):
    return 1 if s%2 == 0 else -1

def _A_imp(u1, u2, Nl):
    mat1 = numpy.array([u1(l) for l in range(Nl)])
    mat2 = numpy.array([u2(l) for l in range(Nl)])
    return numpy.einsum('i,j->ij', mat1, mat2)

class basis_beta(object):
    def __init__(self, stat, beta, Lambda):
        wmax = Lambda/beta
        self._Lambda = Lambda
        self._b = irbasis.load(stat,  Lambda)
        self._scale = numpy.sqrt(2/self._beta)
        self._scale2 = numpy.sqrt(1/wmax)
        self._Unl_cache = {}

    def dim(self):
        return self._b.dim()

    def Ultau(self, l, tau):
        return self._scale * self._b.ulx(l, 2*tau/self._beta-1)

    def Ultau_all_l(self, tau):
        return self._scale * self._b.ulx_all_l(2*tau/self._beta-1)

    def Vlomega(self, l, omega):
        return self._scale2 * self._b.vly(l, omega/wmax)

    def _compute_Unl(self, nvec):
        nvec_compt = [n for n in nvec if not self._unl_cache.has_key(n)]
        if len(nvec_compt) == 0:
            return
        unl = self._bf.compute_unl(nvec_compt)
        for i in range(nlen(nvec_compt):
            self._Unl_cache[nvec_compt[i]] = numpy.sqrt(beta) * unl[i,:]

    def compute_Unl(self, nvec):
        self._compute_Unl(nvec)
        num_n = len(n_vec)
        Unl = numpy.empty((num_n, self.dim()), dtype=complex)
        for i in range(num_n):
            Unl[i, :] = self._Unl_cache[n_vec[i]]
        return Unl

class ThreePointPHBasis(object):
    def __init__(self, boson_freq, Lambda, beta, Nl = -1, cutoff = 1e-8):
        if not isinstance(boson_freq, int):
            raise RuntimeError("boson_freq should be an integer")

        self._Lambda = Lambda
        self._beta = beta
        self._bf = irbasis.load('F', Lambda)
        self._bb = AugmentedBasisBoson(Lambda)
        if Nl == -1:
            self._Nl = min(self._bf.dim(), self._bb.dim())
        else:
            self._Nl = Nl
        assert Nl <= self._bf.dim()
        assert Nl <= self._bb.dim()
        self._scale = numpy.sqrt(2/self._beta)
        self._sqrt_beta = numpy.sqrt(self._beta)

        self._unl_f_cache = {}
        self._unl_b_cache = {}

        self._nshift = 2
        self._m = boson_freq

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
        self.precompute_unl_f(numpy.unique(n_f))
        self.precompute_unl_b(numpy.unique(n_b))

        r = []
        for i in range(len(n1_vec)):
            r.append(self.projector_to_matsubara(n1_vec[i], n2_vec[i]))
        return r

    def projector_to_matsubara(self, n1, n2):
        M = numpy.zeros((3, self._nshift, self._nshift, self._Nl, self._Nl), dtype=complex)
        for s1, s2 in product(range(self._nshift), repeat=2):
            sign = -1 * _sign(s1)
            M[0, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_usnl_f(s1, n1),             self._get_usnl_f(s2, n2))
            M[1, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_usnl_b(s1, n1 + sign * n2), self._get_usnl_f(s2, n2))
            M[2, s1, s2, :, :] = numpy.einsum('i,j->ij', self._get_usnl_b(s1, n2 + sign * n1), self._get_usnl_f(s2, n1))
        return self._beta * M

    def precompute_unl_b(self, n_vec):
        num_n = len(n_vec)
        unl_b = self._bb.compute_unl(n_vec)[:,0:self._Nl].reshape((num_n, self._Nl))
        for i in range(num_n):
            self._unl_b_cache[n_vec[i]] = unl_b[i,:]

    def precompute_unl_f(self, n_vec):
        num_n = len(n_vec)
        unl_f = self._bf.compute_unl(n_vec)[:,0:self._Nl].reshape((num_n, self._Nl))
        for i in range(num_n):
            self._unl_f_cache[n_vec[i]] = unl_f[i,:]

    def _get_usnl_f(self, s, n):
        return self._get_unl_f(n + s * self._m)

    def _get_usnl_b(self, s, n):
        return self._get_unl_b(n + s * self._m)

    def _get_unl_f(self, n):
        if not n in self._unl_f_cache:
            self._unl_f_cache[n] = self._bf.compute_unl(numpy.array([n]))[:,0:self._Nl].reshape((self._Nl))
        return self._unl_f_cache[n]

    def _get_unl_b(self, n):
        if not n in self._unl_b_cache:
            self._unl_b_cache[n] = self._bb.compute_unl(numpy.array([n]))[:,0:self._Nl].reshape((self._Nl))
        return self._unl_b_cache[n]

    def _ultauf(self, l, tau):
        t, s = _my_mod(tau, self._beta) 
        x = 2 * t / self._beta - 1
        assert -1 < x and x < 1
        return s * self._bf.ulx(l, x) * self._scale

    def _ultaub(self, l, tau):
        t, s = _my_mod(tau, self._beta) 
        x = 2 * t / self._beta - 1
        assert -1 < x and x < 1
        return self._bb.ulx(l, x) * self._scale

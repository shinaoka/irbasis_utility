from __future__ import print_function
import numpy
import scipy
import scipy.linalg
from irbasis import *
from itertools import product, chain
from .internal import *
from .two_point_basis import *

def _sign(s):
    return 1 if ((s % 2) == 0) else -1

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
        # HS: For simplicity of implementation, we use the same number of basis functions for fermions and bosons
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

    def normalized_S(self, decomposed_form = False):
        Nl = self._Nl
        sf = numpy.array([self._Bf.Sl(l) / self._Bf.Sl(0) for l in range(Nl)])
        sb = numpy.array([self._Bb.Sl(l) / self._Bb.Sl(0) for l in range(Nl)])
        if decomposed_form:
            svec1 = numpy.zeros((3, self._nshift, self._nshift, Nl))
            svec2 = numpy.zeros((3, self._nshift, self._nshift, Nl))
            for s1, s2 in product(range(self._nshift), repeat=2):
                svec1[0, s1, s2, :] = sf[:]
                svec2[0, s1, s2, :] = sf[:]

                svec1[1, s1, s2, :] = sb[:]
                svec2[1, s1, s2, :] = sf[:]

                svec1[2, s1, s2, :] = sb[:]
                svec2[2, s1, s2, :] = sf[:]
            return svec1, svec2
        else:
            svec = numpy.zeros((3, self._nshift, self._nshift, Nl, Nl))
            for s1, s2 in product(range(self._nshift), repeat=2):
                svec[0, s1, s2, :, :] = sf[:, None] * sf[None, :]
                svec[1, s1, s2, :, :] = sb[:, None] * sf[None, :]
                svec[2, s1, s2, :, :] = sb[:, None] * sf[None, :]
            return svec

    def projector_to_matsubara_vec(self, n1_n2_vec, decomposed_form = False):
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

        if decomposed_form:
            nw = len(n1_n2_vec)
            r = [numpy.zeros((nw, 3, self._nshift, self._nshift, self._Nl), dtype=complex) for i in range(2)]
            for i in range(len(n1_n2_vec)):
                # M1 (3, nshift, nshift, Nl)
                # M2 (3, nshift, nshift, Nl)
                M1, M2 = self.projector_to_matsubara(n1_n2_vec[i][0], n1_n2_vec[i][1], True)
                r[0][i, :, :, :, :] = M1
                r[1][i, :, :, :, :] = M2
            return r
        else:
            r = []
            for i in range(len(n1_n2_vec)):
                r.append(self.projector_to_matsubara(n1_n2_vec[i][0], n1_n2_vec[i][1], False))
            return r

    def projector_to_matsubara(self, n1, n2, decomposed_form = False):
        """
        Return a projector from IR to a Matsubara frequency
        """
        o1, o2 = 2*n1 + 1, 2*n2 + 1
        if decomposed_form:
            M1 = numpy.zeros((3, self._nshift, self._nshift, self._Nl), dtype=complex)
            M2 = numpy.zeros((3, self._nshift, self._nshift, self._Nl), dtype=complex)
            for s1, s2 in product(range(self._nshift), repeat=2):
                sign = -1 * _sign(s1)
                M1[0, s1, s2, :] = self._get_Usol(s1, o1)
                M2[0, s1, s2, :] = self._get_Usol(s2, o2)

                M1[1, s1, s2, :] = self._get_Usol(s1, o1 + sign * o2)
                M2[1, s1, s2, :] = self._get_Usol(s2, o2)

                M1[2, s1, s2, :] = self._get_Usol(s1, o2 + sign * o1)
                M2[2, s1, s2, :] = self._get_Usol(s2, o1)
            return M1, M2
        else:
            M = numpy.zeros((3, self._nshift, self._nshift, self._Nl, self._Nl), dtype=complex)
            for s1, s2 in product(range(self._nshift), repeat=2):
                sign = -1 * _sign(s1)
                M[0, s1, s2, :, :] = self._get_Usol(s1, o1)[:, None]             * self._get_Usol(s2, o2)[None, :]
                M[1, s1, s2, :, :] = self._get_Usol(s1, o1 + sign * o2)[:, None] * self._get_Usol(s2, o2)[None, :]
                M[2, s1, s2, :, :] = self._get_Usol(s1, o2 + sign * o1)[:, None] * self._get_Usol(s2, o1)[None, :]
            return M

    def sampling_points_matsubara(self, whichl, full_freqs=True):
        """
        Return sampling points in two-fermion-frequency convention
        """
        sp_o_f = 2 * sampling_points_matsubara(self._Bf, whichl) + 1
        sp_o_b = 2 * sampling_points_matsubara(self._Bb, whichl)
        sp_o = []
        Nf = len(sp_o_f)
        Nb = len(sp_o_b)

        for s1, s2 in product(range(2), repeat=2):
            if full_freqs:
                for i, j in product(range(Nf), repeat=2):
                    # Fermion, Fermion
                    sp_o.append((sp_o_f[i] - s1 * self._o, sp_o_f[j] - s2 * self._o))
                for i, j in product(range(Nb), range(Nf)):
                    # Boson, Fermion
                    o2 = sp_o_f[j] - s2 * self._o
                    o1 = sp_o_b[i] - s1 * self._o + _sign(s1) * o2
                    sp_o.append((o1, o2))
                    sp_o.append((o2, o1))
            else:
                mini_f = [ Nf //2 - 1,  Nf //2]
                mini_b = [ Nb //2 ]

                # Fermion, Fermion
                for i, j in chain(product(range(Nf), mini_f), product(mini_f, range(Nf))):
                    sp_o.append((sp_o_f[i] - s1 * self._o, sp_o_f[j] - s2 * self._o))

                # Boson, Fermion
                for i, j in chain(product(range(Nb), mini_f), product(mini_b, range(Nf))):
                    o2 = sp_o_f[j] - s2 * self._o
                    o1 = sp_o_b[i] - s1 * self._o + _sign(s1) * o2
                    sp_o.append((o1, o2))
                    sp_o.append((o2, o1))

        # Remove duplicate elements
        sp_o = sorted(list(set(sp_o)))

        # From "o" convention to normal Matsubara convention
        return [(o_to_matsubara_idx_f(p[0]), o_to_matsubara_idx_f(p[1])) for p in sp_o]

    def to_two_fermion_convention(self, n, r, s1, s2):
        """
        Return a frequency point in the two-fermion convention corresponding to the given frequency point in the r-th representation

        Parameters
        ----------
        n : tuple of two integers
            A frequency point in the r-th representation

        r : int
            r = 0, 1, 2

        s1, s2 : int
            shift (0 or 1)

        Returns
        -------
        n_r0 : A tuple of two integers
            A tuple of two frequencies in the two-fermion convention
        """

        assert s1 == 0 or s1 == 1
        assert s2 == 0 or s2 == 1
        assert r == 0 or r == 1 or r == 2

        if r == 0:
            return (n[0] - s1 * self._m, n[1] - s2 * self._m)
        elif r == 1:
            o_r1 = (2 * n[0], 2 * n[1] + 1)
            o_tf_1 = o_r1[1] - s2 * self._o
            o_tf_0 = o_r1[0] - s1 * self._o + _sign(s1) * o_tf_1
            return (o_to_matsubara_idx_f(o_tf_0), o_to_matsubara_idx_f(o_tf_1))
        elif r == 2:
            o_r2 = (2 * n[0], 2 * n[1] + 1)
            o_tf_0 = o_r2[1] - s2 * self._o
            o_tf_1 = o_r2[0] - s1 * self._o + _sign(s1) * o_tf_0
            return (o_to_matsubara_idx_f(o_tf_0), o_to_matsubara_idx_f(o_tf_1))
        else:
            raise RuntimeError("r must be either 0, 1 or 2.")

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

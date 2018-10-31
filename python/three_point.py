from __future__ import print_function

import numpy
import scipy
import scipy.linalg
from irbasis import *
from itertools import product
from .internal import *
from .two_point_basis import *

class ThreePoint(object):
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
        svec = numpy.zeros((3, Nl, Nl))
        sf = numpy.array([self._Bf.Sl(l) / self._Bf.Sl(0) for l in range(Nl)])
        sb = numpy.array([self._Bb.Sl(l) / self._Bb.Sl(0) for l in range(Nl)])
        svec[0, :, :] = sf[:, None] * sf[None, :]
        svec[1, :, :] = sb[:, None] * sf[None, :]
        svec[2, :, :] = sf[:, None] * sb[None, :]

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
            n_b.append(n1 + n2 + 1)
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
        M = numpy.zeros((3, self._Nl, self._Nl), dtype=complex)
        # Note: with this signature, einsum does not actually perform any summation
        M[0, :, :] = numpy.einsum('i,j->ij', self._get_Unl_f(n1), self._get_Unl_f(n2))
        M[1, :, :] = numpy.einsum('i,j->ij', self._get_Unl_b(n1+n2+1), self._get_Unl_f(n2))
        M[2, :, :] = numpy.einsum('i,j->ij', self._get_Unl_f(n1), self._get_Unl_b(n1+n2+1))
        return M

    def sampling_points_matsubara(self, whichl):
        """
        Return sampling points in two-fermion-frequency convention
        """
        sp_o_f = 2*sampling_points_matsubara(self._Bf, whichl) + 1
        sp_o_b = 2*sampling_points_matsubara(self._Bb, whichl)
        sp_o = []
        Nf = len(sp_o_f)
        Nb = len(sp_o_b)

        # Fermion, Fermion
        for i, j in product(range(Nf), repeat=2):
            sp_o.append((sp_o_f[i], sp_o_f[j]))

        # Boson, Fermion
        for i, j in product(range(Nb), range(Nf)):
            o1 = sp_o_b[i] - sp_o_f[j]
            o2 = sp_o_f[j]
            sp_o.append((o1, o2))
            sp_o.append((o2, o1))

        conv = lambda x: tuple(map(o_to_matsubara_idx_f, x))

        return list(map(conv, list(set(sp_o))))


    def _get_Unl_f(self, n):
        return self._Bf.compute_Unl([n])[:,0:self._Nl].reshape((self._Nl))

    def _get_Unl_b(self, n):
        return self._Bb.compute_Unl([n])[:,0:self._Nl].reshape((self._Nl))

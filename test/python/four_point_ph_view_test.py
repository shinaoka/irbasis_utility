from __future__ import print_function
import unittest
import numpy
import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.four_point_ph_view import *
from irbasis_util.internal import *
from irbasis_util.regression import *

from atomic_limit import *

def _compute_Gl(phb, pole, s1, s2, r):
    Nl = phb.Nl
    coeffs = numpy.zeros((3, 2, 2, Nl, Nl))
    if r == 0:
        a = Gl_pole(phb.basis_beta_f, pole) 
        b = Gl_pole(phb.basis_beta_f, pole) 
    elif r == 1:
        a = Gl_pole(phb.basis_beta_b, pole) 
        b = Gl_pole(phb.basis_beta_f, pole) 
    elif r == 2:
        a = Gl_pole(phb.basis_beta_b, pole) 
        b = Gl_pole(phb.basis_beta_f, pole) 
    coeffs[r, s1, s2, :, :] = a[:Nl, None] * b[None, :Nl]
    return coeffs

def _compute_Giw(beta, pole, s1, s2, r, n1, n2, boson_freq):
    iwn_f = lambda n : 1J * (2*n + 1) * numpy.pi / beta 
    iwn_b = lambda n : 1J * (2 * n) * numpy.pi / beta
    o1, o2 = 2*n1+1, 2*n2+1
    if r == 0:
        return 1 / ((iwn_f(n1) + s1 * iwn_b(boson_freq) - pole) * (iwn_f(n2) + s2 * iwn_b(boson_freq) - pole))
    elif r == 1:
        o1_tmp = o1 + o2 * (-1) ** (s1+1)
        n1_tmp = o_to_matsubara_idx_b(o1_tmp)
        return 1 / ((iwn_b(n1_tmp) + s1 * iwn_b(boson_freq) - pole) * (iwn_f(n2) + s2 * iwn_b(boson_freq) - pole))
    elif r == 2:
        o_tmp = o2 + o1 * (-1)**(s1+1)
        n_tmp = o_to_matsubara_idx_b(o_tmp)
        return 1 / ((iwn_b(n_tmp) + s1 * iwn_b(boson_freq) - pole) * (iwn_f(n1) + s2 * iwn_b(boson_freq) - pole))

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_matsubara(self):
        boson_freq = 10
        Lambda = 10.0
        beta = 0.2
        wmax = Lambda/beta
        phb = FourPointPHView(boson_freq, Lambda, beta, 1e-12)
        Nl = phb.Nl
        iwn_f = lambda n : 1J*(2*n+1)*numpy.pi/beta 
        iwn_b = lambda n : 1J*(2*n)*numpy.pi/beta 
        pole = 0.2 * wmax
        n1 = 30
        n2 = 20
        prj = phb.projector_to_matsubara(n1, n2)
        # r = 0: Fermion, Fermion
        # r = 1: Boson, Fermion
        # r = 2: Boson, Fermion
        for r in range(3):
            for s1, s2 in product(range(2), repeat=2):
                coeffs = _compute_Gl(phb, pole, s1, s2, r)
                Giwn_ref = _compute_Giw(beta, pole, s1, s2, r, n1, n2, boson_freq)
                Giwn = numpy.sum(prj * coeffs)
                self.assertLessEqual(numpy.abs(Giwn_ref/Giwn - 1), 1e-7)

    def test_sampling_points_matsubara(self):
        boson_freq = 10
        Lambda = 10.0
        beta = 0.2
        alpha = 1e-15
        augmented = True
        wmax = Lambda / beta
        phb = FourPointPHView(boson_freq, Lambda, beta, 1e-5, augmented)
        Nl = phb.Nl
        whichl = Nl - 1
        pole = 0.2 * wmax
        # build the sampling frequency structure
        sp = phb.sampling_points_matsubara(whichl)
        S = phb.normalized_S()
        n_sp = len(sp)
        prj = numpy.array(phb.projector_to_matsubara_vec(sp))[:, :, :, :, :, :]# * S[None, :]
        prj_mat = prj[:, :, :, :, :, :].reshape((n_sp, 3 * 2 * 2 * Nl * Nl))
        # Build the check frequency structure
        n1n2_check = []
        niw = 100
        for i, j in product(range(-niw, niw, 10), repeat=2):
            n1n2_check.append((i, j))
        prj_check = numpy.array(phb.projector_to_matsubara_vec(n1n2_check))[:, :, :, :, :, :]# * S[None, :]
        # r = 0: Fermion, Fermion
        # r = 1: Boson, Fermion
        # r = 2: Boson, Fermion
        for r in range(3):
            for s1, s2 in product(range(2), repeat=2):
                #coeffs_ref = _compute_Gl(phb, pole, s1, s2, r)
                Giwn = numpy.array([_compute_Giw(beta, pole, s1, s2, r, n1n2[0],
                                                     n1n2[1], boson_freq) for n1n2 in sp])
                Giwn_check_ref = numpy.array([_compute_Giw(beta, pole, s1, s2, r,
                                                               n1n2[0], n1n2[1], boson_freq) for n1n2 in n1n2_check])
                coeffs = ridge_complex(prj_mat, Giwn, alpha).reshape((3, 2, 2, Nl, Nl))
                Giwn_check = numpy.dot(prj_check.reshape((len(n1n2_check), 3 * 2 * 2 * Nl * Nl)),
                                           (coeffs).reshape((3 * 2 * 2 * Nl * Nl)))
                self.assertLessEqual(numpy.amax(numpy.abs(Giwn_check - Giwn_check_ref)), 1e-5)

    def test_sampling_atomic_limit(self):
        boson_freq = 10
        Lambda = 10.0
        beta = 5.0
        U = 2.0
        alpha = 1e-15
        noise = 0.0
        augmented = True
        phb = FourPointPHView(boson_freq, Lambda, beta, 1e-5, augmented)
        Nl = phb.Nl
        whichl = Nl - 1
        # build the sampling frequency structure
        sp = phb.sampling_points_matsubara(whichl)
        S = phb.normalized_S()
        n_sp = len(sp)
        prj = numpy.array(phb.projector_to_matsubara_vec(sp))[:, :, :, :, :, :] * S[None, :]
        prj_mat = prj[:, :, :, :, :, :].reshape((n_sp, 3 * 2 * 2 * Nl * Nl))
        # Build the check frequency structure
        n1n2_check = []
        niw = 100
        niw_hf = 10000
        wide_niw_check = numpy.hstack((range(-niw_hf, -niw, 500), range(-niw, niw, 10), range(niw, niw_hf, 500)))
        for i, j in product(wide_niw_check, repeat=2):
            n1n2_check.append((i, j))
        prj_check = numpy.array(phb.projector_to_matsubara_vec(n1n2_check))[:, :, :, :, :, :] * S[None, :]
<<<<<<< HEAD
        Giwn = numpy.array([_G2_conn_ph(U, beta, n1n2[0], n1n2[1], boson_freq) for n1n2 in sp])
=======
        Giwn = numpy.array([G2_conn_ph(U, beta, n1n2[0], n1n2[1], boson_freq) for n1n2 in sp])
        noise_iwn = numpy.random.normal(loc=0.0, scale=noise, size=(len(sp)))
        Giwn = noise_iwn + Giwn
>>>>>>> e6f6fa19647cf42b1d3e288c1ff0bfc4ece53f51
        coeffs = ridge_complex(prj_mat, Giwn, alpha).reshape((3, 2, 2, Nl, Nl))
        Giwn_check = numpy.dot(prj_check.reshape((len(n1n2_check), 3 * 2 * 2 * Nl * Nl)),
                                   (coeffs).reshape((3 * 2 * 2 * Nl * Nl)))
        Giwn_check_ref = numpy.array([G2_conn_ph(U, beta, n1n2[0], n1n2[1], boson_freq) for n1n2 in n1n2_check])
        # absolute error
        self.assertLessEqual(numpy.amax(numpy.abs(Giwn_check - Giwn_check_ref)), 1e-3)

        
if __name__ == '__main__':
    unittest.main()

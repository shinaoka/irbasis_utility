from __future__ import print_function
import unittest
import numpy
import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.four_point_ph_view import *
from irbasis_util.internal import *
from irbasis_util.regression import *

def _delta(i, j):
    if i==j:
        return 1
    else:
        return 0

def _F_ph(U, beta, n, np, m):
    nu = (2 * n + 1) * numpy.pi / beta
    nu_p = (2 * np + 1) * numpy.pi / beta
    omega = 2 * m * numpy.pi / beta
    r1 = nu + omega
    r2 = nu_p + omega
    tmp = 1. / (nu * r1 * r2 * nu_p)
    Fuu = (-0.25 * beta * (U**2) * (_delta(n,np) - _delta(m,0)) *
               (1 + 0.25 * (U / nu)**2) * (1 + 0.25 * (U / r2)**2))    
    t1 = 0.125 * (U**3) * (nu**2 + r1**2 + r2**2 + nu_p**2) * tmp
    t2 = (3.0 / 16.0) * (U**5) * tmp
    t3 = (beta * 0.25 * (U**2) *
              (1 / (1 + numpy.exp(0.5 * beta * U)))
              * (2 * _delta(nu, -nu_p - m) + _delta(m, 0)) *
              (1 + 0.25 * (U / r1)**2) * (1 + 0.25 * (U / r2)**2))
    t4 = (-beta * 0.25 * (U**2) *
              (1 / (1 + numpy.exp(-0.5 * beta * U)))
              * (2 * _delta(nu, nu_p) + _delta(m, 0)) *
              (1 + 0.25 * (U / nu)**2) * (1 + 0.25 * (U / r2)**2))
    Fud = -U + t1 + t2 + t3 + t4
    return Fuu, Fud

def _G2_conn_ph(U, beta, n, np, m):
    Fuu, Fud = _F_ph(U, beta, n, np, m)
    nu = (2 * n + 1) * numpy.pi / beta
    nu_p = (2 * np + 1) * numpy.pi / beta
    omega = 2 * m * numpy.pi / beta    
    hU = 0.5 * U
    leggs_1 = nu * (nu + omega) * nu_p * (nu_p + omega)
    leggs_2 = ((hU**2 + nu**2) * (hU**2 + nu_p**2) *
                   (hU**2 + (nu + omega)**2) * (hU**2 + (nu_p + omega)**2))
    leggs = leggs_1 / leggs_2
    return leggs * Fuu + leggs * Fud

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
    if r == 0:
        return 1 / ((iwn_f(n1) + s1 * iwn_b(boson_freq) - pole) * (iwn_f(n2) + s2 * iwn_b(boson_freq) - pole))
    elif r == 1:
        n1_tmp = n1 + n2 * (-1) ** (s1+1)
        return 1 / ((iwn_b(n1_tmp) + s1 * iwn_b(boson_freq) - pole) * (iwn_f(n2) + s2 * iwn_b(boson_freq) - pole))
    elif r == 2:
        n_tmp = n2 + n1 * (-1)**(s1+1)
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
                assert numpy.abs(Giwn_ref/Giwn - 1) < 1e-7

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
        augmented = True
        wmax = Lambda / beta
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
        # r = 0: Fermion, Fermion
        # r = 1: Boson, Fermion
        # r = 2: Boson, Fermion
        Giwn = numpy.array([_G2_conn_ph(U, beta, n1n2[0], n1n2[1], boson_freq) for n1n2 in sp])
        print ("adding noise")
        noise_iwn = numpy.random.normal(loc=0.0, scale=0.0001, size=(len(sp)))
        Giwn = noise_iwn + Giwn
        coeffs = ridge_complex(prj_mat, Giwn, alpha).reshape((3, 2, 2, Nl, Nl))
        Giwn_check = numpy.dot(prj_check.reshape((len(n1n2_check), 3 * 2 * 2 * Nl * Nl)),
                                   (coeffs).reshape((3 * 2 * 2 * Nl * Nl)))
        Giwn_check_ref = numpy.array([_G2_conn_ph(U, beta, n1n2[0], n1n2[1], boson_freq) for n1n2 in n1n2_check])
        # absolute error
        #self.assertLessEqual(numpy.amax(numpy.abs(Giwn_check - Giwn_check_ref), 1e-3)
        # try relative error
        self.assertLessEqual(numpy.amax(numpy.abs(Giwn_check - Giwn_check_ref) /
                                            (numpy.abs(Giwn_check_ref) + numpy.abs(Giwn_check))), 1e-3)

        
if __name__ == '__main__':
    unittest.main()

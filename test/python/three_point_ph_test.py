from __future__ import print_function
import unittest
import numpy
import irbasis

from two_point_basis import *
from three_point_ph import *
from internal import *
from regression import *

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

def _compute_Giw(phb, pole, s1, s2, r, n1, n2, boson_freq):
    beta = phb.beta
    iwn_f = lambda n : 1J*(2*n+1)*numpy.pi/beta 
    iwn_b = lambda n : 1J*(2*n)*numpy.pi/beta 

    if r == 0:
        return 1/((iwn_f(n1) + s1 * iwn_b(boson_freq) - pole) * (iwn_f(n2) + s2 * iwn_b(boson_freq) - pole))
    elif r == 1:
        n1_tmp = n1 + n2 * (-1) ** (s1+1)
        return 1/((iwn_b(n1_tmp) + s1 * iwn_b(boson_freq) - pole) * (iwn_f(n2) + s2 * iwn_b(boson_freq) - pole))
    elif r == 2:
        n_tmp = n2 + n1 * (-1) ** (s1+1)
        return 1/((iwn_b(n_tmp) + s1 * iwn_b(boson_freq) - pole) * (iwn_f(n1) + s2 * iwn_b(boson_freq) - pole))

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_matsubara(self):
        boson_freq = 10
        Lambda = 10.0
        beta = 0.2
        wmax = Lambda/beta
        phb = ThreePointPHBasis(boson_freq, Lambda, beta, 1e-12)
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
                Giwn_ref = _compute_Giw(phb, pole, s1, s2, r, n1, n2, boson_freq)
                Giwn = numpy.sum(prj * coeffs)
                assert numpy.abs(Giwn_ref/Giwn - 1) < 1e-7

    def test_sampling_points_matsubara(self):
        boson_freq = 10
        Lambda = 10.0
        beta = 0.2
        alpha = 1e-15
        augmented = True

        wmax = Lambda/beta
        phb = ThreePointPHBasis(boson_freq, Lambda, beta, 1e-5, augmented)
        Nl = phb.Nl

        pole = 0.2 * wmax

        sp = phb.sampling_points_matsubara()
        n_sp = len(sp)

        n1n2_check = []
        niw = 100
        for i,j in product(range(-niw,niw,10), repeat=2):
            n1n2_check.append((i,j))
        S = phb.normalized_S()
        prj_check = numpy.array(phb.projector_to_matsubara_vec(n1n2_check))[:, :,:,:,:,:] * S[None, :]

        # r = 0: Fermion, Fermion
        # r = 1: Boson, Fermion
        # r = 2: Boson, Fermion
        for r in range(3):
            for s1, s2 in product(range(2), repeat=2):
                coeffs_ref = _compute_Gl(phb, pole, s1, s2, r)
                Giwn = numpy.array([_compute_Giw(phb, pole, s1, s2, r, n1n2[0], n1n2[1], boson_freq) for n1n2 in sp])

                prj = numpy.array(phb.projector_to_matsubara_vec(sp))[:, :,:,:,:,:] * S[None, :]

                prj_mat = prj[:,:,:,:,:,:].reshape((n_sp, 3*2*2*Nl*Nl))
                coeffs = ridge_complex(prj_mat, Giwn, alpha).reshape((3,2,2,Nl,Nl)) * S

                Giwn_check_ref = numpy.array([_compute_Giw(phb, pole, s1, s2, r, n1n2[0], n1n2[1], boson_freq) for n1n2 in n1n2_check])
                Giwn_check = numpy.dot(prj_check.reshape((len(n1n2_check), 3*2*2*Nl*Nl)), (coeffs/S).reshape((3*2*2*Nl*Nl)))
                #print("diff ", alpha, numpy.amax(numpy.abs(Giwn_check - Giwn_check_ref)))
                self.assertLessEqual(numpy.amax(numpy.abs(Giwn_check - Giwn_check_ref)), 1e-5)

if __name__ == '__main__':
    unittest.main()

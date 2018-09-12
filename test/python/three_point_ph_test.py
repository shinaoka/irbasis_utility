from __future__ import print_function
import unittest
import numpy
import irbasis

from two_point_basis import *
from three_point_ph import *
from internal import *

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_matsubara(self):
        boson_freq = 10
        Lambda = 10.0
        beta = 0.2
        wmax = Lambda/beta
        phb = ThreePointPHBasis(boson_freq, Lambda, beta)
        Nl = phb.Nl

        iwn_f = lambda n : 1J*(2*n+1)*numpy.pi/beta 
        iwn_b = lambda n : 1J*(2*n)*numpy.pi/beta 

        pole = 0.2 * wmax

        n1 = 30
        n2 = 20
        prj = phb.projector_to_matsubara(n1, n2)

        print("pole ", pole)
        print("wmax ", wmax)

        # Fermion, Fermion
        for s1, s2 in product(range(2), repeat=2):
            a = Gl_pole(phb.basis_beta_f, pole) 
            b = Gl_pole(phb.basis_beta_f, pole) 
            coeffs = numpy.zeros((3, 2, 2, Nl, Nl))
            coeffs[0, s1, s2, :, :] = a[:Nl, None] * b[None, :Nl]
            Giwn_ref = 1/((iwn_f(n1) + s1 * iwn_b(boson_freq) - pole) * (iwn_f(n2) + s2 * iwn_b(boson_freq) - pole))
            Giwn = numpy.sum(prj * coeffs)
            print(Giwn_ref, Giwn, Giwn_ref/Giwn)
            assert numpy.abs(Giwn_ref/Giwn - 1) < 1e-8

        # Boson, Fermion
        for s1, s2 in product(range(2), repeat=2):
            a = Gl_pole(phb.basis_beta_b, pole) 
            b = Gl_pole(phb.basis_beta_f, pole) 
            coeffs = numpy.zeros((3, 2, 2, Nl, Nl))
            coeffs[1, s1, s2, :, :] = a[:Nl, None] * b[None, :Nl]
            n1_tmp = n1 + n2 * (-1) ** (s1+1)
            Giwn_ref = 1/((iwn_b(n1_tmp) + s1 * iwn_b(boson_freq) - pole) * (iwn_f(n2) + s2 * iwn_b(boson_freq) - pole))
            Giwn = numpy.sum(prj * coeffs)
            print("debug", Giwn_ref/Giwn)
            assert numpy.abs(Giwn_ref/Giwn - 1) < 1e-8

        # Boson, Fermion
        for s1, s2 in product(range(2), repeat=2):
            a = Gl_pole(phb.basis_beta_b, pole) 
            b = Gl_pole(phb.basis_beta_f, pole) 
            coeffs = numpy.zeros((3, 2, 2, Nl, Nl))
            coeffs[2, s1, s2, :, :] = a[:Nl, None] * b[None, :Nl]
            n_tmp = n2 + n1 * (-1) ** (s1+1)
            Giwn_ref = 1/((iwn_b(n_tmp) + s1 * iwn_b(boson_freq) - pole) * (iwn_f(n1) + s2 * iwn_b(boson_freq) - pole))
            Giwn = numpy.sum(prj * coeffs)
            assert numpy.abs(Giwn_ref/Giwn - 1) < 1e-8

if __name__ == '__main__':
    unittest.main()

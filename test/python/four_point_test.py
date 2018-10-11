from __future__ import print_function
import unittest
import numpy
import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.four_point import *
from irbasis_util.internal import *
from irbasis_util.regression import *

from atomic_limit import *


def G1_iw_pole_f(n, pole, beta):
    return 1/(1J * (2 * n + 1) * numpy.pi / beta - pole)

def G1_iw_pole_b(n, pole, beta):
    return 1/(1J * (2 * n) * numpy.pi / beta - pole)

def _compute_G4pt_iw(beta, pole, r, nvec):
    if r == 0:
        return G1_iw_pole_f(nvec[0], pole, beta) * G1_iw_pole_f(nvec[1], pole, beta) * G1_iw_pole_f(nvec[2], pole, beta)
    elif r == 4:
        return G1_iw_pole_f(nvec[0], pole, beta) * G1_iw_pole_b(nvec[0] + nvec[1] + 1, pole, beta)\
               * G1_iw_pole_f(-nvec[3] - 1, pole, beta)
    else:
        raise RuntimeError("Not supported")

def _outer_product(Gl1, Gl2, Gl3):
    tensor12 =  Gl1[:, numpy.newaxis] * Gl2[numpy.newaxis, :]
    return tensor12 * Gl3[numpy.newaxis, numpy.newaxis, :]

def _compute_G4pt_l(b4pt, pole, r):
    Nl = b4pt.Nl
    Gl = numpy.zeros((16, Nl, Nl, Nl))
    if r == 0:
        Gl[r, :, :, :] = _outer_product(
            Gl_pole(b4pt.basis_beta_f, pole),
            Gl_pole(b4pt.basis_beta_f, pole),
            Gl_pole(b4pt.basis_beta_f, pole))
    else:
        raise RuntimeError("Not supported")

    return Gl

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_matsubara(self):
        Lambda = 10.0
        beta = 0.2
        wmax = Lambda/beta
        b4pt = FourPoint(Lambda, beta, 1e-8)
        Nl = b4pt.Nl

        pole = 0.2 * wmax

        n1, n2, n3 = 1, 2, 4
        n4 = - n1 - n2 - n3

        prj = b4pt.projector_to_matsubara(n1, n2, n3, n4)
        print("A", prj.shape)

        for r in [0]:
            giw_4pt = _compute_G4pt_iw(beta, pole, r, numpy.array([n1, n2, n3, n4]))
            gl_4pt = _compute_G4pt_l(b4pt, pole, r)

            print(numpy.sum(prj[r,:,:,:] * gl_4pt[r,:,:,:])/ giw_4pt)
            self.assertLess(numpy.abs(numpy.sum(prj * gl_4pt) - giw_4pt), 1e-10)

if __name__ == '__main__':
    unittest.main()
from __future__ import print_function
import unittest
import numpy
import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.three_point import *
from irbasis_util.internal import *
from irbasis_util.regression import *

from common import Gl_pole_F, Gl_pole_barB

def G1_iw_pole_f(n, pole, beta):
    return 1/(1J * (2 * n + 1) * numpy.pi / beta - pole)

def G1_iw_pole_b(n, pole, beta):
    return 1/(1J * (2 * n) * numpy.pi / beta - pole)

def _compute_G3pt_iw(beta, pole, r, nvec):
    if r == 0:
        return G1_iw_pole_f(nvec[0], pole, beta) * G1_iw_pole_f(nvec[1], pole, beta)
    elif r == 1:
        return G1_iw_pole_b(nvec[0] + nvec[1] + 1, pole, beta) * G1_iw_pole_f(nvec[1], pole, beta)
    elif r == 2:
        return G1_iw_pole_f(nvec[0], pole, beta) * G1_iw_pole_b(nvec[0] + nvec[1] + 1, pole, beta)
    else:
        raise RuntimeError("Not supported")

def _compute_G3pt_l(b3pt, pole, r):
    Nl = b3pt.Nl
    Gl = numpy.zeros((3, Nl, Nl))
    if r == 0:
        Gl[r, :, :] = Gl_pole_F(b3pt.basis_beta_f, pole)[:Nl, None] * Gl_pole_F(b3pt.basis_beta_f, pole)[None, :Nl]
    elif r == 1:
        Gl[r, :, :] = Gl_pole_barB(b3pt.basis_beta_b, pole)[:Nl, None] * Gl_pole_F(b3pt.basis_beta_f, pole)[None, :Nl]
    elif r == 2:
        Gl[r, :, :] = Gl_pole_F(b3pt.basis_beta_f, pole)[:Nl, None] * Gl_pole_barB(b3pt.basis_beta_b, pole)[None, :Nl]
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
        b3pt = ThreePoint(Lambda, beta, 1e-12)
        Nl = b3pt.Nl

        pole = 0.2 * wmax

        n1, n2 = 1, 2

        prj = b3pt.projector_to_matsubara(n1, n2)

        for r in [0, 1, 2]:
            giw_3pt = _compute_G3pt_iw(beta, pole, r, numpy.array([n1, n2]))
            gl_3pt = _compute_G3pt_l(b3pt, pole, r)

            diff = numpy.abs(numpy.sum(prj * gl_3pt) - giw_3pt)
            self.assertLess(diff, 1e-10)

    def test_sampling_points_matsubara(self):
        Lambda = 10.0
        beta = 0.2
        alpha = 1e-15
        augmented = True
        wmax = Lambda / beta
        b3pt = ThreePoint(Lambda, beta, 1e-2, augmented)
        Nl = b3pt.Nl
        whichl = Nl - 1
        pole = 0.2 * wmax

        # build the sampling frequency structure
        sp = b3pt.sampling_points_matsubara(whichl)
        S = b3pt.normalized_S()
        n_sp = len(sp)
        prj = numpy.array(b3pt.projector_to_matsubara_vec(sp))
        prj_mat = prj.reshape((n_sp, 3 * Nl**2))

        # Build the check frequency structure
        n12_check = []
        niw = 100
        for i, j in product(range(-niw, niw, 10), repeat=2):
            n12_check.append((i, j))
        prj_check = numpy.array(b3pt.projector_to_matsubara_vec(n12_check))

        # Test No. 1, 2, 3
        for r in [0, 1, 2]:
            Giwn = numpy.array([_compute_G3pt_iw(beta, pole, r, n12) for n12 in sp])
            Giwn_check_ref = numpy.array([_compute_G3pt_iw(beta, pole, r, n12) for n12 in n12_check])
            coeffs = ridge_complex(prj_mat, Giwn, alpha).reshape((3, Nl, Nl))
            Giwn_check = numpy.dot(prj_check.reshape((len(n12_check), 3 * Nl**2)), (coeffs).reshape((3 * Nl**2)))
            self.assertLessEqual(numpy.amax(numpy.abs(Giwn_check - Giwn_check_ref)), 1e-2)


if __name__ == '__main__':
    unittest.main()

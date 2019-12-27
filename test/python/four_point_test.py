from __future__ import print_function
import unittest
import numpy
import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.four_point import *
from irbasis_util.internal import *
from irbasis_util.regression import *

from common import *


def G1_iw_pole_f(n, pole, beta):
    return 1/(1J * (2 * n + 1) * numpy.pi / beta - pole)

def G1_iw_pole_b(n, pole, beta):
    return 1/(1J * (2 * n) * numpy.pi / beta - pole)

def _compute_G4pt_iw(beta, pole, r, nvec):
    if r == 0:
        return G1_iw_pole_f(nvec[0], pole, beta) * G1_iw_pole_f(nvec[1], pole, beta) * G1_iw_pole_f(nvec[2], pole, beta)
    elif r == 1:
        return G1_iw_pole_f(nvec[0], pole, beta) * G1_iw_pole_f(nvec[1], pole, beta) * G1_iw_pole_f(nvec[3], pole, beta)
    elif r == 4:
        return G1_iw_pole_f(nvec[0], pole, beta) * G1_iw_pole_b(nvec[0] + nvec[1] + 1, pole, beta)\
               * G1_iw_pole_f(-nvec[3] - 1, pole, beta)
    elif r == 5:
        return G1_iw_pole_f(nvec[0], pole, beta) * G1_iw_pole_b(nvec[0] + nvec[1] + 1, pole, beta) \
               * G1_iw_pole_f(-nvec[2] - 1, pole, beta)
    else:
        raise RuntimeError("Not supported")

def _outer_product(Gl1, Gl2, Gl3):
    tensor12 =  Gl1[:, numpy.newaxis] * Gl2[numpy.newaxis, :]
    return tensor12[:, :, numpy.newaxis] * Gl3[numpy.newaxis, numpy.newaxis, :]

def _compute_G4pt_l(b4pt, pole, r):
    Nl = b4pt.Nl
    Gl = numpy.zeros((16, Nl, Nl, Nl))
    if r >= 0  and r <= 3:
        Gl[r, :, :, :] = _outer_product(
            Gl_pole_F(b4pt.basis_beta_f, pole)[:Nl],
            Gl_pole_F(b4pt.basis_beta_f, pole)[:Nl],
            Gl_pole_F(b4pt.basis_beta_f, pole)[:Nl])
    elif r <= 15:
        Gl[r, :, :, :] = _outer_product(
            Gl_pole_F(b4pt.basis_beta_f, pole)[:Nl],
            Gl_pole_barB(b4pt.basis_beta_b, pole)[:Nl],
            Gl_pole_F(b4pt.basis_beta_f, pole)[:Nl])
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
        b4pt = FourPoint(Lambda, beta, 1e-12)
        Nl = b4pt.Nl

        pole = 0.2 * wmax

        n1, n2, n3 = 1, 2, 4
        n4 = - n1 - n2 - n3 - 2

        prj = b4pt.projector_to_matsubara(n1, n2, n3, n4)

        # Test No. 1, 2, 5, 6
        for r in [0, 1, 4, 5]:
            giw_4pt = _compute_G4pt_iw(beta, pole, r, numpy.array([n1, n2, n3, n4]))
            gl_4pt = _compute_G4pt_l(b4pt, pole, r)

            diff = numpy.abs(numpy.sum(prj * gl_4pt) - giw_4pt)
            self.assertLess(diff, 1e-10)

    def test_sampling_points_matsubara(self):
        Lambda = 10.0
        beta = 0.2
        alpha = 1e-15
        augmented = True
        wmax = Lambda / beta
        b4pt = FourPoint(Lambda, beta, 1e-2, augmented)
        Nl = b4pt.Nl
        whichl = Nl - 1
        pole = 0.2 * wmax
        # build the sampling frequency structure
        sp = b4pt.sampling_points_matsubara(whichl)
        n_sp = len(sp)
        prj = b4pt.projector_to_matsubara_vec(sp)
        prj_mat = numpy.einsum('nrL,nrM,nrN->nrLMN', *prj, optimize=True).reshape((n_sp, 16 * Nl**3))
        # Build the check frequency structure
        n1234_check = []
        niw = 100
        for i, j, k in product(range(-niw, niw, 10), repeat=3):
            n1234_check.append((i, j, k, - i - j - k - 2))
        prj_check = b4pt.projector_to_matsubara_vec(n1234_check)
        prj_check = numpy.einsum('nrL,nrM,nrN->nrLMN', *prj_check, optimize=True)

        # Test No. 1, 2, 5, 6
        for r in [0, 1, 4, 5]:
            Giwn = numpy.array([ _compute_G4pt_iw(beta, pole, r, n1234) for n1234 in sp])
            Giwn_check_ref = numpy.array([_compute_G4pt_iw(beta, pole, r, n1234) for n1234 in n1234_check])
            coeffs = ridge_complex(prj_mat, Giwn, alpha).reshape((16, Nl, Nl, Nl))
            Giwn_check = numpy.dot(prj_check.reshape((len(n1234_check), 16 * Nl**3)), (coeffs).reshape((16 * Nl**3)))
            self.assertLessEqual(numpy.amax(numpy.abs(Giwn_check - Giwn_check_ref)), 1e-2)
        
    def test_transformation_to_PH(self):
        self.assertEqual(to_PH_convention(from_PH_convention( (0,1,2) )),  (0,1,2))

        n_points = 100
        n_np_m = numpy.random.randint(-100, 100, size=(n_points, 3))
        assert numpy.allclose(to_PH_convention(from_PH_convention(n_np_m)), n_np_m)


if __name__ == '__main__':
    unittest.main()

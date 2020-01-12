from __future__ import print_function
import unittest

from common import *

import numpy
#import irbasis
#from itertools import *

from irbasis_util.two_point_basis import *

Lambda = 1000.0
beta = 100.0
wmax = Lambda/beta
stat = 'B'


def sampling_points_leggauss(basis_beta, whichl, deg):
    """
    Computes the sample points and weights for composite Gauss-Legendre quadrature
    according to the zeros of the given basis function

    Parameters
    ----------
    basis_beta : Basis
        Basis object
    whichl: int
        Index of reference basis function "l"
    deg: int
        Number of sample points and weights between neighboring zeros

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points (in tau)
    y : ndarray
        1-D ndarray containing the weights.

    """

    check_type(basis_beta, [Basis])
    ulx = lambda x: basis_beta.basis_xy.ulx(whichl, x)
    section_edges = numpy.hstack((-1., find_zeros(ulx), 1.))

    x, y = composite_leggauss(deg, section_edges)

    return tau_for_x(x, basis_beta.beta), .5 * basis_beta.beta * y


all_stat = ['F', 'B', 'barB']

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_Utau(self):
        for stat in all_stat:
            with self.subTest('stat = %s' % (stat)):
                b, B = load_basis(stat, Lambda, beta)
                l = 1
                for tau in [0, 0.5, 0.9999999*beta]:
                    x = 2 * tau/beta - 1
                    self.assertAlmostEqual(numpy.sqrt(2/beta) * b.ulx(l, x), B.Ultau(l, tau), delta=1e-10)

    def test_Unl(self):
        for stat in all_stat:
            with self.subTest('stat = %s' % (stat)):
                b, B = load_basis(stat, Lambda, beta)
                n = numpy.arange(-10, 10)
                Unl = B.compute_Unl(n)
                unl = b.compute_unl(n)
                self.assertTrue(numpy.allclose(Unl, numpy.sqrt(beta) * unl, 1e-10))

    def test_sampling_points_tau(self):
        for stat in all_stat:
            with self.subTest('stat = %s' % (stat)):
                b, B = load_basis(stat, Lambda, beta)
                dim = B.dim
                whichl = dim -1
                taus, w = sampling_points_leggauss(B, whichl, deg=10)

                self.assertAlmostEqual(numpy.sum(w), beta, delta=1e-10)

    def test_extended_basis(self):
        nvec = numpy.arange(-10, 10)
        num_n = len(nvec)
        for stat, stat_V in [('F', 'FV'), ('barB', 'barBV')]:
            with self.subTest('stat = %s %s' % (stat, stat_V)):
                _, B = load_basis(stat, Lambda, beta)
                _, B_V = load_basis(stat_V, Lambda, beta)

                Unl = B.compute_Unl(nvec)
                Unl_V = B_V.compute_Unl(nvec)

                # The first basis function is a constant: sqrt(beta)
                numpy.testing.assert_allclose(Unl_V[:, 0], numpy.sqrt(beta) * numpy.ones(num_n, dtype=complex))

                numpy.testing.assert_allclose(Unl_V[:, 1:], Unl[:, :])


if __name__ == '__main__':
    unittest.main()

from __future__ import print_function
import unittest
import numpy
import irbasis
from itertools import *

from irbasis_util.two_point_basis import *

Lambda = 1000.0
beta = 100.0
wmax = Lambda/beta

# TO DO: parameterize stat
stat = 'B'

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

        if stat in ['F', 'B']:
            self._b = irbasis.load(stat, Lambda)
        elif stat == 'barB':
            self._b = augmented_basis_b(irbasis.load('B', Lambda))
        self._B = Basis(self._b, beta)

    def test_Utau(self):
        l = 1
        for tau in [0, 0.5, 0.9999999*beta]:
            x = 2 * tau/beta - 1
            self.assertAlmostEqual(numpy.sqrt(2/beta) * self._b.ulx(l, x), self._B.Ultau(l, tau), delta=1e-10)

    def test_Unl(self):
        n = numpy.arange(-10, 10)
        Unl = self._B.compute_Unl(n)
        unl = self._b.compute_unl(n)
        self.assertTrue(numpy.allclose(Unl, numpy.sqrt(beta) * unl, 1e-10))

    def test_sampling_points_tau(self):
        dim = self._B.dim
        whichl = dim -1
        taus, w = sampling_points_leggauss(self._B, whichl, deg=10)

        self.assertAlmostEqual(numpy.sum(w), beta, delta=1e-10)

        ntau = len(taus)
        basis_vals = numpy.array([self._B.Ultau(l, tau) for tau in taus for l in range(dim)]).reshape((ntau, dim))
        overlap = numpy.dot(basis_vals.T, w[:, None] * basis_vals)

        #print(overlap)
        #evals,evecs = eigh_ordered(overlap)
        #print(evals)

        self.assertTrue(numpy.allclose(overlap, numpy.identity(dim), 1e-10))


if __name__ == '__main__':
    unittest.main()

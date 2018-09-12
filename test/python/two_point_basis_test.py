from __future__ import print_function
import unittest
import numpy
import irbasis

from two_point_basis import *

Lambda = 1000.0
beta = 100.0
wmax = Lambda/beta

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

        self._b = irbasis.load('F', Lambda)
        self._B = Basis(self._b, beta)

    def test_Utau(self):
        l = 1
        for tau in [0, 0.5, beta]:
            x = 2 * tau/beta - 1
            self.assertAlmostEqual(numpy.sqrt(2/beta) * self._b.ulx(l, x), self._B.Ultau(l, tau), delta=1e-10)

    def test_Unl(self):
        n = numpy.arange(-10, 10)
        Unl = self._B.compute_Unl(n)
        unl = self._b.compute_unl(n)
        self.assertTrue(numpy.allclose(Unl, numpy.sqrt(beta) * unl, 1e-10))

if __name__ == '__main__':
    unittest.main()

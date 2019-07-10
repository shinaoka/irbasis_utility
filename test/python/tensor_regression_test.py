from __future__ import print_function

import unittest
import numpy

import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.internal import *
from irbasis_util.regression import *
from irbasis_util.tensor_regression import fit, predict

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    """
    def test_als(self):
        numpy.random.seed(100)

        Nw = 50
        Nr = 2
        linear_dim = 5
        D = 2
        num_o = 1**4
        alpha = 1e-10

        def create_tensor_3(N, M, L):
            rand = numpy.random.rand(N, M, L) + 1J * numpy.random.rand(N, M, L)
            return rand

        for freq_dim in [2,3]:
            tensors_A = [create_tensor_3(Nw, Nr, linear_dim) for i in range(freq_dim)]
            y = numpy.random.randn(Nw, num_o) +\
                1J * numpy.random.randn(Nw, num_o)

            numpy.random.seed(100)
            model = OvercompleteGFModel(Nw, Nr, freq_dim, num_o, linear_dim, tensors_A, y, alpha, D)
            info = optimize_als(model, nite = 400, rtol=1e-10, verbose=1)

            self.assertLess(numpy.abs(info['losss'][-1] - info['losss'][-2])/numpy.abs(info['losss'][-2]), 1e-3)
    """

    def test_als(self):
        numpy.random.seed(100)

        Nw = 20
        Nr = 2
        linear_dim = 5
        D = 10
        num_o = 2
        alpha = 0.0

        def create_tensor_3(N, M, L):
            rand = numpy.random.rand(N, M, L) + 1J * numpy.random.rand(N, M, L)
            return rand

        for freq_dim in [2, 3]:
            tensors_A = [create_tensor_3(Nw, Nr, linear_dim) for i in range(freq_dim)]
            y = numpy.random.randn(Nw, num_o) +\
                1J * numpy.random.randn(Nw, num_o)

            numpy.random.seed(100)


            x_tensors = fit(y, tensors_A, D, 100, verbose=1, x0=None,
                            random_init=True, alpha=1e-8, comm=None, seed=1, nesterov=True)

            y_pred = predict(tensors_A, x_tensors)

            amax = numpy.amax(numpy.abs(y))
            adiff = numpy.amax(numpy.abs(y - y_pred))

            print(adiff/amax)
            assert adiff/amax < 1e-1


if __name__ == '__main__':
    unittest.main()

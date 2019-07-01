from __future__ import print_function

import unittest
import numpy

import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.internal import *
from irbasis_util.regression import *
from irbasis_util.tensor_regression_mps import fit, predict

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_als(self):
        numpy.random.seed(100)

        Nw = 4
        Nr = 2
        linear_dim = 5
        D = 10
        num_o = 4

        def create_tensor_3(N, M, L):
            rand = numpy.random.rand(N, M, L) + 1J * numpy.random.rand(N, M, L)
            return rand

        for freq_dim in [2, 3]:
            tensors_A = [create_tensor_3(Nw, Nr, linear_dim) for i in range(freq_dim)]
            y = numpy.random.randn(Nw, num_o) +\
                1J * numpy.random.randn(Nw, num_o)

            numpy.random.seed(100)

            x_tensors = fit(y, tensors_A, D, 10000, rtol=1e-5, verbose=0, random_init=True, optimize_alpha=-1, print_interval=20, comm=None, seed=1)

            y_pred, _ = predict(tensors_A, x_tensors)

            amax = numpy.amax(numpy.abs(y))
            adiff = numpy.amax(numpy.abs(y - y_pred))

            print(adiff/amax)
            assert adiff/amax < 1e-1

            #for i in range(Nw):
                #for j in range(num_o):
                    #print(i, j, y[i,j].real, y[i,j].imag, y_pred[i,j].real, y_pred[i,j].imag)
        #assert False


if __name__ == '__main__':
    unittest.main()

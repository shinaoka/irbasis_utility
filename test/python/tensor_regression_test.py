from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import unittest
import numpy

import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.internal import *
from irbasis_util.regression import *
from irbasis_util.tensor_regression import *

real_dtype = tf.float64
cmplx_dtype = tf.complex128

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)



    def test_als(self):
        numpy.random.seed(100)

        Nw = 100
        Nr = 2
        linear_dim = 2
        freq_dim = 2
        D = 20
        alpha = 0.1

        #Nw = 10000
        #Nr = 12
        #linear_dim = 30
        #freq_dim = 2
        #D = 10
        #alpha = 0.1

        def create_tensor_3(N, M, L):
            rand = numpy.random.rand(N, M, L) + 1J * numpy.random.rand(N, M, L)
            return tf.constant(rand, dtype=cmplx_dtype)

        tensors_A = [create_tensor_3(Nw, Nr, linear_dim) for i in range(freq_dim)]
        y = tf.constant(numpy.random.randn(Nw) + 1J * numpy.random.randn(Nw), dtype=cmplx_dtype)

        for solver in ['svd', 'lsqr']:
            numpy.random.seed(100)
            model = OvercompleteGFModel(Nw, Nr, freq_dim, linear_dim, tensors_A, y, alpha, D)
            info = optimize_als(model, nite = 1000, verbose=0, solver=solver)

            print("loss", info['losss'][-1])
            self.assertLess(numpy.abs(info['losss'][-1] - info['losss'][-2]), 1e-10)


if __name__ == '__main__':
    unittest.main()

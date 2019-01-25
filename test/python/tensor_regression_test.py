from __future__ import print_function

import unittest
import numpy

import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.internal import *
from irbasis_util.regression import *
from irbasis_util.tensor_regression import *

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
        num_orb = 2
        alpha = 0.1

        #Nw = 10000
        #Nr = 12
        #linear_dim = 30
        #freq_dim = 2
        #D = 10
        #alpha = 0.1

        def create_tensor_3(N, M, L):
            rand = numpy.random.rand(N, M, L) + 1J * numpy.random.rand(N, M, L)
            return rand

        tensors_A = [create_tensor_3(Nw, Nr, linear_dim) for i in range(freq_dim)]
        y = numpy.random.randn(Nw, num_orb, num_orb, num_orb, num_orb) +\
            1J * numpy.random.randn(Nw, num_orb, num_orb, num_orb, num_orb)

        for solver in ['svd', 'lsqr']:
            numpy.random.seed(100)
            model = OvercompleteGFModel(Nw, Nr, freq_dim, num_orb, linear_dim, tensors_A, y, alpha, D)
            info = optimize_als(model, nite = 1000, tol_rmse=1e-10, verbose=0, sketch_size_fact=1E+8, solver=solver)

            #print("loss", info['losss'][-1])
            #for i, loss in enumerate(info['losss']):
                #print(i, loss)
            self.assertLess(numpy.abs(info['rmses'][-1] - info['rmses'][-2]), 1e-9)


if __name__ == '__main__':
    unittest.main()

from __future__ import print_function

import unittest
import numpy

import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.internal import *
from irbasis_util.regression import *
from irbasis_util.tensor_regression_mps import fit, predict
import irbasis_util.tensor_regression
#from irbasis_util.tensor_regression import fit as fit_cp
#from irbasis_util.tensor_regression import predict as predict_cp


def from_cp_to_mps(x_tensors_cp):
    N = len(x_tensors_cp)
    D = x_tensors_cp[0].shape[0]

    x_tensors_mps = []
    x_tensors_mps.append(x_tensors_cp[0].copy())
    for i in range(1,N-1):
        phys_dim = x_tensors_cp[i].shape[1]
        x_mps = numpy.zeros((D, D, phys_dim), dtype=complex)
        for j in range(phys_dim):
            x_mps[:, :, j] = numpy.diag(x_tensors_cp[i][:, j])
        x_tensors_mps.append(x_mps)
    x_tensors_mps.append(x_tensors_cp[-1].copy())

    return x_tensors_mps



class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_als(self):
        numpy.random.seed(100)

        Nw = 20
        Nr = 2
        linear_dim = 5
        D = 2
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

            model = irbasis_util.tensor_regression.OvercompleteGFModel(Nw, Nr, freq_dim, num_o, linear_dim, tensors_A, y, alpha, D)
            info = irbasis_util.tensor_regression.optimize_als(model, nite = 400, rtol=1e-10, verbose=1)
            print("loss", info['losss'][-1])

            x0_mps = from_cp_to_mps(model.x_tensors())

            y_pred, _ = predict(tensors_A, x0_mps)
            loss_mps0 = numpy.linalg.norm(y - y_pred)**2

            #print("loss ", loss_mps0- info['losss'][-1])
            #numpy.testing.assert_allclose([loss_mps0], [info['losss'][-1]], atol=1e-5, rtol=1e-5)
            numpy.testing.assert_allclose(loss_mps0, info['losss'][-1])
            #loss_new = numpy.sqrt((numpy.linalg.norm(y - y_pred)**2)/(Nw * num_o))


            x_tensors = fit(y, tensors_A, D, 10000, verbose=0, x0=x0_mps,
                            random_init=False, optimize_alpha=-1, comm=None, seed=1, method='adam')

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

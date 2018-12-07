from __future__ import print_function
import unittest
import numpy
import irbasis

from irbasis_util.regression import *

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_lsqr(self):
        numpy.random.seed(100)

        N1, N2 = 1000, 100

        # Small regularization parameter
        alpha = 1e-5

        # Columns of A decay exponentially
        exp_dump = numpy.array([numpy.exp(-20*i/(1.*N2)) for i in range(N2)])
        A = numpy.random.rand(N1*N2) + 1J * numpy.random.rand(N1*N2)
        A = A.reshape((N1, N2))
        A = A[:, :] * exp_dump[None, :]
        y = numpy.random.rand(N1) + 1J * numpy.random.rand(N1)

        # Use precondition to "cancel out" the exponential decay of the columns of A.
        precond = 1/numpy.sqrt(exp_dump**2 + alpha)

        x_svd = ridge_complex(A, y, alpha, solver='svd')
        x_lsqr = ridge_complex(A, y, alpha, solver='lsqr')
        x_lsqr_precond = ridge_complex(A, y, alpha, solver='lsqr', precond=precond)

        # Without preconditioning LSQR nearly fails for a small value of alpha
        # Preconditioning improves accuracy a lot!
        print(numpy.amax(numpy.abs(x_svd-x_lsqr)))
        self.assertTrue(numpy.allclose(x_svd, x_lsqr, atol = 1e-1))
        self.assertTrue(numpy.allclose(x_svd, x_lsqr_precond, atol = 1e-7))

if __name__ == '__main__':
    unittest.main()

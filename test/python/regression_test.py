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

        import time

        t1 = time.time()
        x_svd = ridge_complex(A, y, alpha, solver='svd')
        t2 = time.time()
        x_lsqr = ridge_complex(A, y, alpha, solver='lsqr')
        t3 = time.time()
        x_lsqr_precond = ridge_complex(A, y, alpha, solver='lsqr', precond=precond)

        #x_lsqr_precond_column = ridge_complex(A, y, alpha, solver='lsqr', precond='column')
        #t4 = time.time()

        #print("diff ", numpy.amax(numpy.abs(x_svd - x_lsqr_precond)))
        #print("diff ", numpy.amax(numpy.abs(x_svd - x_lsqr_precond_column)))

        # Without preconditioning LSQR nearly fails for a small value of alpha
        # Preconditioning improves accuracy a lot!
        self.assertTrue(numpy.allclose(x_svd, x_lsqr, atol = 1e-2))
        self.assertTrue(numpy.allclose(x_svd, x_lsqr_precond, atol = 1e-7))
        #self.assertTrue(numpy.allclose(x_svd, x_lsqr_precond_column, atol = 1e-7))

        #print(t2-t1, t3-t2, t4-t3)
        #print("diff ", numpy.amax(numpy.abs(x_svd - x_lsqr)))
        #print("diff ", numpy.amax(numpy.abs(x_svd - x_lsqr_precond)))
        # Change data bit
        #y_diff = delta * (numpy.random.rand(N1) + 1J * numpy.random.rand(N1))
#
        #t1 = time.time()
        #x_svd2 = ridge_complex(A, y + y_diff, alpha, solver='svd')
        #t2 = time.time()
        #x_lsqr2 = ridge_complex(A, y + y_diff, alpha, solver='lsqr', precond=precond)
        #t3 = time.time()
        #x_lsqr2_x0 = ridge_complex(A, y + y_diff, alpha, solver='lsqr', precond=precond, x0=x_lsqr_precond)
        #t4 = time.time()
        #print(t2-t1, t3-t2, t4-t3)

        #x_lsqr2 = ridge_complex(A, y + y_diff, alpha, solver='lsqr')
        #print(numpy.amax(numpy.abs(x_svd2 - x_lsqr2)))
        #print(numpy.amax(numpy.abs(x_svd2 - x_lsqr2_x0)))
        #self.assertTrue(numpy.allclose(x_svd2, x_lsqr2, atol = 1e-7))

if __name__ == '__main__':
    unittest.main()

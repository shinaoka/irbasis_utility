from __future__ import print_function
import unittest
import numpy

from irbasis_util.regression import *
from scipy.sparse.linalg import aslinearoperator, lsqr

N1 = 30
N2 = 10
alpha = 10.0

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_ridge(self):
        numpy.random.seed(100)

        X = numpy.random.randn(N1, N2)
        y = numpy.random.randn(N1)

        reg = Ridge(X)

        x_correct = ridge_svd(X, y, alpha)

        numpy.testing.assert_allclose(x_correct, reg.fit(y, alpha), rtol = 1e-7)

        reg_svd = Ridge(reg.svd)
        numpy.testing.assert_allclose(x_correct, reg_svd.fit(y, alpha), rtol = 1e-7)

        op_X = aslinearoperator(X)
        x_lsqr = lsqr(op_X, y, numpy.sqrt(alpha))
        numpy.testing.assert_allclose(x_correct, x_lsqr[0], rtol = 1e-7)

    def test_ridge_complex(self):
        numpy.random.seed(100)


        X = numpy.random.randn(N1, N2) + 1J * numpy.random.randn(N1, N2)
        y = numpy.random.randn(N1) + 1J * numpy.random.randn(N1)

        reg = RidgeComplex(X)
        x_correct = ridge_complex(X, y, alpha)

        numpy.testing.assert_allclose(x_correct, reg.fit(y, alpha), rtol = 1e-7)

        reg_svd = RidgeComplex(reg.svd)
        numpy.testing.assert_allclose(x_correct, reg_svd.fit(y, alpha), rtol = 1e-7)

        op_X = FloatizedLinearOperator(aslinearoperator(X))
        # y_float: float(2*N1)
        y_float = complex_to_float(y)
        # x_lsqrt: float(N2)
        x_lsqr = float_to_complex(lsqr(op_X, y_float, numpy.sqrt(alpha))[0])
        numpy.testing.assert_allclose(x_correct, x_lsqr, rtol = 1e-7)


if __name__ == '__main__':
    unittest.main()

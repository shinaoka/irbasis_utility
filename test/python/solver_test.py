from __future__ import print_function

import unittest

from irbasis_util.solver import FourPointBasisTransform

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def fit_test(self):
        beta = 1.0
        wmax = 1.0
        transform = FourPointBasisTransform(beta, wmax, scut = 0.9)

        transform.generate_projectors_for_fit()


if __name__ == '__main__':
    unittest.main()

from __future__ import print_function
import unittest
import numpy
import irbasis

from irbasis_util.internal import *

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_o_to_matsubara(self):
        for n in range(-10,10):
            assert o_to_matsubara_idx_f(2*n+1) == n
            assert o_to_matsubara_idx_b(2*n) == n

if __name__ == '__main__':
    unittest.main()

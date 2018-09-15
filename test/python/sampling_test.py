from __future__ import print_function
import unittest
import numpy
import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.internal import *

Lambda = 10000.0
beta = 100.0
wmax = Lambda/beta

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_sampling_point_matsubara(self):
        for stat in ['F', 'B']:
            if stat == 'barB':
                b = Basis(augmented_basis_b(Lambda), beta)
            else:
                b = Basis(irbasis.load(stat, Lambda), beta)

            dim = b.dim()-4
            whichl = dim-1
            sp = sampling_points_matsubara(b, whichl)

            # just for debug
            Unl = b.compute_Unl(sp)[:,2:dim]
            Unl_real = from_complex_to_real_coeff_matrix(Unl)
            U, S, Vh = scipy.linalg.svd(Unl_real, full_matrices=False)
            cond_num = S[0]/S[-1]

            print("cond_num ", cond_num)
            print(sp)
            print(Unl.shape)
            print(U[:,-1])
            print(Vh.T[:,-1])
            for s in S:
                print(s)
            self.assertLessEqual(cond_num, 1000)

if __name__ == '__main__':
    unittest.main()

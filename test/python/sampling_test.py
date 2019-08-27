from __future__ import print_function
import unittest
import numpy
import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.internal import *

Lambda = 1E+7
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

            dim = b.dim - 4
            whichl = dim - 1
            sp = sampling_points_matsubara(b, whichl)
            if stat == 'F':
                assert numpy.all([-s-1 in sp for s in sp])
            elif stat in ['B', 'bB']:
                assert numpy.all([-s in sp for s in sp])

            assert len(sp) == whichl+1

            Unl = b.compute_Unl(sp)[:, :dim]
            Unl_real = from_complex_to_real_coeff_matrix(Unl)
            U, S, Vh = scipy.linalg.svd(Unl_real, full_matrices=False)
            cond_num = S[0] / S[-1]

            print("cond_num ", cond_num)
            self.assertLessEqual(cond_num, 1E+4)

    def test_sampling_point_tau(self):
        for stat in ['F', 'B']:
            if stat == 'barB':
                b = Basis(augmented_basis_b(Lambda), beta)
            else:
                b = Basis(irbasis.load(stat, Lambda), beta)

            dim = b.dim
            whichl = dim - 1
            sp = sampling_points_tau(b, whichl)
            assert len(sp) == whichl+1
            Utaul = numpy.array([b.Ultau(l, tau) for l in range(dim) for tau in sp]).reshape((dim, dim))
            Utaul_real = from_complex_to_real_coeff_matrix(Utaul)
            U, S, Vh = scipy.linalg.svd(Utaul_real, full_matrices=False)
            cond_num = S[0] / S[-1]

            print("cond_num ", cond_num)
            self.assertLessEqual(cond_num, 1E+4)

if __name__ == '__main__':
    unittest.main()

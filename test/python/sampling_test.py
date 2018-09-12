from __future__ import print_function
import unittest
import numpy
import irbasis

from two_point_basis import *
from internal import *

Lambda = 1000.0
beta = 100.0
wmax = Lambda/beta

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_sampling_point_matsubara(self):
        for stat in ['F', 'B', 'barB']:
            if stat == 'barB':
                b = Basis(augmented_basis_b(Lambda), beta)
            else:
                b = Basis(irbasis.load(stat, Lambda), beta)

            whichl = b.dim()-1
            sp = sampling_points_matsubara(b, whichl)

            Unl = b.compute_Unl(sp)
            Unl_real = from_complex_to_real_coeff_matrix(Unl)
            U, S, Vh = scipy.linalg.svd(Unl_real, full_matrices=False)
            cond_num = S[0]/S[-1]

            print("cond_num ", cond_num)
            self.assertLessEqual(cond_num, 1000)

        #l = 1
        #for tau in [0, 0.5, beta]:
            #x = 2 * tau/beta - 1
            #self.assertAlmostEqual(numpy.sqrt(2/beta) * self._b.ulx(l, x), self._B.Ultau(l, tau), delta=1e-10)


if __name__ == '__main__':
    unittest.main()

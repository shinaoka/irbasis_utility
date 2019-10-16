from __future__ import print_function
import unittest
import numpy
import irbasis

from irbasis_util.two_point_basis import *
from irbasis_util.internal import *

from common import load_basis

wmax = 10.0
Lambda = 1E+7
beta = Lambda/wmax

#beta = 100.0
#wmax = Lambda/beta

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_sampling_point_matsubara(self):
        for stat in ['F', 'B']:
            _, b = load_basis(stat, Lambda, beta)

            dim = b.dim - 4
            whichl = dim - 1
            sp = b.sampling_points_matsubara(whichl)
            if 'F' in stat:
                assert numpy.all([-s-1 in sp for s in sp])
            elif 'B' in stat:
                assert numpy.all([-s in sp for s in sp])

            assert len(sp) >= whichl+1

            Unl = b.compute_Unl(sp)[:, :dim]
            U, S, Vh = scipy.linalg.svd(Unl, full_matrices=False)
            cond_num = S[0] / S[-1]

            print("cond_num ", cond_num)
            self.assertLessEqual(cond_num, 1E+4)

    def test_sampling_point_matsubara_vertex(self):
        """
        Check sparse sampling of "self-energy".
        G(iwn) = 1 + 1/(iwn - epsilon), where epsilon = 1
        """
        epsilon = 1
        assert epsilon < wmax

        for stat in ['FV', 'barBV']:
            _, b = load_basis(stat, Lambda, beta)

            dim = b.dim - 4
            whichl = dim - 1
            sp = b.sampling_points_matsubara(whichl)

            Unl = b.compute_Unl(sp)[:, :dim]
            U, S, Vh = scipy.linalg.svd(Unl, full_matrices=False)

            # For bosons, compute approximate inverse matrix without the smallest (nearly zero) singular value
            if stat == 'barBV':
                inv_Unl = numpy.dot(numpy.dot(Vh[:-1,:].conjugate().transpose(), numpy.diag(1/(S[:-1]))), U[:,:-1].conjugate().transpose())
            else:
                inv_Unl = numpy.dot(numpy.dot(Vh.conjugate().transpose(), numpy.diag(1/S)), U.conjugate().transpose())

            # Compute G(iwn) on sparse grid
            iwn = 1J * (2*sp+1) * numpy.pi/beta
            Giwn = 1/(iwn - epsilon) + 1

            # Transformation from sparse grid to IR
            Gl = numpy.dot(inv_Unl, Giwn)

            # Inverse transformation
            Giwn_reconst = numpy.dot(Unl, Gl)

            diff = numpy.amax(numpy.abs(Giwn - Giwn_reconst))
            self.assertLessEqual(diff, 1E-11)


    def test_sampling_point_tau(self):
        for stat in ['F', 'B']:
            _, b = load_basis(stat, Lambda, beta)

            dim = b.dim
            whichl = dim - 1
            sp = b.sampling_points_tau(whichl)
            assert len(sp) >= whichl+1
            Utaul = numpy.array([b.Ultau(l, tau) for l in range(dim) for tau in sp]).reshape((dim, dim))
            U, S, Vh = scipy.linalg.svd(Utaul, full_matrices=False)
            cond_num = S[0] / S[-1]

            print("cond_num ", cond_num)
            self.assertLessEqual(cond_num, 1E+4)

if __name__ == '__main__':
    unittest.main()

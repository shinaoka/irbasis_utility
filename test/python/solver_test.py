from __future__ import print_function

import unittest
import numpy

from irbasis_util.solver import FourPointBasisTransform, _innersum_freq_PH, _innersum_freq_PH_ref, _construct_sampling_points_multiplication
from irbasis_util.gf import LocalGf2CP, _construct_random_LocalGf2CP

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_multiplication_freq(self):
        numpy.random.seed(100)

        beta = 10.0
        wmax = 1.0
        Lambda = beta * wmax
        No = 1

        D_L = 1
        D_R = 1

        transform = FourPointBasisTransform(beta, wmax, scut = 0.999)

        Nw = transform.n_sp_local
        sp_outer = transform._sp_local_F
        inner_w_range = range(0, 1)
        num_w_inner = len(inner_w_range)

        gf_left = _construct_random_LocalGf2CP(beta, Lambda, transform.basis_G2.Nl, No, D_L, vertex=False)
        gf_right = _construct_random_LocalGf2CP(beta, Lambda, transform.basis_G2.Nl, No, D_R, vertex=False)
        print("Nl ", transform.basis_G2.Nl)

        sp_L, sp_R = _construct_sampling_points_multiplication(inner_w_range, sp_outer)
        sp_L = sp_L.reshape((Nw*num_w_inner, 4))
        sp_R = sp_R.reshape((Nw*num_w_inner, 4))

        prj_L = transform.basis_G2.projector_to_matsubara_vec(sp_L, reduced_memory=True)
        prj_R = transform.basis_G2.projector_to_matsubara_vec(sp_R, reduced_memory=True)

        xfreqs_L = gf_left.tensors[0:5]
        xfreqs_R = gf_right.tensors[0:5]


        x0 = _innersum_freq_PH(Nw, num_w_inner, xfreqs_L, xfreqs_R, prj_L, prj_R)

        # Using tensor network
        x0_ref = _innersum_freq_PH_ref(Nw, num_w_inner, xfreqs_L, xfreqs_R, prj_L, prj_R)

        assert numpy.allclose(x0, x0_ref)


if __name__ == '__main__':
    unittest.main()

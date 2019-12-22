from __future__ import print_function
import unittest
import numpy
import h5py

from irbasis_util.gf import LocalGf2CP

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_LocalGf2CP(self):

        Lambda = 10.0
        Nl = 10
        No = 3
        D = 5
        tensors = [numpy.random.randn(D, 16)] + [numpy.random.randn(D, Nl)]*3 + [numpy.random.randn(D, No)]

        for vertex in [True, False]:
            gf = LocalGf2CP(Lambda, Nl, No, D, tensors, vertex)
            assert gf.tensors == tensors
            assert gf.Nl == Nl
            assert gf.vertex == vertex
            assert gf.Lambda == Lambda


    def test_LocalGf2CP_io(self):

        Lambda = 10.0
        Nl = 10
        No = 3
        D = 5
        tensors = [numpy.random.randn(D, 16)] + [numpy.random.randn(D, Nl)]*3 + [numpy.random.randn(D, No)]

        for vertex in [True, False]:
            gf = LocalGf2CP(Lambda, Nl, No, D, tensors, vertex)
            with h5py.File('tmp.h5', 'w') as f:
                gf.save(f, '/somepath')

            with h5py.File('tmp.h5', 'r') as f:
                gf_loaded = LocalGf2CP.load(f, '/somepath')

                assert gf_loaded.Lambda == gf.Lambda
                assert gf_loaded.vertex == gf.vertex
                assert gf_loaded.Nl == gf.Nl
                assert numpy.all([numpy.allclose(gf.tensors[i], gf_loaded.tensors[i]) for i in range(5)])


if __name__ == '__main__':
    unittest.main()

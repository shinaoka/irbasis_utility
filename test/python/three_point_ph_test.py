import unittest
import numpy
import irbasis

from three_point_ph import *

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_matsubara(self):
        boson_freq = 10
        Lambda = 10.0
        beta = 1.0
        phb = ThreePointPHBasis(boson_freq, Lambda, beta)

        #for _lambda in [10000.0]:
            #prefix = "basis_f-mp-Lambda"+str(_lambda)+"_np10"
            #rf = irbasis.basis("../irbasis.h5", prefix)
            #unl = rf.compute_unl([0,1,2])


if __name__ == '__main__':
    unittest.main()

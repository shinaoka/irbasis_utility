from __future__ import print_function

import unittest
import numpy
from irbasis_util.tensor_network import Tensor, TensorNetwork, conj_a_b
from irbasis_util.auto_als import AutoALS

from mpi4py import MPI

class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_auto_als(self):
        """
        Target tensors share no index.
        """
        numpy.random.seed(100)

        comm = MPI.COMM_WORLD

        nproc = comm.Get_size()

        N1, N2 = 3, 5
        y1 = Tensor("y1", (N1,))
        y = TensorNetwork([y1], [(1,)])

        """
        a = Tensor("a", (N1, N2))
        x = Tensor("x", (N2,))
        tilde_y = TensorNetwork([a, x], [(1,2),(2,)])

        auto_als = AutoALS(y, tilde_y, [x], comm=comm)

        values = {}
        if comm.Get_rank() == 0:
            y1_all = [numpy.random.randn(N1) for p in range(nproc)]
            a_all = [numpy.random.randn(N1, N2) for p in range(nproc)]
            values['x'] = numpy.random.randn(N2)
        else:
            y1_all = None
            a_all = None
            values['x'] = None

        values['y1'] = comm.scatter(y1_all, root=0)
        values['a'] = comm.scatter(a_all, root=0)
        values['x'] = comm.bcast(values['x'], root=0)

        print("a", comm.Get_rank(), values['a'])
        print("x", comm.Get_rank(), values['x'])

        auto_als.fit(niter=100, tensors_value=values)
        self.assertLess(numpy.abs(auto_als.squared_error(values)), 1e-10)
        """

if __name__ == '__main__':
    unittest.main()

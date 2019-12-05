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

        comm = MPI.COMM_WORLD

        nproc = comm.Get_size()
        rank = comm.Get_rank()

        numpy.random.seed(100 * rank + 100)

        Nw, Nr, D, Nl = 1, 2, 10, 7
        y1 = Tensor("y1", (Nw,))
        y = TensorNetwork([y1], [(0,)])

        U = Tensor('U', (Nw, Nr, Nl))
        x0 = Tensor('x0', (D, Nr))
        x1 = Tensor('x1', (D, Nl))
        tilde_y = TensorNetwork([U, x0, x1], [(0,10,30),(20,10),(20,30)])

        auto_als = AutoALS(y, tilde_y, [x0, x1], comm=comm, distributed_subscript=0)

        values = {}
        if comm.Get_rank() == 0:
            y1_all = [numpy.random.randn(Nw)+1J*numpy.random.randn(Nw) for p in range(nproc)]
            U_all = [numpy.random.randn(Nw, Nr, Nl)+1J*numpy.random.randn(Nw, Nr, Nl) for p in range(nproc)]
            values['x0'] = numpy.random.randn(D, Nr)+1J*numpy.random.randn(D, Nr)
            values['x1'] = numpy.random.randn(D, Nl)+1J*numpy.random.randn(D, Nl)
        else:
            y1_all = None
            U_all = None
            values['x0'] = None
            values['x1'] = None

        values['y1'] = comm.scatter(y1_all, root=0)
        values['U'] = comm.scatter(U_all, root=0)
        values['x0'] = comm.bcast(values['x0'], root=0)
        values['x1'] = comm.bcast(values['x1'], root=0)

        auto_als.fit(niter=100, tensors_value=values)
        self.assertLess(numpy.abs(auto_als.squared_error(values)), 1e-10)

if __name__ == '__main__':
    unittest.main()

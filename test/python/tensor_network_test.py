from __future__ import print_function

import unittest
import numpy
from irbasis_util.tensor_network import Tensor, TensorNetwork, conj_a_b


class TestMethods(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        
        super(TestMethods, self).__init__(*args, **kwargs)

    def test_tensor(self):
        N1 = 10
        N2 = 20

        # "A"
        t = Tensor("A", (N1, N2))
        self.assertEqual(t.name, "A")
        self.assertEqual(t.shape, (N1,N2))

        # Conj of "A"
        t_conj = Tensor("A", (N1, N2), True)
        self.assertTrue(t_conj.is_conj)

    def test_tensor_network(self):
        N1, N2, N3 = 10, 20, 30
        # A_{01} (B_{12})^*
        A = Tensor("A", (N1, N2))
        B = Tensor("B", (N2, N3), True)
        tn = TensorNetwork([A, B], [(0,1),(1,2)])

        numpy.testing.assert_allclose(tn.external_subscripts, numpy.array([0,2]))

        tn.find_contraction_path(True)

        vals = {}
        vals['A'] = numpy.random.randn(N1, N2) + 1J * numpy.random.randn(N1, N2)
        vals['B'] = numpy.random.randn(N2, N3) + 1J * numpy.random.randn(N2, N3)
        val = tn.evaluate(vals)
        val_ref = numpy.einsum('ij,jk->ik', vals['A'], vals['B'].conjugate())
        numpy.testing.assert_allclose(val, val_ref)

    def test_conj_a_b(self):
        """

        """
        N = 10
        a1 = Tensor("a1", (N, N, N))
        a2 = Tensor("a2", (N, N, N))
        a = TensorNetwork([a1, a2], [(0,2,3), (1,2,3)])

        b1 = Tensor("b1", (N, N))
        b2 = Tensor("b2", (N, N))
        b = TensorNetwork([b1, b2], [(0,2), (1,2)])

        c = conj_a_b(a, b)

        a1v = numpy.random.randn(N, N, N) + 1J*numpy.random.randn(N, N, N)
        a2v = numpy.random.randn(N, N, N) + 1J*numpy.random.randn(N, N, N)
        av = numpy.einsum('ikl,jkl->ij', a1v, a2v)

        b1v = numpy.random.randn(N, N) + 1J*numpy.random.randn(N, N)
        b2v = numpy.random.randn(N, N) + 1J*numpy.random.randn(N, N)
        bv = numpy.einsum('ik,jk->ij', b1v, b2v)

        cv = numpy.einsum('ij,ij', av.conj(), bv)

        c.find_contraction_path()
        cv2 = c.evaluate({"a1" : a1v, "a2" : a2v, "b1" : b1v, "b2" : b2v})
        self.assertAlmostEqual(cv, cv2)


if __name__ == '__main__':
    unittest.main()

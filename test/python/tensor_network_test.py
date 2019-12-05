from __future__ import print_function

import unittest
import numpy
from irbasis_util.tensor_network import Tensor, TensorNetwork, conj_a_b
from irbasis_util.auto_als import AutoALS


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

        self.assertEqual(tn.find_tensor(B), 1)
        self.assertEqual(tn.shape, (N1, N3))

        numpy.testing.assert_allclose(tn.external_subscripts, numpy.array([0,2]))

        tn.find_contraction_path(True)

        vals = {}
        vals['A'] = numpy.random.randn(N1, N2) + 1J * numpy.random.randn(N1, N2)
        vals['B'] = numpy.random.randn(N2, N3) + 1J * numpy.random.randn(N2, N3)
        val = tn.evaluate(vals)
        val_ref = numpy.einsum('ij,jk->ik', vals['A'], vals['B'].conjugate())
        numpy.testing.assert_allclose(val, val_ref)

    def test_tensor_network_order_external(self):
        """
        External indices are in the order in which those of the tensors appear
        """
        N1, N2, N3 = 10, 20, 30
        A = Tensor("A", (N1, N2))
        B = Tensor("B", (N2, N3))
        tn = TensorNetwork([A, B], [(3, 1), (1, 2)])
        self.assertEqual(tn.external_subscripts, (3, 2))

    def test_tensor_network_removal(self):
        N1, N2, N3, N4, N5 = 2, 4, 6, 8, 10
        A = Tensor("A", (N1, N2))
        B = Tensor("B", (N2, N3))
        C = Tensor("C", (N3, N4))
        D = Tensor("D", (N4, N5))
        tn = TensorNetwork([A, B, C, D], [(0,1),(1,2),(2,3),(3,-1)])
        tn2 = tn.remove(B)

        assert tn2.tensors == [A, C, D]
        assert tn2.subscripts == [(0,1),(2,3),(3,-1)]

    def test_tensor_network_removal2(self):
        # Three tensors share one identical subscript.
        N1, N2, N3, N4 = 2, 4, 6, 8
        A = Tensor("A", (N1, N2))
        B = Tensor("B", (N1, N3))
        C = Tensor("C", (N1, N4))
        tn = TensorNetwork([A, B, C], [(0,1),(0,2),(0,3)])
        tn2 = tn.remove(A)

        assert tn2.tensors == [B, C]
        assert tn2.external_subscripts == (0,2,3)

    def test_tensor_network_multi_removal(self):
        N1, N2, N3, N4, N5 = 2, 4, 6, 8, 10
        A = Tensor("A", (N1, N2))
        B = Tensor("B", (N2, N3))
        C = Tensor("C", (N3, N4))
        D = Tensor("D", (N4, N5))
        tn = TensorNetwork([A, B, C, D], [(0,1),(1,2),(2,3),(3,-1)])
        tn2 = tn.remove([D, B])

        assert tn2.tensors == [A, C]
        assert tn2.subscripts == [(0,1),(2,3)]

    def test_conj_a_b(self):
        """

        """
        numpy.random.seed(100)
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

    def test_auto_als(self):
        """
        Target tensors share no index.
        """
        numpy.random.seed(100)

        N1, N2, N3, N4 = 2, 2, 4, 4
        y1 = Tensor("y1", (N1, N2))
        y = TensorNetwork([y1], [(0,1)])

        a = Tensor("a", (N1, N3))
        x = Tensor("x", (N3, N4))
        b = Tensor("b", (N4, N2))
        tilde_y = TensorNetwork([a, x, b], [(0,2),(2,3),(3,1)])

        auto_als = AutoALS(y, tilde_y, [x])

        values = {}
        values['y1'] = numpy.random.randn(N1, N2)
        values['a'] = numpy.random.randn(N1, N3)
        values['x'] = numpy.random.randn(N3, N4)
        values['b'] = numpy.random.randn(N4, N2)

        auto_als.fit(niter=100, tensors_value=values)
        self.assertLess(numpy.abs(auto_als.squared_error(values)), 1e-10)

    def test_auto_als_share_indices(self):
        """
        Target tensors share indices.
        """
        numpy.random.seed(100)

        N1, N2, N3 = 3, 2, 10
        y1 = Tensor("y1", (N1, N2))
        y = TensorNetwork([y1], [(1,2)])

        a = Tensor("a", (N1, N3))
        x = Tensor("x", (N3, N2))
        tilde_y = TensorNetwork([a, x], [(1,3),(3,2)])

        auto_als = AutoALS(y, tilde_y, [x])

        values = {}
        values['y1'] = numpy.random.randn(N1, N2)
        values['a'] = numpy.random.randn(N1, N3)
        values['x'] = numpy.random.randn(N3, N2)

        auto_als.fit(niter=100, tensors_value=values)
        self.assertLess(numpy.abs(auto_als.squared_error(values)), 1e-10)

    def test_auto_als_multiple_target(self):
        """
        Optimize multiple tensors
        """
        numpy.random.seed(100)

        N1, N2, N3, N4, N5 = 3, 2, 2, 10, 10
        y1 = Tensor("y1", (N1, N2, N3))
        y = TensorNetwork([y1], [(1,2,3)])

        a = Tensor("a", (N1, N4))
        x1 = Tensor("x1", (N4, N5))
        x2 = Tensor("x2", (N5, N2, N3))
        tilde_y = TensorNetwork([a, x1, x2], [(1,-1),(-1,-2),(-2,2,3)])

        auto_als = AutoALS(y, tilde_y, [x1, x2])

        values = {}
        values['y1'] = numpy.random.randn(N1, N2, N3)
        values['a'] = numpy.random.randn(N1, N4)
        values['x1'] = numpy.random.randn(N4, N5)
        values['x2'] = numpy.random.randn(N5, N2, N3)

        auto_als.fit(niter=1, tensors_value=values)
        self.assertLess(numpy.abs(auto_als.squared_error(values)), 1e-10)

    def test_auto_als_multi_tensor_networks(self):
        """
        Fit multiple tensor networks by multiple tensor networks
        """
        numpy.random.seed(100)

        N1, N2, N3 = 3, 2, 10
        y1 = TensorNetwork([Tensor("y1", (N1, N2))], [(1,2)])
        y2 = TensorNetwork([Tensor("y2", (N1, N2))], [(1,2)])
        Y = [y1, y2]

        x = Tensor("x", (N3, N2))
        tilde_y1 = TensorNetwork([Tensor("a1", (N1, N3)), x], [(1,3),(3,2)])
        tilde_y2 = TensorNetwork([Tensor("a2", (N1, N3)), x], [(1,3),(3,2)])
        tilde_Y = [tilde_y1, tilde_y2]

        auto_als = AutoALS(Y, tilde_Y, [x])

        values = {}
        values['y1'] = numpy.random.randn(N1, N2)
        values['y2'] = numpy.random.randn(N1, N2)
        values['a1'] = numpy.random.randn(N1, N3)
        values['a2'] = numpy.random.randn(N1, N3)
        values['x'] = numpy.random.randn(N3, N2)

        auto_als.fit(niter=100, tensors_value=values)
        self.assertLess(numpy.abs(auto_als.squared_error(values)), 1e-10)

    def test_auto_als_regularization(self):
        """
        Ridge regression
        x^* = argmin_x  |y-a*x|^2 + lambda * |x|^2
        x^* = a * y/(lambda + a^2)
        """
        numpy.random.seed(100)

        N = 1
        y1 = Tensor("y1", (N,))
        y = TensorNetwork([y1], [(1,)])

        a = Tensor("a", (N, N))
        x = Tensor("x", (N,))
        tilde_y = TensorNetwork([a, x], [(1,2),(2,)])

        reg_L2 = 0.1
        auto_als = AutoALS(y, tilde_y, [x], reg_L2=reg_L2)

        values = {}
        values['y1'] = numpy.array([1.0])
        values['a'] = numpy.array([[1.0]])
        values['x'] = numpy.random.randn(N)

        auto_als.fit(niter=100, tensors_value=values)

        opt_x = values['x'][0]
        self.assertAlmostEqual(opt_x, 1/(reg_L2 + 1))


if __name__ == '__main__':
    unittest.main()

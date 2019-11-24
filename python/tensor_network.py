from __future__ import print_function

import numpy

def _to_alphabet(i):
    """
    Convert an integer (from 0 to 51) to an _to_alphabet
    """
    if i >= 0 and i <= 25:
        return chr(97 + i)
    elif i >= 26 and i <= 51:
        return chr(65 + i - 26)
    else:
        raise RuntimeError("Cannot convert {} to alphabet!".format(i))


class Tensor(object):
    def __init__(self, name, shape, is_conj=False):
        """
        Tensor

        :param name: str
            Name of tensor
        :param shape: tuple of integers
            Shape of tensor
        :param is_conj: bool
            Complex conjugate
        """

        self._name = name
        self._shape = shape
        self._is_conj = is_conj

    def conjugate(self):
        """
        Create a complex conjugate
        """
        return Tensor(self._name, self._shape, not self._conj)

    @property
    def name(self):
        """
        Name
        """
        return self._name

    @property
    def shape(self):
        """
        Shape
        """
        return self._shape

    @property
    def is_conj(self):
        """
        Is complex conjugate
        """
        return self._is_conj


class TensorNetwork(object):
    def __init__(self, tensors, subscripts):
        """
        Network of tensors

        :param tensors: list of Tensors
            Tensors
        :param subscripts: list of tuples of integers
            Specifies the subscripts for einstein summation
            Each tuple denotes the subscript of a tensor.
        """

        if len(tensors) != len(subscripts):
            raise RuntimeError("tensors and subscripts must contain the same number of elements.")

        for tensor, s in zip(tensors, subscripts):
            if len(tensor.shape) != len(s):
                raise RuntimeError("Dimension mismatch")

        unique_subscripts = numpy.unique(sum(subscripts, ()))
        num_subscripts = len(unique_subscripts)
        if numpy.amax(unique_subscripts) >= num_subscripts:
            raise RuntimeError("Subscripts must be smaller than the number of unique subscripts.")
        if numpy.amin(unique_subscripts) < 0:
            raise RuntimeError("Subscripts must be 0 or positive.")

        # Mark outer subscript
        #  An outer subscript appears only once.
        counter_subscripts = numpy.zeros(num_subscripts, dtype=int)
        for s in sum(subscripts, ()):
            counter_subscripts[s] += 1
        self._is_outer_subscript = numpy.array(counter_subscripts == 1)
        self._outer_subscripts = unique_subscripts[self._is_outer_subscript]

        self._tensors = tensors
        self._num_tensors = len(tensors)
        self._subscripts = subscripts
        self._contraction_path = None

        # Create str ver. of subscripts
        ints_to_alphabets = lambda integers: ''.join(map(_to_alphabet, integers))
        left_str = ','.join(map(ints_to_alphabets, self._subscripts))
        right_str = ints_to_alphabets(self._outer_subscripts)
        self._str_sub = left_str + '->' + right_str

    @property
    def tensors(self):
        return self._tensors

    @property
    def num_tensors(self):
        return self._num_tensors

    @property
    def subscripts(self):
        return self._subscripts

    @property
    def outer_subscripts(self):
        return self._outer_subscripts

    def find_contraction_path(self, verbose=False):
        """
        Find a path for contracting tensors
        The subscripts of the contracted tensor are ordered in ascending order from left to right.
        """

        dummy_arrays = [numpy.empty(t.shape) for t in self._tensors]
        print(self._str_sub)
        print(len(dummy_arrays))
        self._contraction_path, string_repr = numpy.einsum_path(self._str_sub, *dummy_arrays, optimize=('optimal', 1E+18))

        if verbose:
            print(string_repr)


    def evaluate(self, values_of_tensors):
        """
        Evaluate the tensor network

        :param values_of_tensors: dict
            key = name of a tensor, value = value of the tensor

        :return: ndarray
            Results of contraction
        """
        if self._contraction_path is None:
            raise RuntimeError("Call find_contraction first!")

        arrays = []
        for t in self.tensors:
            if t.is_conj:
                arrays.append(values_of_tensors[t.name].conjugate())
            else:
                arrays.append(values_of_tensors[t.name])

        return numpy.einsum(self._str_sub, *arrays, optimize=self._contraction_path)

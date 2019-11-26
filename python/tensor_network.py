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

    def __str__(self):
        if self._is_conj:
            return "conj({}){}".format(self._name, self._shape)
        else:
            return "{}{}".format(self._name, self._shape)

    def __repr__(self):
        return self.__str__()

    def conjugate(self):
        """
        Create a complex conjugate
        """
        return Tensor(self._name, self._shape, not self._is_conj)

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

    def __eq__(self, other):
        if isinstance(other, Tensor):
            return self.__dict__ == other.__dict__
        return False



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

        #unique_subscripts = numpy.unique(sum(subscripts, ()))
        unique_subscripts = _unique_order_preserved(sum(subscripts, ()))

        # Mark external subscript
        #  An external subscript appears only once.
        all_subscripts = numpy.array(sum(subscripts, ()))
        self._is_external_subscript = []
        for s in unique_subscripts:
            self._is_external_subscript.append(numpy.count_nonzero(all_subscripts==s) == 1)

        self._external_subscripts = tuple(unique_subscripts[self._is_external_subscript])

        self._tensors = tensors
        self._unique_subscripts = tuple(unique_subscripts)
        self._num_tensors = len(tensors)
        self._subscripts = subscripts
        self._contraction_path = None

        # Mapping from subscripts to alphabets
        self._map_subscript_char = {}
        for idx, subscript in enumerate(self._unique_subscripts):
            self._map_subscript_char[subscript] = _to_alphabet(idx)

        # Create str ver. of subscripts
        f = lambda x : self._map_subscript_char[x]
        ints_to_alphabets = lambda x: ''.join(map(f, x))
        left_str = ','.join(map(ints_to_alphabets, self._subscripts))
        right_str = ints_to_alphabets(self._external_subscripts)
        self._str_sub = left_str + '->' + right_str

        # Shape
        self._extern_subs_order = {s : i for i, s in enumerate(self._external_subscripts)}
        shape = numpy.empty((len(self._external_subscripts),), dtype=int)
        for tensor, sub in zip(self._tensors, self._subscripts):
            for dim, s in zip(tensor.shape, sub):
                if s in self._external_subscripts:
                    shape[self._extern_subs_order[s]] = dim
        self._shape = tuple(shape)

    def __str__(self):
        str1 =  ', '.join([t.__str__() for t in self._tensors])
        str2 =  ', '.join([s.__str__() for s in self._subscripts])

        return str1 + '\n' + str2

    def __repr__(self):
        return self.__str__()

    @property
    def tensors(self):
        return self._tensors

    @property
    def shape(self):
        return self._shape

    @property
    def num_tensors(self):
        return self._num_tensors

    @property
    def subscripts(self):
        return self._subscripts

    @property
    def external_subscripts(self):
        return self._external_subscripts

    @property
    def unique_subscripts(self):
        return self._unique_subscripts

    def find_contraction_path(self, verbose=False):
        """
        Find a path for contracting tensors
        The subscripts of the contracted tensor are ordered in ascending order from left to right.
        """

        dummy_arrays = [numpy.empty(t.shape) for t in self._tensors]
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
            if t.shape != values_of_tensors[t.name].shape:
                raise RuntimeError("Dimension mismatch between tensors and values_of_tensors!")
            if t.is_conj:
                arrays.append(values_of_tensors[t.name].conjugate())
            else:
                arrays.append(values_of_tensors[t.name])

        r = numpy.einsum(self._str_sub, *arrays, optimize=self._contraction_path)
        return r

    def find_tensor(self, tensor):
        """
        Find the given tensor
        :param tensor: Tensor
          Tensor
        :return:
          Index of the first tensor that equals to the given tensor
        TODO: Does not work if there are multiple tensors match the given tensor
        """

        assert isinstance(tensor, Tensor)
        return self._tensors.index(tensor)

    def tensor_subscripts(self, tensor):
        """
        Return the subscripts for a given tensor
        TODO: Does not work if there are multiple tensors match the given tensor
        """

        return self._subscripts[self._tensors.index(tensor)]

    def remove(self, tensor_to_be_removed):
        """
        Make a new tensor network object by removing a tensor

        :param tensors_to_be_removed: Tensor or list of tensors
           Tensor(s) to be removed.
        :return:
           New tensor network object
        """

        if isinstance(tensor_to_be_removed, list):
            if len(tensor_to_be_removed) > 1:
                return self.remove(tensor_to_be_removed[0]).remove(tensor_to_be_removed[1:])
            else:
                return self.remove(tensor_to_be_removed[0])

        new_tensors = []
        new_subscripts = []
        for t, s in zip(self.tensors, self.subscripts):
            if t == tensor_to_be_removed:
                continue
            new_tensors.append(t)
            new_subscripts.append(s)
        return TensorNetwork(new_tensors, new_subscripts)

def conj_a_b(a, b):
    """
    Contract a tensor network representation of <a|b>
    a and b must share the same subscripts for the external indices.
    In the resulting tensor network, the contracted indices have the smallest subscripts.

    :param a: TensorNetwork
    :param b: TensorNetwork
    """

    if not numpy.all(a.external_subscripts == b.external_subscripts):
        raise RuntimeError("Outer indices do not match!")

    num_ext_idx = len(a.external_subscripts)
    num_idx_a = len(a.unique_subscripts)
    num_idx_b = len(b.unique_subscripts)

    # Create map from old subscripts to new subscripts
    new_subscripts_a = {}
    new_subscripts_b = {}

    # External indices
    next_new_subscript = 0
    for subscript in a.unique_subscripts:
        if subscript in a.external_subscripts:
            new_subscripts_a[subscript] = next_new_subscript
            next_new_subscript += 1

    next_new_subscript = 0
    for subscript in b.unique_subscripts:
        if subscript in b.external_subscripts:
            new_subscripts_b[subscript] = next_new_subscript
            next_new_subscript += 1

    # Internal indices in a
    for subscript in a.unique_subscripts:
        if not subscript in a.external_subscripts:
            new_subscripts_a[subscript] = next_new_subscript
            next_new_subscript += 1

    # Internal indices in b
    for subscript in b.unique_subscripts:
        if not subscript in b.external_subscripts:
            new_subscripts_b[subscript] = next_new_subscript
            next_new_subscript += 1

    num_new_subscripts = num_idx_a + num_idx_b - num_ext_idx
    assert next_new_subscript == num_new_subscripts

    new_tensors = [t.conjugate() for t in a.tensors]
    new_tensors.extend(b.tensors)

    new_subscripts = []
    f = lambda x: new_subscripts_a[x]
    new_subscripts.extend([tuple(map(f, tp)) for tp in a.subscripts])
    f = lambda x: new_subscripts_b[x]
    new_subscripts.extend([tuple(map(f, tp)) for tp in b.subscripts])

    return TensorNetwork(new_tensors, new_subscripts)


def _unique_order_preserved(x):
    u, ind = numpy.unique(numpy.asarray(x), return_index=True)
    return u[numpy.argsort(ind)]

def from_int_to_char_subscripts(subscripts):
    unique_subscripts = _unique_order_preserved(sum([list(t) for t in subscripts], []))
    mapping = {unique_subscripts[i] : _to_alphabet(i) for i in range(len(unique_subscripts))}
    return [map(lambda x: mapping[x], s) for s in subscripts]

def differenciate(tensor_network, tensors):
    """
    Differenciate a tensor network w.r.t tensor(s).

    :param tensor_network: TensorNetwork
        Tensor network to be differentiated
    :param tensors: Tensor
        Tensors
    :return: TensorNetwork
        Result of differentiation
    """
    if isinstance(tensors, Tensor):
        tensors = [tensors]

    for target_t in tensors:
        num_match = len([t for t in tensor_network.tensors if t==target_t])
        if num_match == 0:
           raise RuntimeError("Tensor not found")
        if num_match > 1:
            raise RuntimeError("Unsupported: Multiple tensors match!")

    for i1, t1 in enumerate(tensors):
        for i2, t2 in enumerate(tensors):
            if i1 >= i2:
                continue
            if t1 == t2:
                raise RuntimeError("Unsupported: tensors must not contain multiple equivalent tensors.")

    return tensor_network.remove(tensors)

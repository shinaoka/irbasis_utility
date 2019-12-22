from __future__ import print_function

import h5py
import numpy


class LocalGf2CP(object):
    """
    Local Four-point object in CP form
    """
    def __init__(self, Lambda, Nl, No, D, tensors, vertex):
        """
        Initialize a local Green's function object

        :param Lambda: float
            Lambda for IR
        :param Nl: int
            Linear dimension of IR
        :param No: int
            Size of spin-orbital tensor
        :param D: int
            Bond dimension
        :param tensors: list of ndarray (complex or float)
            CP components.
            If tensors is None, all tensors are initialized to zero (complex).
        :param vertex: bool
            Vertex or not
        """

        self._Lambda = float(Lambda)
        self._D, self._Nl, self._No = D, Nl, No
        self._vertex = vertex
        self._ntensors = 5

        shapes = [(self._D, 16)] + [(self._D, self._Nl)] * 3 + [(self._D, No)]
        if tensors is None:
            self._tensors = [numpy.zeros(shape, dtype=complex) for shape in shapes]
        else:
            assert len(tensors) == self._ntensors
            self._tensors = tensors
            for i in range(self._ntensors):
                assert tensors[i].shape == shapes[i]

    @property
    def tensors(self):
        return self._tensors

    @property
    def D(self):
        return self._D

    @property
    def Nl(self):
        return self._Nl

    @property
    def No(self):
        return self._No

    @property
    def Lambda(self):
        return self._Lambda

    @property
    def vertex(self):
        return self._vertex

    def __str__(self):
        print('Lambda=')

    def save(self, f, path):
        """
        Save data into a HDF file

        :param f: h5py file object
        :param path: str
            Path
        """

        if path in f:
            del f[path]

        f[path + '/Lambda'] = self.Lambda
        f[path + '/D'] = self.D
        f[path + '/Nl'] = self.Nl
        f[path + '/No'] = self.No
        f[path + '/vertex'] = self.vertex
        for i, x in enumerate(self.tensors):
            f[path + '/x' + str(i)] = x

    def bcast(self, root=0):
        for it in ['_Lambda', '_vertex', '_Nl', '_D', 'tensors']:
            setattr(self, it, self._comm.bcast(getattr(self, it), root))

    @classmethod
    def load(cls, f, path):
        """
        Load data from a HDF file

        :param f: h5py file object
        :param path: str
            Path
        return:
            A LocalGf2 object
        """

        tensors = [f[path + '/x' + str(i)][()] for i in range(5)]
        Lambda =  f[path + '/Lambda'][()]
        vertex =  f[path + '/vertex'][()]
        Nl =  f[path + '/Nl'][()]
        No =  f[path + '/No'][()]
        D =  f[path + '/D'][()]

        return LocalGf2CP(Lambda, Nl, No, D, tensors, vertex)

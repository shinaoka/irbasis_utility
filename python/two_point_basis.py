from __future__ import print_function
import numpy
import scipy
import irbasis

from .internal import *

def _compute_Tnl_norm_legendre(n, l):
    if l==0:
        return 1 if n == 0 else 0
    elif l==1:
        if n == 0:
            return 0
        else:
            return numpy.sqrt(3.) / (1J*numpy.pi*n)
    else:
        raise RuntimeError("l > 1")

class augmented_basis_b(object):
    """
    Augmented basis for boson (defined in x, y domains)
    Defined by Eq. (14) of Phys. Rev. B 97, 205111 (2018).
    """
    def __init__(self, basis_xy):
        """
        Contructs an object representing augmented basis for boson

        Parameters
        ----------
        basis_xy: object of irbasis (boson)
        """
        check_type(basis_xy, irbasis.basis)
        check_value(basis_xy.statistics, 'B')

        self._bb = basis_xy
        self._dim = self._bb.dim() + 2

    @property
    def Lambda(self):
        return self._bb.Lambda

    @property
    def basis_b(self):
        return self._bb

    @property
    def statistics(self):
        return 'B'

    def ulx(self, l, x):
        if l == 0:
            return numpy.sqrt(0.5)
        elif l == 1:
            return numpy.sqrt(1.5) * x
        else:
            return self._bb.ulx(l-2, x)

    def ulx_all_l(self, x):
        r = numpy.zeros((self._dim))
        r[0] = numpy.sqrt(0.5)
        r[1] = numpy.sqrt(1.5) * x
        r[2:] = self._bb.ulx_all_l(x)
        return r

    def sl(self, l):
        if l == 0 or l == 1:
            return self._bb.sl(0)
        else:
            return self._bb.sl(l-2)

    def compute_unl(self, nvec):
        num_n = len(nvec)
        unl = numpy.zeros((num_n, self._dim), dtype=complex)
        unl[:, 0:2] = numpy.array([_compute_Tnl_norm_legendre(n, l) for n in nvec for l in range(2)]).reshape(num_n, 2)
        unl[:, 2:] = self._bb.compute_unl(nvec)
        return unl

    def dim(self):
        return self._dim

    def sampling_points_x(self, max_l):
        return irbasis.sampling_points_x(self._bb, max_l+2)

    def sampling_points_matsubara(self, max_l):
        return irbasis.sampling_points_matsubara(self._bb, max_l+2)


class vertex_basis(object):
    """
     Basis for two-point Green's function in terms of tau and omega.
    The first basis function is a constant 1 in Matsubara frequency domain.
    """
    def __init__(self, b):
        """

        Parameters
        ----------
        b: object of irbasis
        """
        check_type(b, [irbasis.basis, augmented_basis_b])

        self._b = b
        self._dim = self._b.dim() + 1

    @property
    def Lambda(self):
        return self._b.Lambda

    @property
    def statistics(self):
        return self._b.statistics

    def ulx(self, l, x):
        if l == 0:
            raise RuntimeError("ulx is not defined for l=0.")
        else:
            return self._b.ulx(l-1, x)

    def sl(self, l):
        if l == 0:
            return self._b.sl(0)
        else:
            return self._b.sl(l-1)

    def compute_unl(self, nvec):
        num_n = len(nvec)
        unl = numpy.zeros((num_n, self._dim), dtype=complex)
        unl[:, 0] = 1
        unl[:, 1:] = self._b.compute_unl(nvec)
        return unl

    def dim(self):
        return self._dim

    def sampling_points_x(self, max_l):
        return irbasis.sampling_points_x(self._b, max_l+1)

    def sampling_points_matsubara(self, max_l):
        return irbasis.sampling_points_matsubara(self._b, max_l+1)

class Basis(object):
    """
    Basis for two-point Green's function in terms of tau and omega
    """
    def __init__(self, b, beta, cutoff=1e-15):
        check_type(b, [irbasis.basis, augmented_basis_b])
        check_type(beta, [float])

        self._Lambda = b.Lambda
        self._wmax = self._Lambda / beta
        self._stat = b.statistics
        self._b = b
        self._beta = beta
        self._scale = numpy.sqrt(2 / beta)
        self._scale2 = numpy.sqrt(1 / self._wmax)
        self._Unl_cache = {}

        self._dim = numpy.sum([self._b.sl(l) / self._b.sl(0) > cutoff for l in range(self._b.dim())])

        if self._stat == 'F':
            self._sl_const = numpy.sqrt(0.5 * beta * self._wmax)
        elif self._stat == 'B':
            self._sl_const = numpy.sqrt(0.5 * beta * self._wmax**3)
        else:
            raise RuntimeError("Unknown statistics " + self._stat)

    @property
    def beta(self):
        return self._beta

    @property
    def statistics(self):
        return self._b.statistics

    @property
    def basis_xy(self):
        return self._b

    @property
    def wmax(self):
        return self._wmax

    @property
    def dim(self):
        return self._dim

    @property
    def max_l(self):
        return self._b.dim() - 1

    def Sl(self, l):
        return self._sl_const * self._b.sl(l)

    def Ultau(self, l, tau):
        check_type(l, [int])
        check_type(tau, [float])
        t, s = my_mod(tau, self._beta)
        x = 2 * t / self._beta - 1
        sign = s if self._stat == 'F' else 1
        return sign * self._scale * self._b.ulx(l, x)

    def Ultau_all_l(self, tau):
        check_type(tau, [float])
        t, s = my_mod(tau, self._beta)
        x = 2 * t / self._beta - 1
        sign = s if self._stat == 'F' else 1
        return sign * self._scale * self._b.ulx_all_l(x)

    def Vlomega(self, l, omega):
        check_type(l, [int])
        check_type(omega, [float])
        return self._scale2 * self._b.vly(l, omega / self._wmax)

    def _precompute_Unl(self, nvec):
        nvec_compt = numpy.unique([n for n in nvec if not n in self._Unl_cache])
        if len(nvec_compt) == 0:
            return
        unl = self._b.compute_unl(nvec_compt)
        for i in range(len(nvec_compt)):
            self._Unl_cache[nvec_compt[i]] = numpy.sqrt(self._beta) * unl[i,:self._dim]

    def compute_Unl(self, nvec, auto_compute=True):
        # shortcut
        if len(nvec)==1 and nvec[0] in self._Unl_cache:
            return self._Unl_cache[nvec[0]].reshape((1, -1))

        if auto_compute:
            self._precompute_Unl(nvec)
        num_n = len(nvec)
        Unl = numpy.empty((num_n, self.dim), dtype=complex)
        for i in range(num_n):
            Unl[i, :] = self._Unl_cache[nvec[i]]
        return Unl

    def sampling_points_tau(self, max_l):
        """
        Sparse sampling points in tau

        :param max_l: int
           Max index l
        :return: 1D array of float
        """
        return 0.5 * self._beta * (self._b.sampling_points_x(max_l) + 1)

    def sampling_points_matsubara(self, max_l):
        return self._b.sampling_points_matsubara(max_l)


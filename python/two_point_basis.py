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
    Augmented basis for boson (in terms of x, y)
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
        return 'barB'

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
        elif self._stat == 'B' or self._stat == 'barB':
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

def sampling_points_leggauss(basis_beta, whichl, deg):
    """
    Computes the sample points and weights for composite Gauss-Legendre quadrature
    according to the zeros of the given basis function

    Parameters
    ----------
    basis_beta : Basis
        Basis object
    whichl: int
        Index of reference basis function "l"
    deg: int
        Number of sample points and weights between neighboring zeros

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points (in tau)
    y : ndarray
        1-D ndarray containing the weights.

    """

    check_type(basis_beta, [Basis])
    ulx = lambda x: basis_beta.basis_xy.ulx(whichl, x)
    section_edges = numpy.hstack((-1., find_zeros(ulx), 1.))

    x, y = composite_leggauss(deg, section_edges)

    return tau_for_x(x, basis_beta.beta), .5 * basis_beta.beta * y

def _funique(x, tol=2e-16):
    """Removes duplicates from an 1D array within tolerance"""
    x = numpy.sort(x)
    unique = numpy.ediff1d(x, to_end=2*tol) > tol
    x = x[unique]
    return x

def _find_roots(ulx_data, xoffset, tol=2e-16):
    """Find all roots in the piecewise polynomial representation"""
    nsec, npoly = ulx_data.shape
    if xoffset.shape != (nsec+1,):
        raise ValueError("Invalid section edges shape")

    xsegm = xoffset[1:] - xoffset[:-1]
    roots = []
    for i in range(nsec):
        x0s = numpy.roots(ulx_data[i, ::-1])
        x0s = [(x0 + xoffset[i]).real for x0 in x0s
               if -tol < x0 < xsegm[i]+tol and numpy.abs(x0.imag) < tol]
        roots += x0s

    roots = numpy.asarray(roots)
    roots = numpy.hstack((-roots[::-1], roots))
    roots = _funique(roots, tol)
    return roots

def sampling_points_x(basis_xy, whichl):
    xroots =  _find_roots(basis_xy._ulx_data[whichl], basis_xy._ulx_section_edges, 1e-8)
    xroots_ex = numpy.hstack((-1.0, xroots, 1.0))
    return 0.5 * (xroots_ex[:-1] + xroots_ex[1:])

def sampling_points_tau(basis_beta, whichl):
    return 0.5 * basis_beta.beta * (sampling_points_x(basis_beta.basis_xy, whichl) + 1)

def sampling_points_matsubara(basis_beta, whichl):
    """
    Computes "optimal" sampling points in Matsubara domain for given basis

    Parameters
    ----------
    basis_beta : Basis
        Basis object
    whichl: int
        Index of reference basis function "l"

    Returns
    -------
    sampling_points: list of int
        List of sampling points in Matsubara domain

    """
    check_type(basis_beta, Basis)
    basis = basis_beta.basis_xy
    stat = basis.statistics
    beta = basis_beta.beta

    assert stat == 'F' or stat == 'B' or stat == 'barB'

    whichl_t = whichl
    if stat == 'barB':
        whichl_t = whichl + 2

    if whichl_t > basis_beta.max_l:
        raise RuntimeError("Too large whichl")

    x1 = numpy.arange(1000)
    x2 = numpy.array(numpy.exp(numpy.linspace(numpy.log(1000), numpy.log(1E+8), 1000)), dtype=int)
    x = numpy.unique(numpy.hstack((x1, x2)))
    unl = basis.compute_unl(x)

    shift = 0 if stat=='F' else 1
    if ((whichl_t + shift) % 2 == 0):
        y = numpy.sqrt(beta) * unl[:, whichl_t].imag
    else:
        y = numpy.sqrt(beta) * unl[:, whichl_t].real

    sign_change = [(x[i], x[i+1], y[i+1], i) for i in range(len(x) - 1) if y[i] * y[i+1] <= 0]
    zeros = numpy.array([0.5 * (p[0] + p[1]) for p in sign_change])
    zero_mids = numpy.array([0.5*(zeros[i] + zeros[i+1]) for i in range(len(zeros) - 1)], dtype=int)

    # Find the point where abs(y) takes a maximum value after the last sign change.
    one_after_last_sign_change = sign_change[-1][3]
    last_sampling_point = x[one_after_last_sign_change + numpy.argmax(numpy.abs(y[one_after_last_sign_change:]))]
    sp_half = numpy.hstack(([0], zero_mids, [last_sampling_point]))
    if stat == 'F':
        r = numpy.sort(numpy.hstack((sp_half, -sp_half - 1)))
    else:
        r = numpy.unique(numpy.hstack((sp_half, -sp_half)))
    return r

def Gl_pole(B, pole):
    assert isinstance(B, Basis)
    Sl = numpy.array([B.Sl(l) for l in range(B.dim)])
    if B.statistics == 'F':
        Vlpole = numpy.array([B.Vlomega(l, pole) for l in range(B.dim)])
        return -Sl * Vlpole
    elif B.statistics == 'B':
        assert pole != 0
        Vlpole = numpy.array([B.Vlomega(l, pole) for l in range(B.dim)])
        return -Sl * Vlpole / pole
    elif B.statistics == 'barB':
        assert pole != 0
        Vlpole = numpy.zeros((B.dim))
        Vlpole[2:] = numpy.sqrt(1 / B.wmax) * numpy.array([B.basis_xy.basis_b.vly(l, pole / B.wmax)
                                                               for l in range(B.dim-2)])
        return -Sl * Vlpole / pole

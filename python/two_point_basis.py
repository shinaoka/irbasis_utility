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


def _find_roots(ulx):
    """Find all roots in (-1, 1) using double exponential mesh + bisection"""
    Nx = 10000
    eps = 1e-14
    tvec = numpy.linspace(-3, 3, Nx)  # 3 is a very safe option.
    xvec = numpy.tanh(0.5 * numpy.pi * numpy.sinh(tvec))

    zeros = []
    for i in range(Nx - 1):
        if ulx(xvec[i]) * ulx(xvec[i + 1]) < 0:
            a = xvec[i + 1]
            b = xvec[i]
            u_a = ulx(a)
            u_b = ulx(b)
            while a - b > eps:
                half_point = 0.5 * (a + b)
                if ulx(half_point) * u_a > 0:
                    a = half_point
                else:
                    b = half_point
            zeros.append(0.5 * (a + b))
    return numpy.array(zeros)


#def _find_roots(ulx_data, xoffset, tol=2e-16):
    #"""Find all roots in the piecewise polynomial representation"""
    #nsec, npoly = ulx_data.shape
    #if xoffset.shape != (nsec+1,):
        #raise ValueError("Invalid section edges shape")
#
    #xsegm = xoffset[1:] - xoffset[:-1]
    #roots = []
    #for i in range(nsec):
        #x0s = numpy.roots(ulx_data[i, ::-1])
        #x0s = [(x0 + xoffset[i]).real for x0 in x0s
               #if -tol < x0 < xsegm[i]+tol and numpy.abs(x0.imag) < tol]
        #roots += x0s
#
    #roots = numpy.asarray(roots)
    #roots = numpy.hstack((-roots[::-1], roots))
    #roots = _funique(roots, tol)
    #return roots

def sampling_points_x(basis_xy, whichl):
    xroots =  _find_roots(lambda x: basis_xy.ulx(whichl, x))
    xroots_ex = numpy.hstack((-1.0, xroots, 1.0))
    return 0.5 * (xroots_ex[:-1] + xroots_ex[1:])

def sampling_points_tau(basis_beta, whichl):
    return 0.5 * basis_beta.beta * (sampling_points_x(basis_beta.basis_xy, whichl) + 1)

def _start_guesses(n=1000):
    "Construct points on a logarithmically extended linear interval"
    x1 = numpy.arange(n)
    x2 = numpy.array(numpy.exp(numpy.linspace(numpy.log(n), numpy.log(1E+8), n)), dtype=int)
    x = numpy.unique(numpy.hstack((x1,x2)))
    return x

def _get_unl_real(basis_xy, x):
    "Return highest-order basis function on the Matsubara axis"
    unl = basis_xy.compute_unl(x)
    result = numpy.zeros(unl.shape, float)

    # Purely real functions
    real_loc = 1 if basis_xy.statistics == 'F' else 0
    assert numpy.allclose(unl[:, real_loc::2].imag, 0)
    result[:, real_loc::2] = unl[:, real_loc::2].real

    # Purely imaginary functions
    imag_loc = 1 - real_loc
    assert numpy.allclose(unl[:, imag_loc::2].real, 0)
    result[:, imag_loc::2] = unl[:, imag_loc::2].imag
    return result

def _sampling_points(fn):
    "Given a discretized 1D function, return the location of the extrema"
    fn = numpy.asarray(fn)
    fn_abs = numpy.abs(fn)
    sign_flip = fn[1:] * fn[:-1] < 0
    sign_flip_bounds = numpy.hstack((0, sign_flip.nonzero()[0] + 1, fn.size))
    points = []
    for segment in map(slice, sign_flip_bounds[:-1], sign_flip_bounds[1:]):
        points.append(fn_abs[segment].argmax() + segment.start)
    return numpy.asarray(points)

def _full_interval(sample, stat):
    if stat == 'F':
        return numpy.hstack((-sample[::-1]-1, sample))
    else:
        # If we have a bosonic basis and even order (odd maximum), we have a
        # root at zero. We have to artifically add that zero back, otherwise
        # the condition number will blow up.
        if sample[0] == 0:
            sample = sample[1:]
        return numpy.hstack((-sample[::-1], 0, sample))

def get_mats_sampling(basis_xy, lmax=None):
    "Generate Matsubara sampling points from extrema of basis functions"
    if lmax is None: lmax = basis_xy.dim()-1

    x = _start_guesses()
    y = _get_unl_real(basis_xy, x)[:,lmax]
    x_idx = _sampling_points(y)

    sample = x[x_idx]
    return _full_interval(sample, basis_xy.statistics)

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

    return get_mats_sampling(basis, whichl_t)

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

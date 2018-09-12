from __future__ import print_function
import numpy
import scipy
import irbasis

def _compute_Tnl_norm_legendre(n, l):
    if l==0:
        return 1 if n == 0 else 0
    elif l==1:
        if n == 0:
            return 0
        else:
            return numpy.sqrt(3.)* numpy.exp(1J*numpy.pi*n) * (numpy.cos(n*numpy.pi))/(1J*numpy.pi*n)
    else:
        raise RuntimeError("l > 1")

class augmented_basis_b(object):
    def __init__(self, Lambda):
        self._bb = irbasis.load('B', Lambda)
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

    def ulx_all_l(self, x):
        r = numpy.zeros((dim_))
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
    def __init__(self, b, beta):
        self._Lambda = b.Lambda
        self._wmax = self._Lambda/beta
        self._stat = b.statistics
        self._b = b
        self._beta = beta
        self._scale = numpy.sqrt(2/beta)
        self._scale2 = numpy.sqrt(1/self._wmax)
        self._Unl_cache = {}

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

    def dim(self):
        return self._b.dim()

    def Sl(self, l):
        return self._sl_const * self._b.sl(l)

    def Ultau(self, l, tau):
        return self._scale * self._b.ulx(l, 2*tau/self._beta-1)

    def Ultau_all_l(self, tau):
        t, s = _my_mod(tau, self._beta) 
        x = 2 * t / self._beta - 1
        sign = s if self._statistics == 'F' else 1
        return sign * self._scale * self._b.ulx_all_l(x)

    def Vlomega(self, l, omega):
        return self._scale2 * self._b.vly(l, omega/self._wmax)

    def _precompute_Unl(self, nvec):
        nvec_compt = numpy.unique([n for n in nvec if not n in self._Unl_cache])
        if len(nvec_compt) == 0:
            return
        unl = self._b.compute_unl(nvec_compt)
        for i in range(len(nvec_compt)):
            self._Unl_cache[nvec_compt[i]] = numpy.sqrt(self._beta) * unl[i,:]

    def compute_Unl(self, nvec):
        self._precompute_Unl(nvec)
        num_n = len(nvec)
        Unl = numpy.empty((num_n, self.dim()), dtype=complex)
        for i in range(num_n):
            Unl[i, :] = self._Unl_cache[nvec[i]]
        return Unl

def sampling_points_matsubara(basis_beta, whichl):
    basis = basis_beta.basis_xy
    stat = basis.statistics
    beta = basis_beta.beta

    assert stat == 'F' or stat == 'B' or stat == 'barB'

    if stat == 'barB':
        if whichl < 2:
            raise RuntimeError("whichl < 2!")

    x1 = numpy.arange(1000)
    x2 = numpy.array(numpy.exp(numpy.linspace(numpy.log(1000), numpy.log(1E+8), 1000)), dtype=int)
    x = numpy.unique(numpy.hstack((x1,x2)))
    unl = basis.compute_unl(x)

    shift = 0 if stat=='F' else 1
    if (whichl+shift)%2 == 0:
        y = numpy.sqrt(beta)*unl[:,whichl].imag
    else:
        y = numpy.sqrt(beta)*unl[:,whichl].real

    sign_change = [(x[i], x[i+1], y[i+1], i) for i in range(len(x)-1) if y[i] * y[i+1] <= 0]

    zeros = numpy.array([0.5*(p[0]+p[1]) for p in sign_change])

    zero_mids = numpy.array([0.5*(zeros[i]+zeros[i+1]) for i in range(len(zeros)-1)], dtype=int)

    # Find the point where abs(y) takes a maximum value after the last sign change.
    one_after_last_sign_change = sign_change[-1][3]
    last_sampling_point = x[one_after_last_sign_change + numpy.argmax(numpy.abs(y[one_after_last_sign_change:]))]

    sp_half = numpy.hstack(([0], zero_mids, [last_sampling_point]))

    if stat == 'F':
        r = numpy.sort(numpy.hstack((sp_half, -sp_half-1)))
    else:
        r = numpy.unique(numpy.hstack((sp_half, -sp_half)))

    #assert len(r) >= whichl+1

    return r

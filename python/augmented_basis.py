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

#class Basis(object):
    #def __init__(self, , beta):

class AugmentedBasisBoson(object):
    def __init__(self, Lambda):
        self._bb = irbasis.load('B', Lambda)
        self._dim = self._bb.dim() + 2

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
        unl = numpy.zeros((num_n, self._dim), dtype=dcomplex)
        unl[:, 0] = numpy.array([_compute_Tnl_norm_legendre(n, l) for n in nvec for l in range(2)]).reshape(num_n, 2)
        unl[:, 2:] = self.compute_unl(nvec)
        return unl

    def dim(self):
        return self._dim

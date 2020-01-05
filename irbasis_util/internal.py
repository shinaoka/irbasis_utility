from __future__ import print_function, division
from builtins import range

import numpy
import scipy
import scipy.linalg

def my_mod(t, beta):
    t_new = t
    s = 1
    
    max_loop = 10

    loop = 0
    while t_new >= beta and loop < max_loop:
        t_new -= beta
        s *= -1
        loop += 1
        
    loop = 0
    while t_new < 0 and loop < max_loop:
        t_new += beta
        s *= -1
        loop += 1

    if not (t_new >= 0 and t_new <= beta):
        print("Error in _my_mod ", t, t_new, beta)

    assert t_new >= 0 and t_new <= beta
    
    return t_new, s

def find_zeros(ulx):
    Nx = 10000
    eps = 1e-14
    tvec = numpy.linspace(-3, 3, Nx) #3 is a very safe option.
    xvec = numpy.tanh(0.5*numpy.pi*numpy.sinh(tvec))

    zeros = []
    for i in range(Nx-1):
        if ulx(xvec[i]) * ulx(xvec[i+1]) < 0:
            a = xvec[i+1]
            b = xvec[i]
            u_a = ulx(a)
            u_b = ulx(b)
            while a-b > eps:
                half_point = 0.5*(a+b)
                if ulx(half_point) * u_a > 0:
                    a = half_point
                else:
                    b = half_point
            zeros.append(0.5*(a+b))
    return numpy.array(zeros)

def eigh_ordered(mat):
    n=mat.shape[0]
    evals,evecs=numpy.linalg.eigh(mat)
    idx=numpy.argsort(evals)
    evecs2=numpy.zeros_like(evecs)
    evals2=numpy.zeros_like(evals)
    for ie in range (n):
        evals2[ie]=evals[idx[ie]]
        evecs2[:,ie]=1.0*evecs[:,idx[ie]]
    return evals2,evecs2

def from_complex_to_real_coeff_matrix(A):
    (N1, N2) = A.shape
    A_big = numpy.zeros((2,N1,2,N2), dtype=float)
    A_big[0,:,0,:] =  A.real
    A_big[0,:,1,:] = -A.imag
    A_big[1,:,0,:] =  A.imag
    A_big[1,:,1,:] =  A.real
    return A_big.reshape((2*N1, 2*N2))

def check_type(obj, types):
    if isinstance(obj, type):
        raise RuntimeError("Passed the argument of the wrong " + str(type(obj)))

def check_value(val, expectation):
    if val != expectation:
        raise RuntimeError("Passed the wrong value" + str(val) + ", the expected is " + str(expectation))

def composite_leggauss(deg, section_edges):
    """
    Composite Gauss-Legendre quadrature.
    :param deg: Number of sample points and weights. It must be >= 1.
    :param section_edges: array_like
                          1-D array of the two end points of the integral interval
                          and breaking points in ascending order.
    :return ndarray, ndarray: sampling points and weights
    """
    x_loc, w_loc = numpy.polynomial.legendre.leggauss(deg)

    ns = len(section_edges)-1
    x = numpy.zeros((ns, deg))
    w = numpy.zeros((ns, deg))
    for s in range(ns):
        dx = section_edges[s+1] - section_edges[s]
        x0 = section_edges[s]
        x[s, :] = (dx/2)*(x_loc+1) + x0
        w[s, :] = w_loc*(dx/2)
    return x.reshape((ns*deg)), w.reshape((ns*deg))


def find_roots(ulx_data, xoffset, tol=2e-16):
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
    roots.sort()
    unique = numpy.ediff1d(roots, to_end=2*tol) > tol
    roots = roots[unique]
    return roots

def get_sampling_points(roots):
    """Get sampling points for x, given roots"""
    if roots.min() < -1 or roots.max() > 1:
        raise ValueError("domain of x")

    aug_roots = numpy.hstack((-1., roots, 1.))
    aug_roots.sort()
    x_sampling = .5 * (aug_roots[:-1] + aug_roots[1:])
    return x_sampling

def tau_for_x(x, beta):
    """Rescales tau axis to x -1 ... 1"""
    if x.min() < -1 or x.max() > 1:
        raise ValueError("domain of x")
    return .5 * beta * (x + 1)

def x_for_tau(tau, beta):
    """Rescales xaxis to tau in 0 ... beta"""
    if tau.min() < 0 or tau.max() > beta:
        raise ValueError("domain of tau")
    return 2/beta * (tau - beta/2)

def get_ultau(tau, beta, basis):
    """Get Ul(tau) for all l and tau"""
    x = x_for_tau(tau, beta)
    ulx = numpy.asarray([basis.ulx_all_l(xi) for xi in x]).T
    Ultau = numpy.sqrt(2/beta) * ulx
    return Ultau

def o_to_matsubara_idx_f(o):
    """
    Convert index in "o" convension to fermionic Matsubara index

    Parameters
    ----------
    o 2*n+1

    Returns n
    -------

    """
    assert o%2 == 1
    return int((o-1)/2)

def o_to_matsubara_idx_b(o):
    """
    Convert index in "o" convension to bosonic Matsubara index

    Parameters
    ----------
    o 2*n

    Returns n
    -------

    """
    assert o%2 == 0
    return int(o/2)

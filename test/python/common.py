from __future__ import print_function
import numpy
import irbasis
from irbasis_util.two_point_basis import augmented_basis_b, vertex_basis, Basis

# G. Rohringer et al., PRB 86, 125114 (2012)

def _delta(i, j):
    if i==j:
        return 1
    else:
        return 0

def _F_ph(U, beta, n, np, m):
    nu = (2 * n + 1) * numpy.pi / beta
    nu_p = (2 * np + 1) * numpy.pi / beta
    omega = 2 * m * numpy.pi / beta
    r1 = nu + omega
    r2 = nu_p + omega
    tmp = 1. / (nu * r1 * r2 * nu_p)
    Fuu = (-0.25 * beta * (U**2) * (_delta(n,np) - _delta(m,0)) *
               (1 + 0.25 * (U / nu)**2) * (1 + 0.25 * (U / r2)**2))    
    t1 = 0.125 * (U**3) * (nu**2 + r1**2 + r2**2 + nu_p**2) * tmp
    t2 = (3.0 / 16.0) * (U**5) * tmp
    t3 = (beta * 0.25 * (U**2) *
              (1 / (1 + numpy.exp(0.5 * beta * U)))
              * (2 * _delta(nu, -nu_p - m) + _delta(m, 0)) *
              (1 + 0.25 * (U / r1)**2) * (1 + 0.25 * (U / r2)**2))
    t4 = (-beta * 0.25 * (U**2) *
              (1 / (1 + numpy.exp(-0.5 * beta * U)))
              * (2 * _delta(nu, nu_p) + _delta(m, 0)) *
              (1 + 0.25 * (U / nu)**2) * (1 + 0.25 * (U / r2)**2))
    Fud = -U + t1 + t2 + t3 + t4
    return Fuu, Fud

# The second term of the right-hand side of Eq. (9)
def G2_conn_ph(U, beta, n, np, m):
    Fuu, Fud = _F_ph(U, beta, n, np, m)
    nu = (2 * n + 1) * numpy.pi / beta
    nu_p = (2 * np + 1) * numpy.pi / beta
    omega = 2 * m * numpy.pi / beta    
    hU = 0.5 * U
    leggs_1 = nu * (nu + omega) * nu_p * (nu_p + omega)
    leggs_2 = ((hU**2 + nu**2) * (hU**2 + nu_p**2) *
                   (hU**2 + (nu + omega)**2) * (hU**2 + (nu_p + omega)**2))
    leggs = leggs_1 / leggs_2
    return leggs * Fuu + leggs * Fud

def Gl_pole_F(B, pole):
    Sl = numpy.array([B.Sl(l) for l in range(B.dim)])
    Vlpole = numpy.array([B.Vlomega(l, pole) for l in range(B.dim)])
    return -Sl * Vlpole

def Gl_pole_barB(B, pole):
    Sl = numpy.array([B.Sl(l) for l in range(B.dim)])
    assert pole != 0
    Vlpole = numpy.zeros((B.dim))
    Vlpole[2:] = numpy.sqrt(1 / B.wmax) * numpy.array([B.basis_xy.basis_b.vly(l, pole / B.wmax)
                                                       for l in range(B.dim-2)])
    return -Sl * Vlpole / pole


def load_basis(stat, Lambda, beta):
    if stat in ['F', 'B']:
        b = irbasis.load(stat, Lambda)
    elif stat == 'barB':
        b = augmented_basis_b(irbasis.load('B', Lambda))
    elif stat == 'barBV':
        b = vertex_basis(augmented_basis_b(irbasis.load('B', Lambda)))
    elif stat == 'FV':
        b = vertex_basis(irbasis.load('F', Lambda))
    else:
        raise RuntimeError("Invalid stat")

    return b, Basis(b, beta)

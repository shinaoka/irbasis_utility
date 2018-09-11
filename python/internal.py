import numpy
import scipy

def _my_mod(t, beta):
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

def _find_zeros(ulx):
    Nx = 10000
    eps = 1e-10
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

def _eigh_ordered(mat):
    n=mat.shape[0]
    evals,evecs=numpy.linalg.eigh(mat)
    idx=numpy.argsort(evals)
    evecs2=numpy.zeros_like(evecs)
    evals2=numpy.zeros_like(evals)
    for ie in range (n):
        evals2[ie]=evals[idx[ie]]
        evecs2[:,ie]=1.0*evecs[:,idx[ie]]
    return evals2,evecs2

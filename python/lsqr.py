"""Sparse Equations and Least Squares.

The original Fortran code was written by C. C. Paige and M. A. Saunders as
described in

C. C. Paige and M. A. Saunders, LSQR: An algorithm for sparse linear
equations and sparse least squares, TOMS 8(1), 43--71 (1982).

C. C. Paige and M. A. Saunders, Algorithm 583; LSQR: Sparse linear
equations and least-squares problems, TOMS 8(2), 195--209 (1982).

It is licensed under the following BSD license:

Copyright (c) 2006, Systems Optimization Laboratory
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
      copyright notice, this list of conditions and the following
      disclaimer in the documentation and/or other materials provided
      with the distribution.

    * Neither the name of Stanford University nor the names of its
      contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The Fortran code was translated to Python for use in CVXOPT by Jeffery
Kline with contributions by Mridul Aanjaneya and Bob Myhill.

Adapted for SciPy by Stefan van der Walt.

"""

from __future__ import division, print_function, absolute_import

__all__ = ['lsqr']

import numpy as np
from math import sqrt
from scipy.sparse.linalg.interface import aslinearoperator

eps = np.finfo(np.float64).eps


def _sym_ortho(a, b):
    """
    Stable implementation of Givens rotation.

    Notes
    -----
    The routine 'SymOrtho' was added for numerical stability. This is
    recommended by S.-C. Choi in [1]_.  It removes the unpleasant potential of
    ``1/eps`` in some important places (see, for example text following
    "Compute the next plane rotation Qk" in minres.py).

    References
    ----------
    .. [1] S.-C. Choi, "Iterative Methods for Singular Linear Equations
           and Least-Squares Problems", Dissertation,
           http://www.stanford.edu/group/SOL/dissertations/sou-cheng-choi-thesis.pdf

    """
    if b == 0:
        return np.sign(a), 0, abs(a)
    elif a == 0:
        return 0, np.sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = np.sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = np.sign(a) / sqrt(1+tau*tau)
        s = c * tau
        r = a / c
    return c, s, r


def lsqr(A, b, damp=0.0, atol=1e-8, btol=1e-8, conlim=1e8,
         iter_lim=None, show=False, calc_var=False, x0=None, comm=None, atol_r1norm=None):
    """Find the least-squares solution to a large, sparse, linear system
    of equations.

    The parameters and behavior are the same as scipy.sparse.linalg.lsqr except for the folloing points. 

    A and b must contain a subset of rows of the full matrices, respectively. 
    The full matrix of the solution is returned on all nodes.

    """

    MPI_on = not comm is None

    if MPI_on:
        from mpi4py import MPI
        rank = comm.Get_rank()
    else:
        rank = 0


    A = aslinearoperator(A)
    b = np.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    m, n = A.shape
    if iter_lim is None:
        iter_lim = 2 * n
    var = np.zeros(n)

    msg = ('The exact solution is  x = 0                              ',
         'Ax - b is small enough, given atol, btol                  ',
         'The least-squares solution is good enough, given atol     ',
         'The estimate of cond(Abar) has exceeded conlim            ',
         'Ax - b is small enough for this machine                   ',
         'The least-squares solution is good enough for this machine',
         'Cond(Abar) seems to be too large for this machine         ',
         'The iteration limit has been reached                      ')

    if show:
        print(' ')
        print('LSQR            Least-squares solution of  Ax = b')
        str1 = 'The matrix A has %8g rows  and %8g cols' % (m, n)
        str2 = 'damp = %20.14e   calc_var = %8g' % (damp, calc_var)
        str3 = 'atol = %8.2e                 conlim = %8.2e' % (atol, conlim)
        str4 = 'btol = %8.2e               iter_lim = %8g' % (btol, iter_lim)
        print(str1)
        print(str2)
        print(str3)
        print(str4)

    itn = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1/conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    def norm_allreduce(vec):
        if MPI_on:
            return sqrt(comm.allreduce(np.dot(vec.conjugate(),vec).real))
        else:
            return np.linalg.norm(vec)

    """
    Set up the first vectors u and v for the bidiagonalization.
    These satisfy  beta*u = b - A*x,  alfa*v = A'*u.
    """
    u = b
    bnorm = norm_allreduce(b)
    if x0 is None:
        x = np.zeros(n)
        beta = bnorm
    else:
        x = np.asarray(x0)
        u = u - A.matvec(x)
        beta = norm_allreduce(u)

    if beta > 0:
        u = (1/beta) * u
        if MPI_on:
            v = comm.allreduce(A.rmatvec(u))
        else:
            v = A.rmatvec(u)
        alfa = np.linalg.norm(v)
    else:
        v = x.copy()
        alfa = 0

    if alfa > 0:
        v = (1/alfa) * v
    w = v.copy()

    rhobar = alfa
    phibar = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    arnorm = alfa * beta
    if arnorm == 0:
        #print(msg[0])
        return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

    head1 = '   Itn      x[0]       r1norm     r2norm '
    head2 = ' Compatible    LS      Norm A   Cond A'

    if show:
        print(' ')
        print(head1, head2)
        test1 = 1
        test2 = alfa / beta
        str1 = '%6g %12.5e' % (itn, x[0])
        str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
        str3 = '  %8.1e %8.1e' % (test1, test2)
        print(str1, str2, str3)

    r1norm_old = None

    # Main iteration loop.
    while itn < iter_lim:
        itn = itn + 1
        """
        %     Perform the next step of the bidiagonalization to obtain the
        %     next  beta, u, alfa, v.  These satisfy the relations
        %                beta*u  =  a*v   -  alfa*u,
        %                alfa*v  =  A'*u  -  beta*v.
        """
        u = A.matvec(v) - alfa * u
        #beta = np.linalg.norm(u)
        beta = norm_allreduce(u)

        if beta > 0:
            u = (1/beta) * u
            anorm = sqrt(anorm**2 + alfa**2 + beta**2 + damp**2)
            if MPI_on:
                v = comm.allreduce(A.rmatvec(u)) - beta * v
            else:
                v = A.rmatvec(u) - beta * v
            alfa = np.linalg.norm(v)
            if alfa > 0:
                v = (1 / alfa) * v

        # Use a plane rotation to eliminate the damping parameter.
        # This alters the diagonal (rhobar) of the lower-bidiagonal matrix.
        rhobar1 = sqrt(rhobar**2 + damp**2)
        cs1 = rhobar / rhobar1
        sn1 = damp / rhobar1
        psi = sn1 * phibar
        phibar = cs1 * phibar

        # Use a plane rotation to eliminate the subdiagonal element (beta)
        # of the lower-bidiagonal matrix, giving an upper-bidiagonal matrix.
        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        dk = (1 / rho) * w

        x = x + t1 * w
        w = v + t2 * w
        ddnorm = ddnorm + np.linalg.norm(dk)**2

        if calc_var:
            var = var + dk**2

        # Use a plane rotation on the right to eliminate the
        # super-diagonal element (theta) of the upper-bidiagonal matrix.
        # Then use the result to estimate norm(x).
        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = sqrt(xxnorm + zbar**2)
        gamma = sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        # Test for convergence.
        # First, estimate the condition of the matrix  Abar,
        # and the norms of  rbar  and  Abar'rbar.
        acond = anorm * sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        # Distinguish between
        #    r1norm = ||b - Ax|| and
        #    r2norm = rnorm in current code
        #           = sqrt(r1norm^2 + damp^2*||x||^2).
        #    Estimate r1norm from
        #    r1norm = sqrt(r2norm^2 - damp^2*||x||^2).
        # Although there is cancellation, it might be accurate enough.
        r1sq = rnorm**2 - dampsq * xxnorm
        r1norm = sqrt(abs(r1sq))
        if r1sq < 0:
            r1norm = -r1norm
        r2norm = rnorm

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.
        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1 / (acond + eps)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm

        # The following tests guard against extremely small values of
        # atol, btol  or  ctol.  (The user may have set any or all of
        # the parameters  atol, btol, conlim  to 0.)
        # The effect is equivalent to the normal tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.
        if itn >= iter_lim:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        if not r1norm_old is None:
            if abs(abs(r1norm) - r1norm_old) < atol_r1norm:
                istop = 99
        r1norm_old = abs(r1norm)

        # Allow for tolerances set by the user.
        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        # See if it is time to print something.
        prnt = False
        if n <= 40:
            prnt = True
        if itn <= 10:
            prnt = True
        if itn >= iter_lim-10:
            prnt = True
        # if itn%10 == 0: prnt = True
        if test3 <= 2*ctol:
            prnt = True
        if test2 <= 10*atol:
            prnt = True
        if test1 <= 10*rtol:
            prnt = True
        if istop != 0:
            prnt = True

        if prnt:
            if show:
                str1 = '%6g %12.5e' % (itn, x[0])
                str2 = ' %10.3e %10.3e' % (r1norm, r2norm)
                str3 = '  %8.1e %8.1e' % (test1, test2)
                str4 = ' %8.1e %8.1e' % (anorm, acond)
                print(str1, str2, str3, str4)

        if istop != 0:
            break

    # End of iteration loop.
    # Print the stopping condition.
    if show:
        print(' ')
        print('LSQR finished')
        print(msg[istop])
        print(' ')
        str1 = 'istop =%8g   r1norm =%8.1e' % (istop, r1norm)
        str2 = 'anorm =%8.1e   arnorm =%8.1e' % (anorm, arnorm)
        str3 = 'itn   =%8g   r2norm =%8.1e' % (itn, r2norm)
        str4 = 'acond =%8.1e   xnorm  =%8.1e' % (acond, xnorm)
        print(str1 + '   ' + str2)
        print(str3 + '   ' + str4)
        print(' ')

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

if __name__ == '__main__':
    from mpi4py import MPI
    from scipy.sparse.linalg import lsqr as lsqr_ref

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    np.random.seed(rank)

    n1 = 1000
    N1, N2 = n1 * size, 10
    damp = 0.1

    A_pile = [np.random.randn(n1, N2) for r in range(size)]
    y_pile = [np.random.randn(n1) for r in range(size)]

    if rank == 0:
        A_all = np.array(A_pile).reshape((N1,N2))
        y_all = np.array(y_pile).reshape((N1))

    A = comm.scatter(A_pile)
    y = comm.scatter(y_pile)

    r = lsqr(A, y, damp=damp)
    x = r[0]

    if rank == 0:
        r_ref = lsqr_ref(A_all, y_all, damp=damp)
        x_ref = r_ref[0]

        assert np.allclose(x, x_ref)

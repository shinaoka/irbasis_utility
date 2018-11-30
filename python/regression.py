from __future__ import print_function

import numpy
import scipy
import scipy.linalg
from scipy.sparse.linalg import lsqr

def ridge_svd(X, y, alpha, cutoff = 1e-10):
    N1, N2 = X.shape
    U, s, Vt = scipy.linalg.svd(X, full_matrices=False)
    Nm = s.size
    idx = s > cutoff * s[0]
    s_nnz = s[idx][:, numpy.newaxis]
    UTy = numpy.dot(U.T, y)
    d = numpy.zeros((Nm,1), dtype=X.dtype)
    d[idx] = s_nnz / (s_nnz ** 2 + alpha)
    d_UT_y = d.reshape((Nm,)) * UTy.reshape((Nm,))
    return numpy.dot(Vt.T, d_UT_y)

def ridge_coordinate_descent(X, y, alpha, blocks = [], rtol = 1e-8, cutoff = 1e-10):
    N1, N2 = X.shape

    x = numpy.zeros((N2,))
    r = y - numpy.dot(X, x)
    L2 = 0.0
    f = numpy.linalg.norm(r)**2 + alpha * L2

    step = 0
    while True:
        f_old = f

        for indices in blocks:
            mask = numpy.full((N2,), False, dtype=bool)
            mask[indices] = True
    
            x_A_old = numpy.array(x[indices])
    
            tilde_y = r + numpy.dot(X[:, indices], x_A_old)
            tilde_X = X[:, indices]
            x[indices] = ridge_svd(tilde_X, tilde_y, alpha, cutoff)
            r += numpy.dot(X[:, indices], x_A_old - x[indices])
            L2 += numpy.linalg.norm(x[indices])**2 - numpy.linalg.norm(x_A_old)**2
            f = numpy.linalg.norm(r)**2 + alpha * L2
    
        df = (f_old - f)/f_old
        print(step, df, numpy.linalg.norm(r)**2 + alpha * L2, numpy.linalg.norm(y - numpy.dot(X,x) - r))
    
        if df < rtol: 
            break

        step += 1
    return x

def ridge_lsqr(A, y, alpha, tol=1e-10, precond=None, x0=None, verbose=0):
    """
    Perform Ridge regression using LSQR method (optionally with preconditioning)
    For the moment, we assume A is a matrix (not LinearOperator).

    Parameters
    ----------
    A
    y
    alpha

    Returns
    -------

    """
    N1, N2 = A.shape
    A_extend = numpy.empty((N1 + N2, N2))

    if precond is None:
        A_extend[:N1, :] = A
        A_extend[N1:, :] = numpy.sqrt(alpha) * numpy.identity(N2)
    else:
        if isinstance(precond, str) and precond == "column":
            # May be useful if columns of A decay exponentially
            raise RuntimeError("This does not work efficiently! Please do not use.")
            precond_vec = 1/numpy.linalg.norm(A, axis=0)
        else:
            precond_vec = precond

        A_extend[:N1, :] = A[:, :] * precond_vec[None, :]
        A_extend[N1:, :] = numpy.sqrt(alpha) * numpy.diag(precond_vec)

    if x0 is None:
        x0_extend = None
    else:
        if precond is None:
            x0_extend = x0
        else:
            x0_extend = x0/precond[:]

    y_extend = numpy.zeros((N1 + N2))
    y_extend[:N1] = y

    r = lsqr(A_extend, y_extend, damp=0.0, atol=tol, btol=tol, conlim=1e8, x0=x0_extend)
    if verbose > 0:
        print("Number of iterations in LSQR = ", r[2])
    if precond is None:
        return r[0]
    else:
        return r[0] * precond_vec[:]

def ridge_complex(A, y, alpha, solver='svd', x0 = None, precond = None, tol=1e-12, verbose=0):
    (N1, N2) = A.shape
    A_big = numpy.zeros((2,N1,2,N2), dtype=float)
    A_big[0,:,0,:] =  A.real
    A_big[0,:,1,:] = -A.imag
    A_big[1,:,0,:] =  A.imag
    A_big[1,:,1,:] =  A.real

    assert len(y) == N1
    y_big = numpy.zeros((2,N1), dtype=float)
    y_big[0,:] = y.real
    y_big[1,:] = y.imag
    
    if solver == 'svd':
        coef = ridge_svd(A_big.reshape((2*N1, 2*N2)), y_big.reshape((2*N1)), alpha)
    elif solver == 'lsqr':
        if x0 is None:
            x0_big = None
        else:
            x0_big = numpy.zeros((2, N2), dtype=float)
            x0_big[0, :] = x0.real
            x0_big[1, :] = x0.imag
            x0_big = x0_big.reshape(2*N2)

        if precond is None:
            precond_big = None
        elif isinstance(precond, str):
            precond_big = precond
        else:
            assert precond.dtype == float
            precond_big = numpy.empty((2, N2))
            precond_big[0, :] = precond
            precond_big[1, :] = precond
            precond_big = precond_big.reshape((2*N2,))
        coef = ridge_lsqr(A_big.reshape((2*N1, 2*N2)), y_big.reshape((2*N1)), alpha, precond=precond_big, x0=x0_big, tol=tol, verbose=verbose)
    else:
        raise RuntimeError("Uknown solver: " + solver)

    coef = coef.reshape((2,N2))
    return coef[0,:] + 1J * coef[1,:]

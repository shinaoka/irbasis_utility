from __future__ import print_function

import numpy
import scipy
import scipy.linalg

class Ridge(object):
    """
    Implementation Ridge regression using SVD (only float y, A, y are supported)

    |y - A x|_2^2 + alpha |x|_2^2
    """
    def __init__(self, A):
        """
        Parameters
        ----------
        A : 2D ndarray or a tuple of 2D ndarrays
            If a tuple is given, it must be the results of the SVD of A in the form of (U, s, Vt).
            The size of s is arbitrary.
            See Also
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html
        """
        if isinstance(A, numpy.ndarray):
            self._U, self._s, self._Vt = scipy.linalg.svd(A, full_matrices=False)
        elif isinstance(A, tuple):
            self._U, self._s, self._Vt = A
        else:
            raise RuntimeError("Invalid A")
        self._Nm = self._s.size

        self._N1, self_N2 = self._U.shape[0], self._Vt.shape[1]

        assert self._U.dtype == float
        assert self._s.dtype == float
        assert self._Vt.dtype == float

    @property
    def svd(self):
        """
        SVD results of A

        Returns
        -------
        U, s, Vt
        """

        return (self._U, self._s, self._Vt)

    def fit(self, y, alpha, cutoff = 1e-10):
        """
        Perform fit

        Parameters
        ----------
        y : 1D ndarray (float)
            data to fit
        alpha : double
            regularization parameter

        Returns
        x : 1D ndarray (float)
            fitted parameters
        -------

        """
        assert y.size == self._N1

        self._d = numpy.zeros((self._Nm, 1))
        idx = self._s > cutoff * self._s[0]
        s_nnz = self._s[idx][:, numpy.newaxis]
        self._d[idx] = s_nnz / (s_nnz ** 2 + alpha)

        UTy = numpy.dot(self._U.T, y)
        d_UT_y = self._d.reshape((self._Nm,)) * UTy.reshape((self._Nm,))

        return numpy.dot(self._Vt.T, d_UT_y)


def compute_svd_for_ridge_complex(A):
    """
    Compute the SVD of a complex matrix for RidgeComplex.
    First, A is transformed into a float matrix of doubled linear dimension.
    Then, the SVD of this float matrix is computed.

    Parameters
    ----------
    A : 2D ndarray of complex of shape (N1, N2)

    Returns
    -------
    (U, s, Vt):
        The shapes are U(N1, 2, Ns), s(Ns), Vt(Ns, 2, N2)

    """

    (N1, N2) = A.shape
    A_big = numpy.zeros((2, N1, 2, N2), dtype=float)
    A_big[0, :, 0, :] = A.real
    A_big[0, :, 1, :] = -A.imag
    A_big[1, :, 0, :] = A.imag
    A_big[1, :, 1, :] = A.real

    A_big = A_big.reshape((2 * N1, 2 * N2))

    U, s, Vt = scipy.linalg.svd(A_big, full_matrices=False)
    Ns = len(s)
    U =U.reshape((2, N1, Ns))
    Vt = Vt.reshape((Ns, 2, N2))

    return (U, s, Vt)

class RidgeComplex(object):
    """
    Implementation Ridge regression using SVD (complex y, A, y are supported)

    |y - A x|_2^2 + alpha |x|_2^2
    """
    def __init__(self, A):
        """
        Parameters
        ----------
        A : 2D ndarray or a tuple of 2D ndarrays
            If a tuple is given, it must be the results of the SVD of A in the form of (U, s, Vt).
            The size of s is arbitrary.
            See Also
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.svd.html

            The shape of A is (N1, N2).
            The shapes of U, s, Vt must be (2, N1, Ns), (Ns), (Ns, 2, N2), where Ns is the number of singular values.
            The dimension of 2 corresponds to the real and imaginary parts of A.
        """
        if isinstance(A, numpy.ndarray):
            self._U, self._s, self._Vt = compute_svd_for_ridge_complex(A)
        elif isinstance(A, tuple):
            self._U, self._s, self._Vt = A
        else:
            raise RuntimeError("Invalid A")

        self._Ns = len(self._s)
        self._N1, self._N2 = self._U.shape[1], self._Vt.shape[-1]

        self._ridge_float = Ridge(
            (self._U.reshape((2 * self._N1, self._Ns)), self._s, self._Vt.reshape((self._Ns, 2 * self._N2)))
        )

    @property
    def svd(self):
        """
        SVD results of A

        Returns
        -------
        U, s, Vt
        """

        return (self._U, self._s, self._Vt)

    def fit(self, y, alpha, cutoff = 1e-10):
        """
        Perform fit

        Parameters
        ----------
        y : 1D ndarray (float or complex)
            data to fit
        alpha : double
            regularization parameter

        Returns
        x : 1D ndarray (complex)
            fitted parameters
        -------

        """
        assert y.size == self._N1

        y_big = numpy.empty((2, self._N1))
        y_big[0, :] = y.real
        y_big[1, :] = y.imag
        y_big = y_big.reshape(2 * self._N1)
        x_big = self._ridge_float.fit(y_big, alpha, cutoff).reshape((2, self._N2))
        return x_big[0, :] + 1J * x_big[1, :]

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

"""
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
"""

def ridge_complex(A, y, alpha, solver='svd', blocks=[]):
    (N1, N2) = A.shape
    A_big = numpy.zeros((2,N1,2,N2), dtype=float)
    A_big[0,:,0,:] =  A.real
    A_big[0,:,1,:] = -A.imag
    A_big[1,:,0,:] =  A.imag
    A_big[1,:,1,:] =  A.real

    blocks_big = []
    for b in blocks:
        blocks_big.append(numpy.concatenate([numpy.array(b), numpy.array(b)+N2], axis=0))
    
    assert len(y) == N1
    y_big = numpy.zeros((2,N1), dtype=float)
    y_big[0,:] = y.real
    y_big[1,:] = y.imag
    
    if solver == 'svd':
        coef = ridge_svd(A_big.reshape((2*N1,2*N2)), y_big.reshape((2*N1)), alpha)
    elif solver == 'svd_cd':
        print("calling svd_cd")
        coef = ridge_coordinate_descent(A_big.reshape((2*N1,2*N2)), y_big.reshape((2*N1)), alpha, blocks_big)
    elif solver == 'sparse_cg':
        #clf = Ridge(alpha=alpha, solver='sparse_cg', tol=1e-8, max_iter=1000000)
        clf = Ridge(alpha=alpha, solver='svd')
        clf.fit(A_big.reshape((2*N1,2*N2)), y_big.reshape((2*N1)))
        coef = clf.coef_
        print("nitr ", clf.n_iter_)
    else:
        raise RuntimeError("Uknown solver: " + solver)

    coef = coef.reshape((2,N2))
    return coef[0,:] + 1J * coef[1,:]

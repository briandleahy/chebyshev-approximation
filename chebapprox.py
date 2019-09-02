import numpy as np
from numpy.polynomial.chebyshev import chebval


class ChebyshevApproximant(object):
    def __init__(self, function, window, degree=6, nevalpts=None,
                 function_args=(), function_kwargs={}):
        """A 1D Chebyshev approximation / interpolation for an ND function,
        approximating (N-1)D in in the last dimension.

        Parameters
        ----------
        func : callable
            A function that takes scalar arguments (1D) and returns a N
            dimensional array corresponding to that scalar. Make it such that,
            for an array x, f(x)[.....,a] corresponds to f(x[a])

        args : tuple [optional]
            extra arguments to pass to func

        window : tuple (length 2)
            The bounds of the function over which we desire the interpolation

        degree : integer
            Degree of the Chebyshev interpolating polynomial

        evalpts : integer
            Number of Chebyshev points to evaluate the function at

        Examples
        --------
        >>> import numpy as np
        >>> func = lambda x: np.sin(x)
        >>> cheb = ChebyshevInterpolation1D(func, window=(0, np.pi), degree=9,
        ...                                 evalpts=11)
        >>> np.allclose(cheb(1.0), np.sin(1.0), atol=1e-7)
        True
        """
        self.function = function
        self.window = tuple(window)
        self.degree = int(degree)
        if nevalpts is None:
            nevalpts = self.degree + 3
        self.nevalpts = nevalpts
        self.function_args = function_args  # FIXME test
        self.function_kwargs = function_kwargs  # FIXME test

        if self.nevalpts < self.degree:
            raise ValueError("nevalpts must be > degree")

        self._coeffs = self._construct_coefficients()

    def __call__(self, x):
        chebx = self._transform_raw_to_cheb_coordinates(x)
        return chebval(chebx, self._coeffs, tensor=True)

    @property
    def coefficients(self):
        return self._coeffs.copy()

    def evaluate_cheb_tk(self, k, rawx):  # FIXME test
        chebx = self._transform_raw_to_cheb_coordinates(rawx)
        weights = np.zeros(k + 1)
        weights[k] = 1.0
        return np.polynomial.chebyshev.chebval(chebx, weights)

    def _transform_raw_to_cheb_coordinates(self, rawx):
        chebx = 2 * (rawx - min(self.window)) / np.ptp(self.window)
        chebx -= 1.0
        return chebx

    def _transform_cheb_to_raw_coordinates(self, chebx):
        rawx = ((chebx + 1) * (self.window[1] - self.window[0]) * 0.5 +
                self.window[0])
        return rawx

    def _construct_coefficients(self):
        # TODO: make this faster once you test it for ND functions
        N = float(self.nevalpts)

        lvals = np.arange(self.nevalpts).astype('float')
        xpts = self._transform_cheb_to_raw_coordinates(
            np.cos(np.pi * (lvals + 0.5) / N))
        fpts = np.rollaxis(
            self.function(xpts, *self.function_args, **self.function_kwargs),
            -1)

        coeffs = []
        for a in range(self.degree):
            this_coeff = 0.0
            for b in range(self.nevalpts):
                this_coeff += fpts[b] * np.cos(np.pi*a*(lvals[b]+0.5)/N)
            coeffs.append(this_coeff)
        coeffs = np.array(coeffs)
        coeffs *= 2.0 / N
        coeffs[0] *= 0.5
        return np.array(coeffs)



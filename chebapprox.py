import numpy as np
from numpy.polynomial.chebyshev import chebval


class ChebyshevApproximant(object):
    def __init__(self, function, window, degree=6, nevalpts=None,
                 function_args=(), function_kwargs={}):
        """1D Chebyshev approximation / interpolation.

        For a vector function, ```function(x)[0]``` should correspond to
        ```function(x[0])```

        Parameters
        ----------
        function : callable
            A function that takes scalar arguments (1D) and returns a N
            dimensional array corresponding to that scalar. Make it such that,
            for an array x, f(x)[.....,a] corresponds to f(x[a])

        window : tuple (length 2)
            The bounds of the function over which we desire the interpolation

        degree : integer, optional
            Degree of the Chebyshev interpolating polynomial. Default is 6

        nevalpts : integer, optional
            Number of Chebyshev points to evaluate the function at.
            Default is ``degree + 3``.

        function_args : tuple, optional
            extra arguments to pass to func

        function_kwargs : dict, optional
            extra keyword arguments to pass to func

        Examples
        --------
        >>> import numpy as np
        >>> cheb = ChebyshevApproximant(np.sin, window=(0, np.pi), degree=9,
        ...                             nevalpts=11)
        >>> np.allclose(cheb(1.0), np.sin(1.0), atol=1e-7)
        True
        """
        self.function = function
        self.window = tuple(window)
        self.degree = int(degree)
        if nevalpts is None:
            nevalpts = self.degree + 3
        self.nevalpts = nevalpts
        self.function_args = function_args
        self.function_kwargs = function_kwargs

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
        l_plus_05_over_n = (np.arange(self.nevalpts) + 0.5) / self.nevalpts
        cheb_nodes = self._transform_cheb_to_raw_coordinates(
            np.cos(np.pi * l_plus_05_over_n))
        f_at_cheb_nodes = self.function(
            cheb_nodes, *self.function_args, **self.function_kwargs)

        degree = np.arange(self.degree).reshape(-1, 1)
        coeffs = np.sum(
            f_at_cheb_nodes.reshape(1, -1) *
            np.cos(np.pi * degree * l_plus_05_over_n),
            axis=1) * 2.0 / self.nevalpts
        coeffs[0] *= 0.5
        return np.array(coeffs)


class PiecewiseChebyshevApproximant(object):
    def __init__(self, function, window_breakpoints, **kwargs):
        """
        Approximates on [window_breakpoints[0], window_breakpoints[1])
        """
        self.function = function
        self.window_breakpoints = window_breakpoints
        self.kwargs = kwargs

        self._domain = (window_breakpoints[0], window_breakpoints[-1])
        self._windows = self._setup_windows()
        self._approximants = self._setup_approximants()
        self._dtype = self._approximants[0]._coeffs.dtype

    def _setup_windows(self):
        windows = [
            (start, stop) for start, stop in
            zip(self.window_breakpoints[:-1], self.window_breakpoints[1:])]
        return windows

    def _setup_approximants(self):
        return [ChebyshevApproximant(
                    self.function, window=window, **self.kwargs)
                for window in self._windows]

    def __call__(self, x):
        x = np.asarray(x)
        if x.max() >= self._domain[1] or x.min() < self._domain[0]:
            msg = "x must be within interpolation window [{}, {})".format(
                *self._domain)
            raise ValueError(msg)
        result = np.zeros(x.shape, dtype=self._dtype)
        for window, approximant in zip(self._windows, self._approximants):
            mask = self._mask_window(x, window)
            result[mask] = approximant(x[mask])
        return result

    @classmethod
    def _mask_window(cls, x, window):
        return (x >= window[0]) & (x < window[1])


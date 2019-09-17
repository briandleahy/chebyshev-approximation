import unittest

import numpy as np
from numpy.polynomial.chebyshev import chebval

from chebapprox import ChebyshevApproximant, PiecewiseChebyshevApproximant


WIGGLYTOLS = {'atol': 3e-13, 'rtol': 3e-13}
TOLS = {'atol': 1e-14, 'rtol': 1e-14}


class TestChebApproximant(unittest.TestCase):
    def test_stores_init_kwargs(self):
        window = (0, 1.0)
        function = np.sin
        degree = 8
        nevalpts = 11

        cheb =ChebyshevApproximant(
            function, window, degree=degree, nevalpts=nevalpts)
        self.assertTrue(cheb.function is function)
        self.assertEqual(cheb.window, tuple(window))
        self.assertEqual(cheb.degree, degree)
        self.assertEqual(cheb.nevalpts, nevalpts)

    def test_default_nevalpts(self):
        window = (0, 1.0)
        function = np.sin
        degree = 8

        cheb =ChebyshevApproximant(
            function, window, degree=degree, nevalpts=None)
        self.assertEqual(cheb.nevalpts, cheb.degree + 3)

    def test_raises_error_if_nevalpts_lessthan_degree(self):
        self.assertRaises(
            ValueError, ChebyshevApproximant, np.sin, (0, 1), degree=6,
            nevalpts=4)

    def test_transform_raw_to_cheb_coordinates(self):
        cheb = make_cheb_interpolant()
        rawx = generate_random_points_on_window(cheb.window)
        chebx = cheb._transform_raw_to_cheb_coordinates(rawx)
        self.assertTrue(np.all(chebx <= 1))
        self.assertTrue(np.all(chebx >= -1))
        rescale = 0.5 * np.ptp(cheb.window)
        self.assertAlmostEqual(rawx.ptp(), chebx.ptp() * rescale, places=14)

    def test_transform_cheb_to_raw_coordinates(self):
        cheb = make_cheb_interpolant()
        rawx = generate_random_points_on_window(cheb.window)
        chebx = cheb._transform_raw_to_cheb_coordinates(rawx)
        backagain = cheb._transform_cheb_to_raw_coordinates(chebx)
        self.assertTrue(np.allclose(rawx, backagain, **TOLS))

    def test_construct_coefficients(self):
        np.random.seed(10)
        degree = 6
        true_coeffs = np.random.randn(degree)
        f = lambda x: chebval(x, true_coeffs)
        cheb = ChebyshevApproximant(f, window=(-1, 1), degree=degree)
        constructed_coeffs = cheb._construct_coefficients()
        self.assertTrue(np.allclose(true_coeffs, constructed_coeffs, **TOLS))

    def test_coefficients_returns_copy(self):
        cheb = make_cheb_interpolant()
        self.assertTrue(cheb._coeffs is not cheb.coefficients)
        self.assertTrue(np.all(cheb._coeffs == cheb.coefficients))

    def test_call_accurately_approximates(self):
        cheb = ChebyshevApproximant(function=np.sin, window=(0, 1), degree=12)
        x = np.linspace(0, 1, 101)
        true = np.sin(x)
        approx = cheb(x)
        self.assertTrue(np.allclose(true, approx, **TOLS))

    def test_call_accurately_approximates_with_many_wiggles(self):
        window = (0, 50)
        cheb = ChebyshevApproximant(
            function=np.sin, window=window, degree=500)
        x = np.linspace(*window, 101)
        true = np.sin(x)
        approx = cheb(x)
        self.assertTrue(np.allclose(true, approx, **WIGGLYTOLS))

    @unittest.expectedFailure
    def test_call_returns_correct_shape_for_3d_function(self):
        f3d = lambda t: np.array([(np.cos(ti), np.sin(ti), 0) for ti in t])
        cheb3d = ChebyshevApproximant(f3d, (0, 1), degree=3)
        t = np.linspace(0, 1, 101)
        true = f3d(t)
        approx = cheb3d(t)
        self.assertEqual(true.shape, approx.shape)

    @unittest.expectedFailure
    def test_call_accurately_approximates_on_3D_function(self):
        f3d = lambda t: np.array([(np.cos(ti), np.sin(ti), 0) for ti in t])
        cheb3d = ChebyshevApproximant(f3d, (0, 1), degree=13)
        t = np.linspace(0, 1, 101)
        true = f3d(t)
        approx = cheb3d(t)
        self.assertTrue(np.allclose(true, approx, **TOLS))

    def test_works_with_args(self):
        f = lambda x, b: (np.sin(x) if b else np.cos(x))
        cheb_true = ChebyshevApproximant(
            f, (0, 1), degree=12, function_args=(True,))
        cheb_false = ChebyshevApproximant(
            f, (0, 1), degree=12, function_args=(False,))

        x = 0.5
        self.assertAlmostEqual(cheb_true(x), f(x, True), places=13)
        self.assertAlmostEqual(cheb_false(x), f(x, False), places=13)

    def test_works_with_kwargs(self):
        f = lambda x, b=True: (np.sin(x) if b else np.cos(x))
        cheb_true = ChebyshevApproximant(
            f, (0, 1), degree=12, function_kwargs={'b': True})
        cheb_false = ChebyshevApproximant(
            f, (0, 1), degree=12, function_kwargs={'b': False})

        x = 0.5
        self.assertAlmostEqual(cheb_true(x), f(x, b=True), places=13)
        self.assertAlmostEqual(cheb_false(x), f(x, b=False), places=13)


class TestPiecewiseChebyshevApproximant(unittest.TestCase):
    def test_mask_window(self):
        window = (0, 1)
        x = np.linspace(0, 2, 101)
        mask = PiecewiseChebyshevApproximant._mask_window(x, window)
        self.assertLess(x[mask].max(), 1.0)
        self.assertGreaterEqual(x[~mask].min(), 1.0)

    def test_setup_windows_splits_into_n_windows(self):
        nwindows = 5
        piecewisecheb = PiecewiseChebyshevApproximant(
            np.sin, window_breakpoints=np.linspace(0, 1, nwindows + 1))
        windows = piecewisecheb._setup_windows()
        self.assertEqual(len(windows), nwindows)

    def test_setup_windows_partitions_window(self):
        nwindows = 5
        window = (0, 1)
        piecewisecheb = PiecewiseChebyshevApproximant(
            np.sin, window_breakpoints=np.linspace(*window, nwindows + 1))
        windows = piecewisecheb._setup_windows()

        np.random.seed(72)
        x = np.random.rand(101) * np.ptp(window) + window[0]
        masks = [piecewisecheb._mask_window(x, w) for w in windows]
        number_of_masks_contained = np.sum(masks, axis=0)
        self.assertTrue(np.all(number_of_masks_contained == 1))

    def test_setup_approximants_generates_approximants(self):
        piecewisecheb = PiecewiseChebyshevApproximant(
            np.sin, window_breakpoints=np.linspace(0, 1, 6))
        approximants = piecewisecheb._setup_approximants()
        for approximant in approximants:
            self.assertTrue(isinstance(approximant, ChebyshevApproximant))

    def test_dtype_on_float(self):
        piecewisecheb = PiecewiseChebyshevApproximant(
            np.sin, window_breakpoints=np.linspace(0, 1, 6))
        self.assertEqual(piecewisecheb._dtype.name, 'float64')

    def test_dtype_on_complex(self):
        piecewisecheb = PiecewiseChebyshevApproximant(
            lambda x: np.exp(1j * x), window_breakpoints=np.linspace(0, 1, 6))
        self.assertEqual(piecewisecheb._dtype.name, 'complex128')

    def test_call_raises_error_when_x_less_than_window(self):
        window = (0, 10)
        piecewisecheb = PiecewiseChebyshevApproximant(
            np.sin, window_breakpoints=np.linspace(*window, 6))
        self.assertRaises(ValueError, piecewisecheb, window[0] - 1)

    def test_call_raises_error_when_x_greater_than_window(self):
        window = (0, 10)
        piecewisecheb = PiecewiseChebyshevApproximant(
            np.sin, window_breakpoints=np.linspace(*window, 6))
        self.assertRaises(ValueError, piecewisecheb, window[1] + 1)

    def test_call_raises_error_when_x_equal_to_max_window(self):
        window = (0, 10)
        piecewisecheb = PiecewiseChebyshevApproximant(
            np.sin, window_breakpoints=np.linspace(*window, 6))
        self.assertRaises(ValueError, piecewisecheb, window[1])

    def test_call_returns_correct_shape(self):
        window = (0, 20)
        piecewisecheb = PiecewiseChebyshevApproximant(
            np.sin, window_breakpoints=np.linspace(*window, 6))
        x = np.linspace(window[0], window[1] - 0.1, 101)
        true = np.sin(x)
        approx = piecewisecheb(x)
        self.assertEqual(true.shape, approx.shape)

    def test_call_accurately_approximates(self):
        window = (0, 20)
        piecewisecheb = PiecewiseChebyshevApproximant(
            np.sin, window_breakpoints=np.linspace(*window, 21), degree=12)
        x = np.linspace(window[0], window[1] - 0.1, 101)
        true = np.sin(x)
        approx = piecewisecheb(x)
        self.assertTrue(np.allclose(true, approx, **TOLS))


def make_cheb_interpolant(function=np.sin):
    cheb = ChebyshevApproximant(function, (0, 1.0), degree=8)
    return cheb


def generate_random_points_on_window(window, npts=100, seed=72):
    np.random.seed(seed)
    return min(window) + np.ptp(window) * np.random.rand(npts)


if __name__ == '__main__':
    unittest.main()


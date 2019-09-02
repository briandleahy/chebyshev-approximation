import unittest

import numpy as np
from numpy.polynomial.chebyshev import chebval

from chebapprox import ChebyshevApproximant


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

    def test_call_returns_correct_shape_for_3d_function(self):
        f3d = lambda t: np.transpose([(np.cos(ti), np.sin(ti), 0) for ti in t])
        cheb3d = ChebyshevApproximant(f3d, (0, 1), degree=3)
        t = np.linspace(0, 1, 101)
        true = f3d(t)
        approx = cheb3d(t)
        self.assertEqual(true.shape, approx.shape)

    def test_call_accurately_approximates_on_3D_function(self):
        f3d = lambda t: np.transpose([(np.cos(ti), np.sin(ti), 0) for ti in t])
        cheb3d = ChebyshevApproximant(f3d, (0, 1), degree=13)
        t = np.linspace(0, 1, 101)
        true = f3d(t)
        approx = cheb3d(t)
        self.assertTrue(np.allclose(true, approx, **TOLS))


def make_cheb_interpolant(function=np.sin):
    cheb = ChebyshevApproximant(function, (0, 1.0), degree=8)
    return cheb


def generate_random_points_on_window(window, npts=100, seed=72):
    np.random.seed(seed)
    return min(window) + np.ptp(window) * np.random.rand(npts)


if __name__ == '__main__':
    unittest.main()


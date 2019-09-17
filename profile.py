import timeit

import numpy as np
from numpy.polynomial.chebyshev import Chebyshev

from chebapprox import ChebyshevApproximant


T = np.linspace(0, 1, 1001)


def setup_mine():
    return ChebyshevApproximant(np.sin, (0, 1), degree=11, nevalpts=11)


def setup_and_call_mine():
    cheb = setup_mine()
    return cheb(T)


def setup_numpy():
    return Chebyshev.interpolate(np.sin, deg=10, domain=(0, 1))


def setup_and_call_numpy():
    cheb = setup_numpy()
    return cheb(T)


_CHEB_MINE = setup_mine()
_CHEB_NUMPY = setup_numpy()


def call_mine():
    return _CHEB_MINE(T)


def call_numpy():
    return _CHEB_NUMPY(T)


VALID_NAMES = [
    "setup_mine",
    "setup_numpy",
    "call_mine",
    "call_numpy",
    "setup_and_call_mine",
    "setup_and_call_numpy",
    ]


def time_function(name, ncalls=1000):
    assert name in VALID_NAMES
    t = timeit.Timer(
        '{}()'.format(name),
        setup='from __main__ import {}'.format(name))
    nseconds = t.timeit(ncalls)
    time_per_call = nseconds / ncalls
    msg = '{}:\t{:0.2f} us'.format(name.ljust(21), time_per_call * 1e6)
    print(msg)
    return time_per_call


if __name__ == '__main__':
    for name in VALID_NAMES:
        time_function(name)



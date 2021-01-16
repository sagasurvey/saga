"""
http://www.castor2.ca/07_News/headline_062515.html
sin^2(dist/2) = sin^2(d(Dec)/2) + cos(Dec1) cos(Dec2) sin^2(d(RA)/2)
"""

import numpy as np

__all__ = ["is_within"]


def _is_simple_within(x, lower, upper):
    return (x >= lower) and (x <= upper)


def _cos(x):
    return np.cos(np.deg2rad(x))


def _sin_sq_half(x):
    return np.sin(np.deg2rad(x) / 2) ** 2


def _min_sin_sq_half(x, lower, upper):
    return _sin_sq_half(x - np.array([lower, upper])).min()


def is_within(ra, dec, ra_min, ra_max, dec_min, dec_max, margin=0):

    if _is_simple_within(ra, ra_min, ra_max):
        return _is_simple_within(dec, dec_min - margin, dec_max + margin)

    minda = _min_sin_sq_half(ra, ra_min, ra_max)

    if _is_simple_within(dec, dec_min, dec_max):
        mindd = 0
    else:
        mindd = _min_sin_sq_half(dec, dec_min, dec_max)

    d = mindd + minda * _cos(dec) * _cos(max(abs(dec_min), abs(dec_max)))

    return d <= _sin_sq_half(margin)

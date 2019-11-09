import astropy.constants
import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM

SPEED_OF_LIGHT = astropy.constants.c.to_value("km/s")  # pylint: disable=no-member
COSMO = FlatLambdaCDM(70, 0.27)  # same as HyperLEDA


def ensure_scalar(x, unit=""):
    return x.to_value(unit=unit) if isinstance(x, u.Quantity) else x


def m2d(m):
    return 10.0 ** ((ensure_scalar(m, "mag") - 25.0) / 5.0)


def d2m(d):
    return np.log10(ensure_scalar(d, "Mpc")) * 5.0 + 25.0


def v2z(v):
    return ensure_scalar(v, "km/s") / SPEED_OF_LIGHT


def z2v(z):
    return ensure_scalar(z, "") * SPEED_OF_LIGHT


def z2d(z):
    return ensure_scalar(COSMO.luminosity_distance(ensure_scalar(z, "")), "Mpc")


def d2z(d, zmax=0.1):
    _z = np.linspace(0, zmax, int(1.0e7 * zmax) + 1)
    _d = COSMO.luminosity_distance(_z).to_value("Mpc")
    return np.interp(ensure_scalar(d, "Mpc"), _d, _z, left=np.nan, right=np.nan)


def z2m(z):
    return d2m(z2d(z))


def m2z(m):
    return d2z(m2d(m))


def v2d(v):
    return z2d(v2z(v))


def d2v(d):
    return z2v(d2z(d))


def v2m(v):
    return d2m(z2d(v2z(v)))


def m2v(m):
    return z2v(d2z(m2d(m)))

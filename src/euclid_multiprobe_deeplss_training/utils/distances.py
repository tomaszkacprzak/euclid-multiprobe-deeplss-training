import numpy as np
import astropy.units as u
from astropy.cosmology import Planck18

def angular_size(z, size_mpc, cosmo=Planck18, unit="arcmin"):
    """
    Angular size of an object with a fixed *proper* (physical) size at redshift z.

        theta = D_proper / D_A(z)

    where D_A is the angular diameter distance (small-angle approximation).

    Parameters
    ----------
    z : float or array_like
        Redshift(s).
    size_mpc : float
        Proper (physical, not comoving) size of the object in Mpc.
    cosmo : astropy.cosmology instance, optional
        Default is Planck18 (H0=67.66, Om0=0.30966). Planck15 / Planck13 also work.
    unit : str, optional
        Output angular unit (default "arcmin"); also "arcsec", "deg" or "rad".

    Returns
    -------
    theta : ndarray
        Angular size in the requested unit, same shape as z.
        z = 0 returns inf (object at zero distance).
    """
    z = np.asarray(z, dtype=float)
    d_a = cosmo.angular_diameter_distance(z).to_value(u.Mpc)
    with np.errstate(divide="ignore"):
        theta_rad = size_mpc / d_a
    return (theta_rad * u.rad).to_value(unit)


def smoothing_sigma(z, size_mpc):
    """
    Smoothing sigma for a given redshift and size.
    
    Parameters
    ----------
    z : float or array_like
        Redshift(s).
    size_mpc : float
        Proper (physical, not comoving) size of the object in Mpc.

    Returns
    -------
    sigma : ndarray
        Smoothing sigma in the requested unit, same shape as z.
    """
    sigma = angular_size(z, size_mpc) / 2.355
    return sigma
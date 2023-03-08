"""
From Marla 03/07/2023
"""
import numpy as np

__all__ = ["calc_SFR_NUV", "calc_SFR_Halpha"]


def calc_SFR_NUV(NUV_mag, NUV_mag_err, EBV, dist_mpc, internal_ext=0.5):
    """
    Convert NUV magnitudes into a SFR
    Based on Iglesias-Paramo (2006), Eq 3
    https://ui.adsabs.harvard.edu/abs/2006ApJS..164...38I/abstract
    """

    # GALACTIC REDDENING FROM SALIM 2016, EQ7
    ANUV = 8.36 * EBV + 14.3 * EBV**2 - 82.8 * EBV**3

    # DISTANCE OF HOST (in cm)
    dist = dist_mpc * 3.086e24
    dmod = np.log10(4.0 * np.pi * dist * dist)

    # CORRECT FOR GALACTIC REDDENING AND INTERNAL EXTINCTION = 0.5
    # 0.5 is based on an average correction from WISE band W4
    m_nuv_ab = NUV_mag - ANUV - internal_ext

    # CONVERT GALEX m_AB TO FLUX:  erg sec-1 cm-2 Å-1)
    # https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html
    #
    log_flux_nuv = -0.4 * (m_nuv_ab - 20.08 - 2.5 * np.log10(2.06e-16))
    log_flux_nuv_err = 0.4 * (NUV_mag_err)  # ADD EXTRA ERROR FOR REDDENING OR DISTANCE?

    # LUMINOSITY (erg/s/A-1)
    # 796A is NUV filter width
    log_L_nuv = log_flux_nuv + dmod + np.log10(796)

    # CONVERT TO SOLAR LUMINOSITY
    l_nuv_msun = log_L_nuv - np.log10(3.826e33)

    # CONVVERT TO SFR:   EQ 3, inglesias- paramo 2006
    log_SFR_NUV = l_nuv_msun - 9.33
    log_SFR_NUV_err = log_flux_nuv_err

    return log_SFR_NUV, log_SFR_NUV_err


def calc_SFR_Halpha(EW_Halpha, EW_Halpha_err, spec_z, Mr, EWc=3, BD=3.5):
    """
    Calculate Halpha-based EW SFR
    Bauer+ (2013) https://ui.adsabs.harvard.edu/abs/2013MNRAS.434..209B/abstract
    """

    # Bauer, EQ 2, term1
    term1 = (EW_Halpha + EWc) * 10 ** (-0.4 * (Mr - 34.1))
    term1_err = EW_Halpha_err * 10 ** (-0.4 * (Mr - 34.1))

    # Bauer Eq 2, term2
    term2 = 3e18 / (6564.6 * (1.0 + spec_z)) ** 2

    # Balmer Decrement
    term3 = (BD / 2.86) ** 2.36

    L_Halpha = term1 * term2 * term3
    L_Halpha_err = term1_err * term2 * term3

    # EQ 3, Bauer et al above
    # Account for IMF
    SFR = L_Halpha / (1.27e34 * 1.5)
    SFR_err = L_Halpha_err / (1.27e34 * 1.5)

    log_Ha_SFR = np.log10(SFR)

    # PROPOGATE ERRORS
    log_SFR_err2 = SFR_err**2 * (1.0 / (SFR * np.log(10.0))) ** 2
    log_Ha_SFR_err = np.sqrt(log_SFR_err2)

    return log_Ha_SFR, log_Ha_SFR_err

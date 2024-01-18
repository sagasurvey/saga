"""
From Marla 03/07/2023
"""
import numpy as np

__all__ = ["calc_SFR_NUV", "calc_SFR_Halpha"]

# Salpeter -> Koupa IMF using Madua & Dickinson 2014 (Figure 4)
_IMF_FACTOR = 0.66


def calc_SFR_NUV(NUV_mag, NUV_mag_err, dist_mpc, internal_ext=0.9, internal_ext_err=0.2):
    """
    Convert NUV magnitudes into a SFR
    Based on Iglesias-Paramo (2006), Eq 3
    https://ui.adsabs.harvard.edu/abs/2006ApJS..164...38I/abstract
    """

    # DISTANCE OF HOST (in cm)
    dist = dist_mpc * 3.086e24
    dmod = np.log10(4.0 * np.pi * dist * dist)

    # CORRECT FOR INTERNAL EXTINCTION (assumed to be external extinction corrected)
    m_nuv_ab = NUV_mag - internal_ext

    # CONVERT GALEX m_AB TO FLUX:  erg sec-1 cm-2 Angstrom-1)
    # https://asd.gsfc.nasa.gov/archive/galex/FAQ/counts_background.html
    log_flux_nuv = -0.4 * (m_nuv_ab - 20.08 - 2.5 * np.log10(2.06e-16))

    # LUMINOSITY (erg/s/A-1)
    # 796A is NUV filter width
    log_L_nuv = log_flux_nuv + dmod + np.log10(796)

    # CONVERT TO SOLAR LUMINOSITY
    l_nuv_msun = log_L_nuv - np.log10(3.826e33)

    # CONVVERT TO SFR: EQ 3, inglesias- paramo 2006, also account for Salpeter -> Koupa IMF
    log_SFR_NUV = l_nuv_msun - 9.33 + np.log10(_IMF_FACTOR)

    # PROPAGATE ERRORS: assume ANUV_ERR and measurement errors
    # ANUV_ERR is determined to be consistent with BD_ERR
    log_SFR_NUV_err = 0.4 * np.hypot(NUV_mag_err, internal_ext_err)

    return log_SFR_NUV, log_SFR_NUV_err


def calc_SFR_Halpha(EW_Halpha, EW_Halpha_err, spec_z, Mr, r_err, EWc=2.5, BD=3.25, BD_err=0.1):
    """
    Calculate Halpha-based EW SFR
    Bauer+ (2013) https://ui.adsabs.harvard.edu/abs/2013MNRAS.434..209B/abstract
    """

    # Bauer, EQ 2, term1
    term1 = (EW_Halpha + EWc) * 10 ** (-0.4 * (Mr - 34.1))

    # Bauer Eq 2, term2
    term2 = 3e18 / (6564.6 * (1.0 + spec_z)) ** 2

    # Balmer Decrement
    term3 = (BD / 2.86) ** 2.36

    L_Halpha = term1 * term2 * term3

    # EQ 3, Bauer et al above, also account for Salpeter -> Koupa IMF
    SFR = (L_Halpha * _IMF_FACTOR) / 1.27e34
    log_Ha_SFR = np.log10(SFR)

    # PROPAGATE ERRORS: EW_err, Mr_err and AV_err
    term1_EW_frac_err = EW_Halpha_err / (EW_Halpha + EWc)
    term1_Mr_frac_err = 0.4 * np.log(10) * r_err
    term1_frac_err = np.hypot(term1_EW_frac_err, term1_Mr_frac_err)
    term3_frac_err = 2.36 * (BD_err / BD)
    L_Halpha_frac_err = np.hypot(term1_frac_err, term3_frac_err)
    log_Ha_SFR_err  = L_Halpha_frac_err / np.log(10)

    return log_Ha_SFR, log_Ha_SFR_err
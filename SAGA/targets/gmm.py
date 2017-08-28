import os
import numpy as np
from scipy.misc import logsumexp
from ..utils import get_sdss_colors

# compute distance from GMM model with diagonal or full covariances
def _GMMlogposterior(y, yerr, xmap, xmean, xcovar):
    """
    Examples
    --------
    # cols and colerrs are arrays of nobj rows and 4 columns containing the
    # u-g, g-r, r-i and i-z colors and color-errors.

    allpost_nosat = GMMlogposterior(cols[:, :], colerrs[:, :], xamp_nosat, xmean_nosat, xcovar_nosat)
    allpost_sat = GMMlogposterior(cols[:, :], colerrs[:, :], xamp_sat, xmean_sat, xcovar_sat)
    norms = allpost_nosat + allpost_sat

    # Normalize the probabilities like in a binary Bayes classifier.

    allpost_nosat /= norms
    allpost_sat /= norms
    """
    assert y.shape[1] == xmean.shape[1]
    assert xmean.shape[0] == xcovar.shape[0]
    assert xmean.shape[1] == xcovar.shape[1]
    nobj = y.shape[0]
    ndim = xcovar.shape[1]
    dys = y[:, None, :] - xmean[None, :, :] # nobj * ncomp * ndim
    if len(xcovar.shape) == 2:
        covs = yerr[:, None, :]**2 + xcovar[None, :, :]
        lnprobs = - 0.5 * np.sum(dys**2 / covs, axis=2) - 0.5*np.sum(np.log(covs), axis=2)
    if len(xcovar.shape) == 3:
        eyes = np.repeat(np.eye(ndim), nobj).reshape((ndim, ndim, 1, nobj)).T
        covs = yerr[:, None, :, None]**2 * eyes + xcovar[None, :, :, :]
        temp = np.sum(np.linalg.inv(covs) * dys[:, :, None, :], axis=3)
        lnprobs = - 0.5 * np.sum(temp * dys, axis=2) - 0.5*np.log(np.linalg.det(covs))
    lnprobs += np.log(xmap[None, :])
    return np.exp(logsumexp(lnprobs, axis=1)) # nobj


def _change_table_format(table, cols):
    if table.masked:
        return np.vstack((table[c].data.data for c in cols)).T
    else:
        return np.vstack((table[c].data for c in cols)).T


def calc_gmm_satellite_probability(base, model_parameters):

    colors = _change_table_format(base, get_sdss_colors())
    colors_err = _change_table_format(base, ('{}_err'.format(c) for c in get_sdss_colors()))

    p_notsat = _GMMlogposterior(colors, colors_err,
                                model_parameters['xamp_nosat'],
                                model_parameters['xmean_nosat'],
                                model_parameters['xcovar_nosat'])

    p_sat = _GMMlogposterior(colors, colors_err,
                                model_parameters['xamp_sat'],
                                model_parameters['xmean_sat'],
                                model_parameters['xcovar_sat'])

    p_notsat += p_sat
    p_sat /= p_notsat
    p_sat[p_sat == np.inf] = 1.0
    p_sat[~np.isfinite(p_sat)] = 0.0
    return p_sat

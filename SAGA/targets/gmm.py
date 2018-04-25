"""
GMM related routines
"""
import numpy as np
from scipy.special import logsumexp
from ..utils import get_sdss_bands, get_sdss_colors

__all__ = ['param_labels_sat', 'param_labels_nosat', 'calc_gmm_satellite_probability', 'get_input_data', 'calc_model1_prob']


param_labels_sat = ('xmean_sat', 'xcovar_sat', 'xamp_sat')
param_labels_nosat = ('xmean_nosat', 'xcovar_nosat', 'xamp_nosat')

def table2ndarray(table, cols, dtype=None, copy=False):
    cols = list(cols)
    dtype_orig = getattr(np, table[cols[0]].dtype.name)
    out = np.array(table[cols], copy=copy).view((dtype_orig, len(cols)))
    if dtype and np.dtype(dtype) != dtype_orig:
        out = out.astype(np.dtype(dtype))
    return out


def get_input_data(catalog, colors=None, bands=None, include_covariance=True):
    if colors is None:
        colors = get_sdss_colors()
    if bands is None:
        bands = get_sdss_bands()
    assert len(bands) == len(colors) + 1
    X = table2ndarray(catalog, colors, np.float64)
    Xcov = np.stack((np.diag(e*e) for e in table2ndarray(catalog, [c+'_err' for c in colors], np.float64)))
    if include_covariance:
        Xcov -= np.stack(((np.diag(e*e, 1) + np.diag(e*e, -1)) for e in table2ndarray(catalog, [b+'_err' for b in bands[1:-1]], np.float64)))
    return X, Xcov


def check_calc_log_likelihood_input(data, data_cov, gmm_means, gmm_covs, gmm_weights):
    assert 1 == gmm_weights.ndim
    assert 2 == data.ndim == gmm_means.ndim
    assert 3 == data_cov.ndim == gmm_covs.ndim
    assert data.shape[0] == data_cov.shape[0]
    assert gmm_means.shape[0] == gmm_covs.shape[0] == gmm_weights.shape[0]
    assert data.shape[1] == data_cov.shape[1] == data_cov.shape[2] == gmm_means.shape[1] == gmm_covs.shape[1] == gmm_covs.shape[2]


def calc_log_likelihood(data, data_cov, gmm_means, gmm_covs, gmm_weights):
    check_calc_log_likelihood_input(data, data_cov, gmm_means, gmm_covs, gmm_weights)
    d = data[:, np.newaxis] - gmm_means
    cov = data_cov[:, np.newaxis] + gmm_covs
    tmp_result = np.einsum('...i,...ij,...j->...', d, np.linalg.inv(cov), d)
    tmp_result += np.log(np.fabs(np.linalg.det(cov)))
    tmp_result += (np.log(np.pi*2.0) * data.shape[-1])
    tmp_result *= -0.5
    return logsumexp(tmp_result, axis=1, b=gmm_weights)


def calc_model1_prob(data, data_cov, model_params, priors=None):
    p = np.exp(np.stack((calc_log_likelihood(data, data_cov, *model_params_this) for model_params_this in model_params)))
    if priors:
        priors = np.asarray(priors)
        assert len(priors) == len(p)
        assert (priors >= 0).all() and priors.sum() > 0
        p *= priors[:, np.newaxis]
    p_total = p.sum(axis=0)
    mask = (p_total == 0)
    p_total[mask] = 1.0
    p_out = p[0] / p_total
    p_out[mask] = (priors[0] / priors.sum()) if priors else (1.0 / len(p))
    return p_out


def calc_gmm_satellite_probability(base, model_parameters, p_sat_prior=0.5, include_covariance=True):
    data, data_cov = get_input_data(base, include_covariance=include_covariance)
    model_params = (
        tuple(model_parameters[k] for k in param_labels_sat),
        tuple(model_parameters[k] for k in param_labels_nosat)
    )
    return calc_model1_prob(data, data_cov, model_params, [p_sat_prior, 1-p_sat_prior])

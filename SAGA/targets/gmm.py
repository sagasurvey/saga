"""
GMM related routines
"""
import numpy as np
from scipy.special import logsumexp
from ..utils import get_sdss_bands, get_sdss_colors

try:
    from sklearn.mixture import GaussianMixture
    from extreme_deconvolution.extreme_deconvolution import extreme_deconvolution
except ImportError:
    _XDGMM_AVAILABLE = False
else:
    _XDGMM_AVAILABLE = True

__all__ = ['param_labels_sat', 'param_labels_nosat', 'calc_gmm_satellite_probability', 'XDGMM', 'get_input_data', 'calc_log_likelihood', 'calc_model1_prob']


param_labels_sat = ('xmean_sat', 'xcovar_sat', 'xamp_sat')
param_labels_nosat = ('xmean_nosat', 'xcovar_nosat', 'xamp_nosat')


class XDGMM(object):
    def __init__(self, n_components=1, **kwargs):
        if not _XDGMM_AVAILABLE:
            raise RuntimeError('You need to have both sklearn and extreme_deconvolution installed to use XDGMM')
        self._kwargs = kwargs
        self._gmm = GaussianMixture(int(n_components), max_iter=10, covariance_type='full')

    def fit(self, data, data_cov, **kwargs):
        self._gmm.fit(data)
        xd_kwargs = dict(xamp=self._gmm.weights_, xmean=self._gmm.means_, xcovar=self._gmm.covariances_)
        xd_kwargs.update(self._kwargs)
        xd_kwargs.update(kwargs)
        extreme_deconvolution(data, data_cov, **xd_kwargs)

    @property
    def params(self):
        return self._gmm.means_, self._gmm.covariances_, self._gmm.weights_

    def get_params(self, labels=None):
        return dict(zip(labels, self.params)) if labels else self.params


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
    assert 3 == data_cov.ndim
    assert gmm_covs.ndim in (2, 3)
    assert data.shape[0] == data_cov.shape[0]
    assert gmm_means.shape[0] == gmm_covs.shape[0] == gmm_weights.shape[0]
    assert data.shape[1] == data_cov.shape[1] == data_cov.shape[2] == gmm_means.shape[1] == gmm_covs.shape[1] == gmm_covs.shape[2]


def calc_log_likelihood(data, data_cov, gmm_means, gmm_covs, gmm_weights):
    check_calc_log_likelihood_input(data, data_cov, gmm_means, gmm_covs, gmm_weights)
    if gmm_covs.ndim == 2:
        gmm_covs = np.stack((np.diag(c) for c in gmm_covs))
    d = data[:, np.newaxis] - gmm_means
    cov = data_cov[:, np.newaxis] + gmm_covs
    tmp_result = np.einsum('...i,...ij,...j->...', d, np.linalg.inv(cov), d)
    tmp_result += np.log(np.fabs(np.linalg.det(cov)))
    tmp_result += (np.log(np.pi*2.0) * data.shape[-1])
    tmp_result *= -0.5
    return logsumexp(tmp_result, axis=1, b=gmm_weights)


def calc_model1_prob(data, data_cov, model_params, priors=None):
    p = np.exp(np.stack((calc_log_likelihood(data, data_cov, *model_params_this) for model_params_this in model_params)))
    if priors is not None:
        priors = np.asarray(priors)
        assert len(priors) == len(p)
        assert (priors >= 0).all() and priors.sum() > 0
        p *= priors[:, np.newaxis]
    p_total = p.sum(axis=0)
    mask = (p_total == 0) | (~np.isfinite(p_total)) | (~np.isfinite(p[0]))
    p_total[mask] = 1.0
    p_out = p[0] / p_total
    p_out[mask] = (1.0 / len(p)) if priors is None else (priors[0] / priors.sum())
    return p_out


def calc_gmm_satellite_probability(base, model_parameters, p_sat_prior=0.5, include_covariance=True):
    data, data_cov = get_input_data(base, include_covariance=include_covariance)
    model_params = (
        tuple(model_parameters[k] for k in param_labels_sat),
        tuple(model_parameters[k] for k in param_labels_nosat)
    )
    return calc_model1_prob(data, data_cov, model_params, [p_sat_prior, 1-p_sat_prior])

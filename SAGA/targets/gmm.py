"""
GMM related routines
"""
import numpy as np
from scipy.special import logsumexp
from ..utils import get_sdss_bands, get_sdss_colors, view_table_as_2d_array

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
    def __init__(self, n_components=1, initial_iter=20, **kwargs):
        if not _XDGMM_AVAILABLE:
            raise RuntimeError('You need to have both sklearn and extreme_deconvolution installed to use XDGMM')
        self._kwargs = kwargs
        self._gmm = GaussianMixture(int(n_components), max_iter=int(initial_iter), covariance_type='full')

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


def get_input_data(catalog, colors=None, color_errors=None, mag_errors=None,
                   include_covariance=True):
    if colors is None:
        colors = get_sdss_colors()
    if color_errors is None:
        color_errors = [c+'_err' for c in colors]
    if mag_errors is None:
        mag_errors = [b+'_err' for b in get_sdss_bands()[1:-1]]
    assert len(mag_errors) == len(colors) - 1
    X = view_table_as_2d_array(catalog, colors)
    Xcov = np.stack((np.diag(e*e) for e in view_table_as_2d_array(catalog, color_errors)))
    if include_covariance:
        Xcov -= np.stack(((np.diag(e*e, 1) + np.diag(e*e, -1)) for e in view_table_as_2d_array(catalog, mag_errors)))
    return X, Xcov


def check_calc_log_likelihood_input(data, data_cov, gmm_means, gmm_covs, gmm_weights):
    assert 1 == gmm_weights.ndim
    assert 2 == data.ndim == gmm_means.ndim
    assert 3 == data_cov.ndim == gmm_covs.ndim
    assert data.shape[0] == data_cov.shape[0]
    assert gmm_means.shape[0] == gmm_covs.shape[0] == gmm_weights.shape[0]
    assert data.shape[1] == data_cov.shape[1] == data_cov.shape[2] == gmm_means.shape[1] == gmm_covs.shape[1] == gmm_covs.shape[2]


def calc_log_likelihood(data, data_cov, gmm_means, gmm_covs, gmm_weights):
    if gmm_covs.ndim == 2:
        gmm_covs = np.stack((np.diag(c) for c in gmm_covs))
    check_calc_log_likelihood_input(data, data_cov, gmm_means, gmm_covs, gmm_weights)
    d = data[:, np.newaxis] - gmm_means
    cov = data_cov[:, np.newaxis] + gmm_covs
    tmp_result = np.einsum('...i,...ij,...j->...', d, np.linalg.inv(cov), d)
    tmp_result += np.log(np.fabs(np.linalg.det(cov)))
    tmp_result += (np.log(np.pi*2.0) * data.shape[-1])
    tmp_result *= -0.5
    return logsumexp(tmp_result, axis=1, b=gmm_weights)


def calc_model1_prob(data, data_cov, model_params, priors=None):
    p = np.stack((calc_log_likelihood(data, data_cov, *model_params_this) for model_params_this in model_params))
    if priors is not None:
        priors = np.asarray(priors)
        assert len(priors) == len(p) and (priors > 0).all()
    p = p[0] - logsumexp(p.T, axis=1, b=priors)
    if priors is not None:
        p += np.log(priors[0])
    p = np.exp(p, out=p)
    return p


def calc_gmm_satellite_probability(base, model_parameters, p_sat_prior=None,
                                   colors=None, color_errors=None, mag_errors=None,
                                   include_covariance=True):
    data, data_cov = get_input_data(base, colors=colors, color_errors=color_errors,
                                    mag_errors=mag_errors,
                                    include_covariance=include_covariance)
    model_params = (
        tuple(model_parameters[k] for k in param_labels_sat),
        tuple(model_parameters[k] for k in param_labels_nosat)
    )
    return calc_model1_prob(data,
                            data_cov,
                            model_params,
                            None if p_sat_prior is None else [p_sat_prior, 1-p_sat_prior])

"""Peak Event Coincidence Analysis (pECA).

This module contains functions to compute and estimate trigger
coincidence processes (TCPs) to study the association between event
series and peaks in a time series.

Example:
    timeseries is a real-valued np.ndarray of shape (T,), eventseries is
    a binary np.ndarray (with values 0 or 1 only) of the same shape.

        # Set time tolerance and sequence of thresholds.
        delta = 7
        taus = np.array([2.,3.,4.])

        # Estimate the parameters of the TCP under the independence
        # assumption.  NOTE: The parameters depend only on timeseries,
        # not on eventseries.
        tcp_params = peca.tcp_params_fit(timeseries, delta, taus)

        # Compute the observed trigger coincidence process (TCP).
        tcp_ = peca.tcp(timeseries, eventseries, delta, taus)

        # Compute the marginal p-values for the observed TCP under the
        # independence assumption.
        pvals = peca.tcp_marginal_pval(tcp_,
                                       eventseries.sum(),
                                       tcp_params)
        for (t, k, p) in zip(taus, tcp_, pvals):
            print(f"tau={t:.2f} k={k:.0f} p={p:.4f}")

        # Compute a Monte Carlo p-value for the complete TCP, using
        # the negative log-likelihood (NLL) as the test statistic.
        pval, nll = peca.tcp_nll_pval_shuffle(timeseries,
                                              eventseries,
                                              delta,
                                              taus)
        print(f"TCP {tcp_} nll={nll:.2f} p={pval:.4f}")

"""

__all__ = [
    "tcp", "tcp_params_fit", "tcp_marginal_expectation", "tcp_marginal_pval",
    "tcp_nll", "tcp_nll_pval_shuffle"
]

from typing import Tuple

import numba
import numpy as np
from scipy import stats

import matplotlib.pyplot as plt

TCPParamType = Tuple[np.ndarray, np.ndarray]


@numba.njit
def tcp(timeseries: np.ndarray, eventseries: np.ndarray, delta: int,
        taus: np.ndarray) -> np.ndarray:
    """Compute the TCP K_{tr}^{delta,taus}(E, X).

    Args:
        timeseries: The time series X.
        eventseries: The event series E.
        delta: Time tolerance.
        taus: Thresholds (strictly increasing).

    Returns:
        The trigger coincidence process.

    """
    length = min(len(timeseries), len(eventseries))
    tcp_ = np.zeros_like(taus)
    for i, tau in enumerate(taus):
        tes = (timeseries > tau) * 1  # threshold exceedance series
        tcp_[i] = len([
            t for t in range(length - delta)
            if (eventseries[t] == 1) and np.sum(tes[t:(t + delta + 1)]) >= 1
        ])
    return tcp_


def _fit_gev_blockmaxima(timeseries: np.ndarray, blocksize: int) -> Tuple:
    """Fit GEV parameters to block maxima of timeseries.

    timeseries is first split into blocks of size blocksize and the GEV
    distribution is fitted to the maxima of these blocks. The last block
    is ignored if it is smaller than blocksize.

    Args:
        timeseries: The time series X.
        blocksize: Size of the blocks. ;)

    Returns:
        All GEV parameters as returned by scipy.stats.genextreme.fit().

    """
    length = len(timeseries) - (len(timeseries) % blocksize)
    blockmaxima = np.array([
        timeseries[t:(t + blocksize)].max()
        for t in range(0, length, blocksize)
    ])
    return stats.genextreme.fit(blockmaxima)


def plot_gev_diagnostics(timeseries, blocksize, q_crop_percentile=10, thresh_marker=None, title=None):
    if thresh_marker is None:
        thresh_marker = []

    # fit GEV distribution to block maxima
    length = len(timeseries) - (len(timeseries) % blocksize)
    blockmaxima = np.array([
        timeseries[t:(t + blocksize)].max()
        for t in range(0, length, blocksize)
    ])
    gev_params = stats.genextreme.fit(blockmaxima)

    plt.figure(figsize=(8,3))
    if title is not None:
        plt.suptitle(title)

    # P-P plot
    plt.subplot(131)
    plt.axline((0, 0), (1, 1), lw=1, c='k', ls='--')
    p_est = stats.genextreme.cdf(np.sort(blockmaxima), *gev_params)
    p_emp = np.cumsum(np.full(len(blockmaxima), 1./(len(blockmaxima)+1)))
    plt.scatter(p_est, p_emp, c='orange', marker='.')
    plt.xlabel('estimate')
    plt.ylabel('empirical')
    plt.xticks(np.arange(0, 1.2, 0.2))
    plt.yticks(np.arange(0, 1.2, 0.2))
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title('P-P plot')
    plt.tight_layout()

    # Q-Q plot
    plt.subplot(132)
    q_est = stats.genextreme.ppf(np.cumsum(np.full(len(blockmaxima), 1./(len(blockmaxima)+1))), *gev_params)
    q_emp = np.sort(blockmaxima)
    q_min = min(q_est.min(), q_emp.min())
    q_max = min(q_est.max(), q_emp.max())
    q_crop_idx = int(q_crop_percentile/100.*len(blockmaxima))
    for thresh in thresh_marker:
        plt.axhline(thresh, lw=1, color='k', alpha=0.1)
    plt.axline((q_min, q_min), (q_max, q_max), lw=1, c='k', ls='--')
    plt.scatter(q_est, q_emp, marker='.')
    plt.gca().add_patch(plt.Rectangle(
        (q_est[q_crop_idx], q_emp[q_crop_idx]),
        q_est[-q_crop_idx] - q_est[q_crop_idx],
        q_emp[-q_crop_idx] - q_emp[q_crop_idx],
        fill=False
    ))
    plt.xlabel('estimate')
    plt.ylabel('empirical')
    plt.title('Q-Q plot')
    plt.tight_layout()

    # Q-Q plot (crop)
    plt.subplot(133)
    q_est_crop = q_est[q_crop_idx:-q_crop_idx]
    q_emp_crop = q_emp[q_crop_idx:-q_crop_idx]
    q_min_crop = min(q_est_crop.min(), q_emp_crop.min())
    q_max_crop = min(q_est_crop.max(), q_emp_crop.max())
    for thresh in thresh_marker:
        if (thresh > q_min_crop) and (thresh < q_max_crop):
            plt.axhline(thresh, lw=1, color='k', alpha=0.1)
    plt.axline((q_min_crop, q_min_crop), (q_max_crop, q_max_crop), lw=1, c='k', ls='--')
    plt.scatter(q_est_crop, q_emp_crop, marker='.')
    plt.xlabel('estimate')
    plt.ylabel('empirical')
    plt.title(f'Q-Q plot ({100-q_crop_percentile*2:.0f}% crop)')
    plt.tight_layout()

    plt.show()


def tcp_params_fit(timeseries: np.ndarray, delta: int,
                   taus: np.ndarray) -> TCPParamType:
    """Fit the parameters of the TCP Markov model to timeseries.

    The Markov model has two sets of parameters: the marginal
    probabilities P(K_{tr}^{delta,tau_{i}}) and the conditional
    probabilities P(K_{tr}^{delta,tau_{i}} | K_{tr}^{delta,tau_{i-1}}).

    Args:
        timeseries: The time series X.
        delta: Time tolerance.
        taus: Thresholds (strictly increasing).

    Returns:
        Tuple with all TCP parameters (marginals and conditionals).

    """
    gev_params = _fit_gev_blockmaxima(timeseries, delta + 1)
    ps_marginal = np.array(
        [1. - stats.genextreme.cdf(tau, *gev_params) for tau in taus])
    ps_conditional = np.ones_like(taus) * np.nan
    ps_conditional[1:] = np.array(
        [ps_marginal[i] / ps_marginal[i - 1] for i in range(1, len(taus))])
    tcp_params = (ps_marginal, ps_conditional)
    return tcp_params


def tcp_marginal_expectation(n_events: int,
                             tcp_params: TCPParamType) -> np.ndarray:
    """Compute the marginally expected TCP for independent event series.

    The marginally expected TCP contains all pointwise expected values
    for each threshold.

    Args:
        n_events: Number of events in the independent event series.
        tcp_params: Tuple with all TCP parameters.

    Returns:
        The marginal expectations.

    """
    return tcp_params[0] * n_events


def tcp_marginal_pval(tcp_: np.ndarray, n_events: int,
                      tcp_params: TCPParamType) -> np.ndarray:
    """Compute marginal p-values for the TCP under independence.

    Args:
        tcp_: Observed trigger coincidence process.
        n_events: Number of events in the event series.
        tcp_params: Tuple with all TCP parameters.

    Returns:
        The marginal p-values.

    """
    return (stats.binom.pmf(tcp_, n_events, tcp_params[0]) +
            stats.binom.sf(tcp_, n_events, tcp_params[0]))


def tcp_nll(tcp_: np.ndarray,
            n_events: int,
            tcp_params: TCPParamType,
            idx_start: int = 0) -> float:
    """Compute the negative log-likelihood for the TCP under independence.

    The TCP can be evaluated only at higher thresholds by setting
    idx_start > 0.

    Args:
        tcp_: Observed trigger coincidence process.
        n_events: Number of events in the event series.
        tcp_params: Tuple with all TCP parameters.
        idx_start: Index of the first threshold to evaluate.
                   Defaults to 0.

    Returns:
        The negative log-likelihood.

    """
    ps_marginal, ps_conditional = tcp_params
    return -(stats.binom.logpmf(tcp_[idx_start], n_events,
                                ps_marginal[idx_start])
             + np.sum([
                 stats.binom.logpmf(tcp_[i], tcp_[i - 1], ps_conditional[i])
                 for i in range(idx_start + 1, len(ps_marginal))
             ]))


def tcp_nll_pval_shuffle(timeseries: np.ndarray,
                         eventseries: np.ndarray,
                         delta: int,
                         taus: np.ndarray,
                         samples: int = 10000,
                         idx_start: int = 0) -> Tuple[float, float]:
    """Compute a Monte Carlo p-value for the TCP under independence.

    The p-value is computed from random permutations of the event series
    using the NLL as the test statistic.

    Args:
        timeseries: The time series X.
        eventseries: The event series E.
        delta: Time tolerance.
        taus: Thresholds (strictly increasing).
        samples: Number of Monte Carlo samples.
        idx_start: Index of the first threshold to evaluate.
                   Defaults to 0.

    Returns:
        Tuple with the p-value and the test statistic value.

    """
    n_events = eventseries.sum()
    tcp_params = tcp_params_fit(timeseries, delta, taus)
    nll = tcp_nll(tcp(timeseries, eventseries, delta, taus), n_events,
                  tcp_params, idx_start)
    greater_or_equal = 0
    for _ in range(samples):
        simul_eventseries = np.random.permutation(eventseries)
        simul_nll = tcp_nll(tcp(timeseries, simul_eventseries, delta, taus),
                            n_events, tcp_params, idx_start)
        greater_or_equal += (simul_nll >= nll)
    pval = (greater_or_equal + 1) / (samples + 1)
    return pval, nll

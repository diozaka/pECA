"""Peak Event Coincidence Analysis (pECA).

This module contains functions to compute and estimate trigger coincidence processes (TCPs)
to study the association between event series and peaks in a time series.

Example:
    The time series X is a real-valued np.ndarray of shape (T,), the event series is binary
    np.ndarray (with values 0 or 1 only) of the same shape.

        # set time tolerance and sequence of thresholds
        delta = 7
        taus = np.array([2.,3.,4.])

        # compute the observed trigger coincidence process (TCP)
        K_tr = tcp(X, E, delta, taus)

        # estimate the parameters of the TCP under the independence assumption
        tcp_params = tcp_params_fit(X, delta, taus)

        # compute the p-value for the observed TCP under the independence assumption
        pval = tcp_marginal_pval(K_tr, E.sum(), tcp_params)

"""

from typing import Tuple
from numba import njit
import numpy as np
import scipy.stats as ss

TCPParamType = Tuple[np.ndarray, np.ndarray]

@njit
def tcp(X: np.ndarray, E: np.ndarray, delta: int, taus: np.ndarray) -> np.ndarray:
    """Compute the TCP K_{tr}^{delta,taus}(E, X).

    Args:
        X: Time series.
        E: Event series.
        delta: Time tolerance.
        taus: Thresholds (strictly increasing).

    Returns:
        The trigger coincidence process.

    """
    T = min(len(X), len(E))
    K_tr = np.zeros_like(taus)
    for i, tau in enumerate(taus):
        A = (X > tau)*1
        K_tr[i] = len([t for t in range(T-delta) if (E[t] == 1) and np.sum(A[t:t+delta+1]) >= 1])
    return K_tr

def _fit_gev_blockmaxima(X: np.ndarray, blocksize: int) -> Tuple:
    """Fit the parameters of the GEV distribution to block maxima of X.

    X is first split into blocks of size blocksize and the GEV distribution
    is fitted to the maxima of these blocks.

    Args:
        X: Time series.
        blocksize: Size of the blocks. ;)

    Returns:
        All GEV parameters as returned by ss.genextreme.fit().

    """
    T = len(X) - (len(X)%blocksize) # ignore remainder
    Mk = np.array([X[t:t+blocksize].max() for t in range(0, T, blocksize)])
    gev_params = ss.genextreme.fit(Mk)
    return gev_params

def tcp_params_fit(X: np.ndarray, delta: int, taus: np.ndarray) -> TCPParamType:
    """Fit the parameters of the TCP Markov model to X for the given taus and delta.

    The Markov model has two sets of parameters: the marginal probabilities P(K_{tr}^{delta,tau})
    and the conditional probabilities P(K_{tr}^{delta,tau_{i}} | K_{tr}^{delta,tau_{i-1}}).

    Args:
        X: Time series.
        delta: Time tolerance.
        taus: Thresholds (strictly increasing).

    Returns:
        Tuple with all TCP parameters (marginal and conditional probabilities).

    """
    gev_params = _fit_gev_blockmaxima(X, delta+1)
    ps_marginal = np.array([1.-ss.genextreme.cdf(tau, *gev_params) for tau in taus])
    ps_conditional = np.ones_like(taus)*np.nan
    ps_conditional[1:] = np.array([ps_marginal[i]/ps_marginal[i-1] for i in range(1,len(taus))])
    tcp_params = (ps_marginal, ps_conditional)
    return tcp_params

def tcp_marginal_expectation(N_E: int, tcp_params: TCPParamType) -> np.ndarray:
    """Compute the marginally expected TCP for an independent event series with N_E events.

    The marginally expected TCP contains all pointwise expected values for each threshold.

    Args:
        N_E: Number of events in the event series.
        tcp_params: Tuple with all TCP parameters

    Returns:
        The marginal expectations.

    """
    return tcp_params[0]*N_E

def tcp_marginal_pval(K_tr: np.ndarray, N_E: int, tcp_params: TCPParamType) -> np.ndarray:
    """Compute the marginal p-values for the TCP K_tr under the independence assumption.

    Args:
        K_tr: Observed trigger coincidence process.
        N_E: Number of events in the event series.
        tcp_params: Tuple with all TCP parameters.

    Returns:
        The marginal p-values.

    """
    return ss.binom.pmf(K_tr, N_E, tcp_params[0]) + ss.binom.sf(K_tr, N_E, tcp_params[0])

def tcp_nll(K_tr: np.ndarray, N_E: int, tcp_params: TCPParamType, idx_start: int = 0) -> float:
    """Compute the negative log-likelihood for the TCP K_tr under the independence assumption.

    The TCP can be evaluated only at higher thresholds by setting idx_start > 0.

    Args:
        K_tr: Observed trigger coincidence process.
        N_E: Number of events in the event series.
        tcp_params: Tuple with all TCP parameters.
        idx_start: Index of the first threshold to evaluate. Defaults to 0.

    Returns:
        The negative log-likelihood.

    """
    ps_marginal, ps_conditional  = tcp_params
    return -(ss.binom.logpmf(K_tr[idx_start], N_E, ps_marginal[idx_start])
       + np.sum([ss.binom.logpmf(K_tr[i], K_tr[i-1], ps_conditional[i])
                     for i in range(idx_start+1, len(ps_marginal))]))


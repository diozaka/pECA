"""Peak Event Coincidence Analysis (pECA).

Contains functions to compute and estimate trigger coincidence processes (TCPs) to
study the association between event series and peaks in a time series.
"""

from numba import njit
import numpy as np
import scipy.stats as ss

@njit
def tcp(X, E, delta, taus):
    """Compute the TCP K_{tr}^{delta,taus}(E, X) and return as np.array."""
    T = min(len(X), len(E))
    K_tr = np.zeros_like(taus)
    for i, tau in enumerate(taus):
        A = (X > tau)*1
        K_tr[i] = len([t for t in range(T-delta) if (E[t] == 1) and np.sum(A[t:t+delta+1]) >= 1])
    return K_tr

#def trigger_coincidences_pval(K_tr, N_E, pi):
#    """Compute the p-value for a single number of trigger coincidences K_tr."""
#    return ss.binom.pmf(K_tr, N_E, pi) + ss.binom.sf(K_tr, N_E, pi)

def _fit_gev_blockmaxima(X, blocksize):
    """Fit the parameters of the GEV distribution to block maxima of X.

    X is first split into blocks of size blocksize and the GEV distribution
    is fitted to the maxima of these blocks.
    """
    T = len(X) - (len(X)%blocksize) # ignore remainder
    Mk = np.array([X[t:t+blocksize].max() for t in range(0, T, blocksize)])
    gev_params = ss.genextreme.fit(Mk)
    return gev_params

def tcp_params_fit(X, delta, taus):
    """Fit the parameters of the TCP Markov model to X for the given taus and delta.

    The Markov model has two sets of parameters: the marginal probabilities P(K_{tr}^{delta,tau})
    and the conditional probabilities P(K_{tr}^{delta,tau_{i}} | K_{tr}^{delta,tau_{i-1}}).
    """
    gev_params = _fit_gev_blockmaxima(X, delta+1)
    ps_marginal = np.array([1.-ss.genextreme.cdf(tau, *gev_params) for tau in taus])
    ps_conditional = np.ones_like(taus)*np.nan
    ps_conditional[1:] = np.array([ps_marginal[i]/ps_marginal[i-1] for i in range(1,len(taus))])
    tcp_params = (ps_marginal, ps_conditional)
    return tcp_params

def tcp_marginal_expectation(N_E, tcp_params):
    """Compute the marginally expected TCP for an independent event series with N_E events.

    The marginally expected TCP contains all pointwise expected values for each threshold.
    """
    return tcp_params[0]*N_E

def tcp_marginal_pval(K_tr, N_E, tcp_params):
    """Compute the marginal p-values for the TCP K_tr under the independence assumption."""
    return ss.binom.pmf(K_tr, N_E, tcp_params[0]) + ss.binom.sf(K_tr, N_E, tcp_params[0])

def tcp_nll(K_tr, N_E, tcp_params, idx_start=0):
    """Compute the negative log-likelihood for the TCP K_tr under the independence assumption.

    N_E denotes the total number of event occurrences in the event series E and tcp_params must
    be fitted to the time series X beforehand. The TCP can be evaluated only at higher thresholds
    by setting idx_start > 0.
    """
    ps_marginal, ps_conditional  = tcp_params
    return -(ss.binom.logpmf(K_tr[idx_start], N_E, ps_marginal[idx_start])
       + np.sum([ss.binom.logpmf(K_tr[i], K_tr[i-1], ps_conditional[i])
                     for i in range(idx_start+1, len(ps_marginal))]))


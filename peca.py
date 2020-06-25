from numba import njit
import numpy as np
import scipy.stats as ss

# TCP = trigger coincidence process

@njit
def trigger_coincidences(X, E, taus, delta):
    # Compute the TCP K_{tr}^{delta,taus}(E, X) for time series 'X' and
    # event series 'E' using the sequence of thresholds 'taus' with a
    # time tolerance of 'delta'
    T = min(len(X), len(E))
    K_tr = np.zeros_like(taus)
    for i, tau in enumerate(taus):
        A = (X > tau)*1
        K_tr[i] = len([t for t in range(T-delta) if (E[t] == 1) and np.sum(A[t:t+delta+1]) >= 1])
    return K_tr

def _fit_gev_blockmaxima(X, blocksize):
    # Fit the parameters of the GEV distribution to the maxima of blocks
    # of size 'blocksize' in the time series 'X'
    T = len(X) - (len(X)%blocksize) # ignore remainder
    Mk = np.array([X[t:t+blocksize].max() for t in range(0, T, blocksize)])
    gev_params = ss.genextreme.fit(Mk)
    return gev_params

def tcp_params_fit(X, delta, taus):
    # Fit the parameters of the TCP Markov model (marginal and conditional probabilities)
    # for the time series 'X', with time tolerance 'delta' and threshold sequence 'taus'
    gev_params = _fit_gev_blockmaxima(X, delta+1)
    ps_marginal = np.array([1.-ss.genextreme.cdf(tau, *gev_params) for tau in taus])
    ps_conditional = np.ones_like(taus)*np.nan
    ps_conditional[1:] = np.array([ps_marginal[i]/ps_marginal[i-1] for i in range(1,len(taus))])
    return ps_marginal, ps_conditional

def tcp_marginal_expectation(N_E, tcp_params):
    # Compute the marginally expected TCP, i.e., the TCP that is obtained when taking the
    # pointwise expected values independently for each threshold
    return tcp_params[0]*N_E

def tcp_nll(K_tr, N_E, idx_start, tcp_params):
    # Compute the negative log-likelihood (test statistic value s) for the observed TCP 'K_tr',
    # when the event series has 'N_E' event occurrences. Use threshold at position 'idx_start'
    # as the first threshold (to shift attention to higher quantiles, default: 0), and
    # the TCP model parameters from 'tcp_params'
    ps_marginal, ps_conditional  = tcp_params
    return -(ss.binom.logpmf(K_tr[idx_start], N_E, ps_marginal[idx_start])
       + np.sum([ss.binom.logpmf(K_tr[i], K_tr[i-1], ps_conditional[i])
                     for i in range(idx_start+1, len(ps_marginal))]))

def trigger_coincidences_pval(K_tr, N_E, pi):
    return ss.binom.pmf(K_tr, N_E, pi) + ss.binom.sf(K_tr, N_E, pi)
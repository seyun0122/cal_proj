"""Monte Carlo, Quasi-Monte Carlo, MCMC 적분법."""

import numpy as np


def monte_carlo(func, a, b, n, seed=None):
    rng = np.random.default_rng(seed)
    x = rng.uniform(a, b, int(n))
    return float((b - a) * np.mean(func(x)))


def monte_carlo_nd(func_nd, a, b, dim, n, seed=None):
    rng = np.random.default_rng(seed)
    X = rng.uniform(a, b, (int(n), int(dim)))
    return float((b - a) ** dim * np.mean(func_nd(X)))


def monte_carlo_stats(func, a, b, n, repeat=30, seed=None):
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(int(repeat)):
        s = int(rng.integers(0, 2**31 - 1))
        estimates.append(monte_carlo(func, a, b, n, seed=s))
    return {
        "mean_estimate": float(np.mean(estimates)),
        "std_estimate": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
        "repeat": int(repeat),
    }


def _van_der_corput(n, base):
    seq = np.zeros(int(n), dtype=float)
    for i in range(int(n)):
        f = 1.0
        r = 0.0
        k = i + 1
        while k > 0:
            f /= base
            r += f * (k % base)
            k //= base
        seq[i] = r
    return seq


def _halton_nd(n, dim):
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]
    if dim > len(primes):
        raise ValueError(f"Halton 수열은 현재 {len(primes)}차원까지만 지원한다.")
    cols = [_van_der_corput(n, primes[j]) for j in range(dim)]
    return np.stack(cols, axis=1)


def quasi_mc(func, a, b, n):
    u = _van_der_corput(int(n), 2)
    x = a + (b - a) * u
    return float((b - a) * np.mean(func(x)))


def quasi_mc_nd(func_nd, a, b, dim, n):
    U = _halton_nd(int(n), int(dim))
    X = a + (b - a) * U
    return float((b - a) ** dim * np.mean(func_nd(X)))


def lag1_autocorr(values):
    x = np.asarray(values, dtype=float)
    if x.size < 3:
        return np.nan
    x0 = x[:-1] - np.mean(x[:-1])
    x1 = x[1:] - np.mean(x[1:])
    denom = np.sqrt(np.dot(x0, x0) * np.dot(x1, x1))
    if denom == 0:
        return 0.0
    return float(np.dot(x0, x1) / denom)


def ess_from_lag1(n, rho1):
    if not np.isfinite(rho1):
        return np.nan
    rho = max(min(float(rho1), 0.999), -0.999)
    if rho <= 0:
        return float(n)
    return float(n * (1.0 - rho) / (1.0 + rho))


def mcmc_sample_1d(a, b, n, burn_in=1000, proposal_std=None, seed=None):
    rng = np.random.default_rng(seed)
    if proposal_std is None:
        proposal_std = (b - a) * 0.2
    x_curr = rng.uniform(a, b)
    samples = np.empty(int(n) + int(burn_in), dtype=float)
    accepted = 0
    total = samples.size
    for i in range(total):
        x_prop = x_curr + rng.normal(0.0, proposal_std)
        if a <= x_prop <= b:
            x_curr = x_prop
            accepted += 1
        samples[i] = x_curr
    return samples[int(burn_in):], accepted / total


def mcmc_integrate(func, a, b, n, burn_in=1000, proposal_std=None, seed=None):
    samples, _ = mcmc_sample_1d(a, b, n, burn_in=burn_in, proposal_std=proposal_std, seed=seed)
    return float((b - a) * np.mean(func(samples)))


def mcmc_stats(func, a, b, n, repeat=30, burn_in=1000, proposal_std=None, seed=None):
    rng = np.random.default_rng(seed)
    estimates = []
    accept_rates = []
    lag1_values = []
    ess_values = []
    for _ in range(int(repeat)):
        s = int(rng.integers(0, 2**31 - 1))
        samples, acc = mcmc_sample_1d(a, b, n, burn_in=burn_in, proposal_std=proposal_std, seed=s)
        f_samples = func(samples)
        est = (b - a) * np.mean(f_samples)
        rho1 = lag1_autocorr(f_samples)
        estimates.append(est)
        accept_rates.append(acc)
        lag1_values.append(rho1)
        ess_values.append(ess_from_lag1(int(n), rho1))
    return {
        "mean_estimate": float(np.mean(estimates)),
        "std_estimate": float(np.std(estimates, ddof=1)) if len(estimates) > 1 else 0.0,
        "accept_rate": float(np.nanmean(accept_rates)),
        "lag1_autocorr": float(np.nanmean(lag1_values)),
        "ess_lag1": float(np.nanmean(ess_values)),
        "repeat": int(repeat),
    }


def mcmc_integrate_nd(func_nd, a, b, dim, n, burn_in=1000, proposal_std=None, seed=None):
    rng = np.random.default_rng(seed)
    if proposal_std is None:
        proposal_std = (b - a) * 0.2
    x_curr = rng.uniform(a, b, int(dim))
    samples = np.empty((int(n), int(dim)), dtype=float)
    filled = 0
    for i in range(int(n) + int(burn_in)):
        x_prop = x_curr + rng.normal(0.0, proposal_std, int(dim))
        if np.all((a <= x_prop) & (x_prop <= b)):
            x_curr = x_prop
        if i >= burn_in:
            samples[filled] = x_curr
            filled += 1
    return float((b - a) ** dim * np.mean(func_nd(samples)))


def integrate_mc_1d(method, func, a, b, n, seed=None):
    if method == "monte_carlo":
        return monte_carlo(func, a, b, n, seed=seed)
    if method == "quasi_mc":
        return quasi_mc(func, a, b, n)
    if method == "mcmc":
        return mcmc_integrate(func, a, b, n, seed=seed)
    raise ValueError(f"알 수 없는 방법: {method}")


def integrate_mc_nd(method, func_nd, a, b, dim, n, seed=None):
    if method == "monte_carlo":
        return monte_carlo_nd(func_nd, a, b, dim, n, seed=seed)
    if method == "quasi_mc":
        return quasi_mc_nd(func_nd, a, b, dim, n)
    if method == "mcmc":
        return mcmc_integrate_nd(func_nd, a, b, dim, n, seed=seed)
    raise ValueError(f"알 수 없는 방법: {method}")

"""격자 기반 수치적분법."""

import numpy as np
from numpy.polynomial.legendre import leggauss


def midpoint(func, a, b, n):
    h = (b - a) / n
    x = a + h * (np.arange(n) + 0.5)
    return float(h * np.sum(func(x)))


def trapezoidal(func, a, b, n):
    x = np.linspace(a, b, n + 1)
    y = func(x)
    return float((b - a) / (2.0 * n) * (y[0] + 2.0 * np.sum(y[1:-1]) + y[-1]))


def simpson(func, a, b, n):
    if n % 2 != 0:
        n += 1
    x = np.linspace(a, b, n + 1)
    y = func(x)
    coeff = np.ones(n + 1)
    coeff[1:-1:2] = 4.0
    coeff[2:-2:2] = 2.0
    return float((b - a) / (3.0 * n) * np.dot(coeff, y))


def gauss(func, a, b, n):
    nodes, weights = leggauss(n)
    x = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    return float(0.5 * (b - a) * np.dot(weights, func(x)))


def method_eval_count_1d(method, n):
    if method in ("midpoint", "gauss"):
        return int(n)
    if method in ("trapezoidal", "simpson"):
        return int(n) + 1
    raise ValueError(method)


def n_for_eval_budget(method, budget):
    """함수 평가 횟수가 budget을 넘지 않도록 n을 고른다."""
    budget = int(budget)
    if method in ("midpoint", "gauss"):
        return max(1, budget)
    if method == "trapezoidal":
        return max(1, budget - 1)
    if method == "simpson":
        n = max(2, budget - 1)
        if n % 2 != 0:
            n -= 1
        return max(2, n)
    raise ValueError(method)


def midpoint_nd(func_nd, a, b, dim, n):
    h = (b - a) / n
    grid_1d = a + h * (np.arange(n) + 0.5)
    grids = np.meshgrid(*([grid_1d] * dim), indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)
    return float((h ** dim) * np.sum(func_nd(points)))


def trapezoidal_nd(func_nd, a, b, dim, n):
    x_1d = np.linspace(a, b, n + 1)
    h = (b - a) / n
    grids = np.meshgrid(*([x_1d] * dim), indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)
    w_1d = np.ones(n + 1)
    w_1d[1:-1] = 2.0
    w_grids = np.meshgrid(*([w_1d] * dim), indexing="ij")
    weights = np.ones_like(w_grids[0])
    for wg in w_grids:
        weights *= wg
    return float(((h / 2.0) ** dim) * np.dot(weights.ravel(), func_nd(points)))


def simpson_nd(func_nd, a, b, dim, n):
    if n % 2 != 0:
        n += 1
    x_1d = np.linspace(a, b, n + 1)
    h = (b - a) / n
    grids = np.meshgrid(*([x_1d] * dim), indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)
    c = np.ones(n + 1)
    c[1:-1:2] = 4.0
    c[2:-2:2] = 2.0
    c_grids = np.meshgrid(*([c] * dim), indexing="ij")
    weights = np.ones_like(c_grids[0])
    for cg in c_grids:
        weights *= cg
    return float(((h / 3.0) ** dim) * np.dot(weights.ravel(), func_nd(points)))


def gauss_nd(func_nd, a, b, dim, n):
    nodes, weights = leggauss(n)
    x_1d = 0.5 * (b - a) * nodes + 0.5 * (a + b)
    w_1d = 0.5 * (b - a) * weights
    grids = np.meshgrid(*([x_1d] * dim), indexing="ij")
    w_grids = np.meshgrid(*([w_1d] * dim), indexing="ij")
    points = np.stack([g.ravel() for g in grids], axis=1)
    w = np.ones_like(w_grids[0])
    for wg in w_grids:
        w *= wg
    return float(np.dot(w.ravel(), func_nd(points)))


def method_eval_count_nd(method, dim, n):
    if method in ("midpoint", "gauss"):
        return int(n) ** int(dim)
    if method in ("trapezoidal", "simpson"):
        return (int(n) + 1) ** int(dim)
    raise ValueError(method)


_1D = {"midpoint": midpoint, "trapezoidal": trapezoidal, "simpson": simpson, "gauss": gauss}
_ND = {"midpoint": midpoint_nd, "trapezoidal": trapezoidal_nd, "simpson": simpson_nd, "gauss": gauss_nd}


def integrate_1d(method, func, a, b, n):
    if method not in _1D:
        raise ValueError(f"알 수 없는 방법: {method}")
    return _1D[method](func, a, b, n)


def integrate_nd(method, func_nd, a, b, dim, n):
    if method not in _ND:
        raise ValueError(f"알 수 없는 방법: {method}")
    return _ND[method](func_nd, a, b, dim, n)

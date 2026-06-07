"""테스트 함수와 참값 정의."""

import numpy as np
from scipy.special import erf


def _sin(x):
    return np.sin(x)


def _sin_exact(a, b):
    return -np.cos(b) + np.cos(a)


def _sin10x(x):
    return np.sin(10.0 * x)


def _sin10x_exact(a, b):
    return (-np.cos(10.0 * b) + np.cos(10.0 * a)) / 10.0


def _gaussian_peak(x):
    return np.exp(-100.0 * (x - 0.5) ** 2)


def _gaussian_peak_exact(a, b):
    c = 10.0
    return (np.sqrt(np.pi) / (2.0 * c)) * (
        erf(c * (b - 0.5)) - erf(c * (a - 0.5))
    )


def _runge(x):
    return 1.0 / (1.0 + 25.0 * x ** 2)


def _runge_exact(a, b):
    return (np.arctan(5.0 * b) - np.arctan(5.0 * a)) / 5.0


def make_step(threshold=0.5):
    """f(x)=1_{x>=threshold} 형태의 step 함수를 만든다."""
    threshold = float(threshold)

    def func(x):
        return np.where(x < threshold, 0.0, 1.0)

    def exact(a, b):
        left = max(float(a), threshold)
        right = float(b)
        return max(0.0, right - left)

    return func, exact


def _step(x):
    func, _ = make_step(0.5)
    return func(x)


def _step_exact(a, b):
    _, exact = make_step(0.5)
    return exact(a, b)


def _steep_sigmoid(x):
    k = 80.0
    c = 0.37
    z = np.clip(-k * (x - c), -700.0, 700.0)
    return 1.0 / (1.0 + np.exp(z))


def _steep_sigmoid_exact(a, b):
    k = 80.0
    c = 0.37
    return (
        np.logaddexp(0.0, k * (b - c))
        - np.logaddexp(0.0, k * (a - c))
    ) / k


def _exp_product_nd(X):
    return np.exp(X.sum(axis=1))


def _exp_product_exact_nd(a, b, dim):
    return (np.exp(b) - np.exp(a)) ** dim


def _sum_of_squares_nd(X):
    return (X ** 2).sum(axis=1)


def _sum_of_squares_exact_nd(a, b, dim):
    return dim * ((b ** 3 - a ** 3) / 3.0) * ((b - a) ** (dim - 1))


FUNCTIONS = {
    "sin": {
        "func": _sin,
        "func_nd": None,
        "exact": _sin_exact,
        "exact_nd": None,
        "bounds": (0.0, np.pi),
        "label": r"$\sin x$",
        "kind": "smooth",
    },
    "sin10x": {
        "func": _sin10x,
        "func_nd": None,
        "exact": _sin10x_exact,
        "exact_nd": None,
        "bounds": (0.0, np.pi / 2.0),
        "label": r"$\sin(10x)$",
        "kind": "oscillatory",
    },
    "gaussian_peak": {
        "func": _gaussian_peak,
        "func_nd": None,
        "exact": _gaussian_peak_exact,
        "exact_nd": None,
        "bounds": (0.0, 1.0),
        "label": r"$e^{-100(x-0.5)^2}$",
        "kind": "peaked",
    },
    "runge": {
        "func": _runge,
        "func_nd": None,
        "exact": _runge_exact,
        "exact_nd": None,
        "bounds": (-1.0, 1.0),
        "label": r"$1/(1+25x^2)$",
        "kind": "peaked",
    },
    "step": {
        "func": _step,
        "func_nd": None,
        "exact": _step_exact,
        "exact_nd": None,
        "bounds": (0.0, 1.0),
        "label": "step function",
        "kind": "discontinuous",
    },
    "steep_sigmoid": {
        "func": _steep_sigmoid,
        "func_nd": None,
        "exact": _steep_sigmoid_exact,
        "exact_nd": None,
        "bounds": (0.0, 1.0),
        "label": r"$(1+e^{-80(x-0.37)})^{-1}$",
        "kind": "steep",
    },
    "exp_product": {
        "func": None,
        "func_nd": _exp_product_nd,
        "exact": None,
        "exact_nd": _exp_product_exact_nd,
        "bounds": (0.0, 1.0),
        "label": r"$\exp(x_1+\cdots+x_d)$",
        "kind": "highdim",
    },
    "sum_of_squares": {
        "func": None,
        "func_nd": _sum_of_squares_nd,
        "exact": None,
        "exact_nd": _sum_of_squares_exact_nd,
        "bounds": (0.0, 1.0),
        "label": r"$x_1^2+\cdots+x_d^2$",
        "kind": "highdim",
    },
}


def get_function(name):
    if name not in FUNCTIONS:
        raise KeyError(f"알 수 없는 함수 이름: {name}")
    return FUNCTIONS[name]

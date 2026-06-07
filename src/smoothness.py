"""함수의 수치적 거칠기 지표 계산."""

import numpy as np


def second_derivative_max(func, a, b, n=10000):
    x = np.linspace(a, b, int(n))
    h = x[1] - x[0]
    y = func(x)
    d2 = (y[:-2] - 2.0 * y[1:-1] + y[2:]) / (h ** 2)
    return float(np.max(np.abs(d2)))


def function_variance(func, a, b, n=10000):
    x = np.linspace(a, b, int(n))
    return float(np.var(func(x)))


def discrete_total_variation(func, a, b, n=10000):
    x = np.linspace(a, b, int(n))
    return float(np.sum(np.abs(np.diff(func(x)))))


def classify_roughness(max_second_deriv, total_variation):
    """보고서에 기준을 명시할 수 있도록 단순 기준을 함께 반환한다."""
    if max_second_deriv < 10 and total_variation < 5:
        return "smooth"
    if max_second_deriv < 100 and total_variation < 20:
        return "moderate"
    return "rough"


def analyze(func, a, b, name=""):
    d2 = second_derivative_max(func, a, b)
    var = function_variance(func, a, b)
    tv = discrete_total_variation(func, a, b)
    return {
        "name": name,
        "max_second_deriv_numerical": d2,
        "variance_uniform_grid": var,
        "discrete_total_variation": tv,
        "roughness_class": classify_roughness(d2, tv),
        "classification_rule": "smooth: max_d2<10 and TV<5, moderate: max_d2<100 and TV<20, otherwise rough",
        "note": "불연속 함수의 max_second_deriv_numerical은 실제 도함수가 아니라 중앙차분 폭발을 나타내는 지표이다.",
    }

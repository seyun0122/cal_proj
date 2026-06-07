"""결과 시각화."""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FuncFormatter

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    FIGURES_DIR,
    TABLES_DIR,
    FIGURE_DPI,
    FIGURE_FORMAT,
    FUNCTIONS_1D,
    FUNCTIONS_HD,
    QUADRATURE_METHODS,
    MC_METHODS,
    LOG_ERROR_FLOOR,
)
from functions import get_function


def plain_log_formatter(y, pos):
    if y <= 0:
        return ""

    exp = int(round(np.log10(y)))

    # 10의 거듭제곱인 눈금만 표시
    if abs(y / (10 ** exp) - 1) < 1e-8:
        return f"1e{exp}"

    return ""


def apply_plain_log_ticks(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_log_formatter))


def _savefig(fig, name):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, f"{name}.{FIGURE_FORMAT}")
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    print(f"저장: {path}")
    plt.close(fig)


def _positive_error(values):
    return np.maximum(np.asarray(values, dtype=float), LOG_ERROR_FLOOR)


def plot_function_shapes():
    cols = 3
    rows = int(np.ceil(len(FUNCTIONS_1D) / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12, 7))
    axes = np.ravel(axes)

    for ax, name in zip(axes, FUNCTIONS_1D):
        info = get_function(name)
        a, b = info["bounds"]

        x = np.linspace(a, b, 800)
        y = info["func"](x)

        ax.plot(x, y)
        ax.set_title(f"{name}: {info['label']}")
        ax.grid(True, alpha=0.3)

    for ax in axes[len(FUNCTIONS_1D):]:
        ax.axis("off")

    _savefig(fig, "function_shapes")


def plot_convergence_1d(func_name):
    quad_path = os.path.join(TABLES_DIR, f"{func_name}_quadrature.csv")
    mc_path = os.path.join(TABLES_DIR, f"{func_name}_mc.csv")

    if not os.path.exists(quad_path) or not os.path.exists(mc_path):
        print(f"데이터 없음: {func_name}")
        return

    quad = pd.read_csv(quad_path)
    mc = pd.read_csv(mc_path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 결정론적 격자 기반 방법
    ax = axes[0]

    for method in QUADRATURE_METHODS:
        sub = quad[quad["method"] == method]
        if sub.empty:
            continue

        ax.loglog(
            sub["eval_count"],
            _positive_error(sub["abs_error"]),
            marker="o",
            label=method,
        )

    x = np.array([10, 1000], dtype=float)
    ax.loglog(x, 1e-1 * x ** -2, linestyle="--", label="O(n^-2)")
    ax.loglog(x, 1e-1 * x ** -4, linestyle=":", label="O(n^-4)")

    ax.set_title(f"{func_name} 격자 기반")
    ax.set_xlabel("함수 평가 횟수")
    ax.set_ylabel("절대오차")
    apply_plain_log_ticks(ax)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    # MC 계열 방법
    ax = axes[1]

    for method in MC_METHODS:
        sub = mc[mc["method"] == method]
        if sub.empty:
            continue

        y = _positive_error(sub["mean_abs_error"])

        ax.loglog(
            sub["eval_count"],
            y,
            marker="o",
            label=method,
        )

        if method in ("monte_carlo", "mcmc") and "std_estimate" in sub.columns:
            std = sub["std_estimate"].to_numpy(float)
            mask = np.isfinite(std)

            if mask.any():
                xx = sub["eval_count"].to_numpy(float)[mask]
                yy = y[mask]
                ss = std[mask]

                ax.fill_between(
                    xx,
                    np.maximum(yy - ss, LOG_ERROR_FLOOR),
                    yy + ss,
                    alpha=0.15,
                )

    x = np.array([10, 100000], dtype=float)
    ax.loglog(x, 5e-1 * x ** -0.5, linestyle="--", label="O(N^-1/2)")

    ax.set_title(f"{func_name} MC 계열")
    ax.set_xlabel("함수 평가 횟수")
    ax.set_ylabel("평균 절대오차")
    apply_plain_log_ticks(ax)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    _savefig(fig, f"convergence_1d_{func_name}")


def plot_mc_diagnostics(func_name):
    path = os.path.join(TABLES_DIR, f"{func_name}_mc.csv")

    if not os.path.exists(path):
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(8, 5))

    for method in ["monte_carlo", "mcmc"]:
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        ax.loglog(
            sub["eval_count"],
            _positive_error(sub["std_estimate"]),
            marker="o",
            label=f"{method} std",
        )

    ax.set_title(f"{func_name} 반복 추정값 표준편차")
    ax.set_xlabel("표본 수")
    ax.set_ylabel("표준편차")
    apply_plain_log_ticks(ax)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    _savefig(fig, f"mc_diagnostics_{func_name}")


def plot_step_sensitivity():
    path = os.path.join(TABLES_DIR, "step_sensitivity.csv")

    if not os.path.exists(path):
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(9, 5))

    for method in QUADRATURE_METHODS + MC_METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue

        ax.semilogy(
            sub["threshold"],
            _positive_error(sub["abs_error"]),
            marker="o",
            label=method,
        )

    ax.set_title("step 함수 불연속점 위치 민감도")
    ax.set_xlabel("불연속점 c")
    ax.set_ylabel("절대오차")
    apply_plain_log_ticks(ax)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    _savefig(fig, "step_sensitivity")


def plot_highdim_error(func_name):
    path = os.path.join(TABLES_DIR, f"highdim_{func_name}.csv")

    if not os.path.exists(path):
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(9, 5))

    for method in QUADRATURE_METHODS + MC_METHODS:
        sub = df[(df["method"] == method) & (df["status"] == "ok")]
        if sub.empty:
            continue

        ax.semilogy(
            sub["dim"],
            _positive_error(sub["abs_error"]),
            marker="o",
            label=method,
        )

    ax.set_title(f"고차원 오차: {func_name}")
    ax.set_xlabel("차원 d")
    ax.set_ylabel("절대오차")
    apply_plain_log_ticks(ax)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    _savefig(fig, f"highdim_error_{func_name}")


def plot_highdim_points():
    path = os.path.join(TABLES_DIR, "curse_of_dimensionality.csv")

    if not os.path.exists(path):
        return

    df = pd.read_csv(path)

    fig, ax = plt.subplots(figsize=(9, 5))

    for col in [c for c in df.columns if c.endswith("_points")]:
        ax.semilogy(
            df["dim"],
            df[col],
            marker="o",
            label=col.replace("_points", ""),
        )

    ax.set_title("차원 증가에 따른 함수 평가 횟수")
    ax.set_xlabel("차원 d")
    ax.set_ylabel("평가점 수")
    apply_plain_log_ticks(ax)

    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)

    _savefig(fig, "highdim_function_evaluations")


def plot_all(verbose=True):
    if verbose:
        print("\n그래프 생성")

    plot_function_shapes()

    for name in FUNCTIONS_1D:
        plot_convergence_1d(name)
        plot_mc_diagnostics(name)

    plot_step_sensitivity()
    plot_highdim_points()

    for name in FUNCTIONS_HD:
        plot_highdim_error(name)


if __name__ == "__main__":
    plot_all(verbose=True)
"""고차원 실험."""

import os
import sys
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TABLES_DIR,
    FUNCTIONS_HD,
    QUADRATURE_METHODS,
    MC_METHODS,
    HIGH_DIM_LIST,
    HIGH_DIM_QUAD_N,
    HIGH_DIM_MC_N,
    MAX_TENSOR_POINTS,
)
from functions import get_function
from quadrature import integrate_nd, method_eval_count_nd
from monte_carlo import integrate_mc_nd


def _rel_error(err, exact):
    return err / abs(exact) if exact != 0 else np.nan


def curse_of_dimensionality_table():
    rows = []
    for dim in HIGH_DIM_LIST:
        n = HIGH_DIM_QUAD_N[dim]
        row = {"dim": dim, "n_per_axis": n}
        for method in QUADRATURE_METHODS:
            row[f"{method}_points"] = method_eval_count_nd(method, dim, n)
        row["mc_points"] = HIGH_DIM_MC_N
        rows.append(row)
    return pd.DataFrame(rows)


def run_highdim_experiment(func_name, seed=42, verbose=True):
    info = get_function(func_name)
    func_nd = info["func_nd"]
    exact_fn = info["exact_nd"]
    a, b = info["bounds"]
    rows = []
    rng = np.random.default_rng(seed)

    for method in QUADRATURE_METHODS:
        for dim in HIGH_DIM_LIST:
            n = HIGH_DIM_QUAD_N[dim]
            points = method_eval_count_nd(method, dim, n)
            exact = exact_fn(a, b, dim)
            if points > MAX_TENSOR_POINTS:
                if verbose:
                    print(f" [{method:12s}] d={dim:2d}, points={points:.2e}: 건너뜀")
                rows.append({
                    "function": func_name,
                    "method": method,
                    "dim": dim,
                    "n_or_N": n,
                    "num_points": points,
                    "estimate": np.nan,
                    "exact": exact,
                    "abs_error": np.nan,
                    "rel_error": np.nan,
                    "time_sec": np.nan,
                    "status": "skipped_too_many_tensor_points",
                })
                continue
            if verbose:
                print(f" [{method:12s}] d={dim:2d}, n={n:3d}, points={points:.2e} ... ", end="", flush=True)
            try:
                t0 = time.perf_counter()
                est = integrate_nd(method, func_nd, a, b, dim, n)
                t1 = time.perf_counter()
                err = abs(est - exact)
                if verbose:
                    print(f"err={err:.3e}, time={t1-t0:.3f}s")
                rows.append({
                    "function": func_name,
                    "method": method,
                    "dim": dim,
                    "n_or_N": n,
                    "num_points": points,
                    "estimate": est,
                    "exact": exact,
                    "abs_error": err,
                    "rel_error": _rel_error(err, exact),
                    "time_sec": t1 - t0,
                    "status": "ok",
                })
            except Exception as e:
                if verbose:
                    print(f"실패: {e}")
                rows.append({
                    "function": func_name,
                    "method": method,
                    "dim": dim,
                    "n_or_N": n,
                    "num_points": points,
                    "estimate": np.nan,
                    "exact": exact,
                    "abs_error": np.nan,
                    "rel_error": np.nan,
                    "time_sec": np.nan,
                    "status": f"failed: {e}",
                })

    for method in MC_METHODS:
        for dim in HIGH_DIM_LIST:
            n = HIGH_DIM_MC_N
            exact = exact_fn(a, b, dim)
            if verbose:
                print(f" [{method:12s}] d={dim:2d}, N={n} ... ", end="", flush=True)
            try:
                t0 = time.perf_counter()
                est = integrate_mc_nd(method, func_nd, a, b, dim, n, seed=int(rng.integers(0, 2**31 - 1)))
                t1 = time.perf_counter()
                err = abs(est - exact)
                if verbose:
                    print(f"err={err:.3e}, time={t1-t0:.3f}s")
                rows.append({
                    "function": func_name,
                    "method": method,
                    "dim": dim,
                    "n_or_N": n,
                    "num_points": n,
                    "estimate": est,
                    "exact": exact,
                    "abs_error": err,
                    "rel_error": _rel_error(err, exact),
                    "time_sec": t1 - t0,
                    "status": "ok",
                })
            except Exception as e:
                if verbose:
                    print(f"실패: {e}")
                rows.append({
                    "function": func_name,
                    "method": method,
                    "dim": dim,
                    "n_or_N": n,
                    "num_points": n,
                    "estimate": np.nan,
                    "exact": exact,
                    "abs_error": np.nan,
                    "rel_error": np.nan,
                    "time_sec": np.nan,
                    "status": f"failed: {e}",
                })
    return pd.DataFrame(rows)


def run_all_highdim(verbose=True):
    os.makedirs(TABLES_DIR, exist_ok=True)
    results = {}
    cod = curse_of_dimensionality_table()
    cod.to_csv(os.path.join(TABLES_DIR, "curse_of_dimensionality.csv"), index=False)
    results["curse_of_dimensionality"] = cod
    if verbose:
        print("\n[고차원] 차원의 저주 평가점 수 표")
        print(cod.to_string(index=False))

    for func_name in FUNCTIONS_HD:
        if verbose:
            print(f"\n[고차원] {func_name}")
        df = run_highdim_experiment(func_name, verbose=verbose)
        df.to_csv(os.path.join(TABLES_DIR, f"highdim_{func_name}.csv"), index=False)
        results[func_name] = df
    return results


if __name__ == "__main__":
    run_all_highdim(verbose=True)

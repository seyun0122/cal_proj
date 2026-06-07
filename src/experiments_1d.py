"""1차원 실험과 보고서용 보조 표 생성."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    TABLES_DIR,
    FUNCTIONS_1D,
    QUADRATURE_METHODS,
    MC_METHODS,
    QUADRATURE_N_LIST,
    MC_N_LIST,
    MC_REPEAT,
    REPRESENTATIVE_QUAD_N,
    REPRESENTATIVE_MC_N,
    EQUAL_BUDGET_LIST,
    STEP_THRESHOLDS,
    MCMC_BURN_IN,
    MCMC_PROPOSAL_SCALE,
    CONVERGENCE_TAIL_POINTS,
)
from functions import get_function, make_step
from quadrature import integrate_1d, method_eval_count_1d, n_for_eval_budget
from monte_carlo import monte_carlo_stats, quasi_mc, mcmc_stats
from smoothness import analyze as smoothness_analyze


def _rel_error(err, exact):
    return err / abs(exact) if exact != 0 else np.nan


def run_quadrature_1d(func_name):
    info = get_function(func_name)
    func = info["func"]
    a, b = info["bounds"]
    exact = info["exact"](a, b)
    rows = []
    for method in QUADRATURE_METHODS:
        for n in QUADRATURE_N_LIST:
            est = integrate_1d(method, func, a, b, n)
            err = abs(est - exact)
            rows.append({
                "function": func_name,
                "method": method,
                "n": n,
                "eval_count": method_eval_count_1d(method, n),
                "estimate": est,
                "exact": exact,
                "abs_error": err,
                "rel_error": _rel_error(err, exact),
            })
    return pd.DataFrame(rows)


def run_mc_1d(func_name, seed=42):
    info = get_function(func_name)
    func = info["func"]
    a, b = info["bounds"]
    exact = info["exact"](a, b)
    rows = []
    for method in MC_METHODS:
        for n in MC_N_LIST:
            if method == "monte_carlo":
                r = monte_carlo_stats(func, a, b, n, repeat=MC_REPEAT, seed=seed)
            elif method == "quasi_mc":
                est = quasi_mc(func, a, b, n)
                r = {
                    "mean_estimate": est,
                    "std_estimate": np.nan,
                    "accept_rate": np.nan,
                    "lag1_autocorr": np.nan,
                    "ess_lag1": np.nan,
                    "repeat": np.nan,
                }
            elif method == "mcmc":
                proposal_std = (b - a) * MCMC_PROPOSAL_SCALE
                r = mcmc_stats(
                    func,
                    a,
                    b,
                    n,
                    repeat=MC_REPEAT,
                    burn_in=MCMC_BURN_IN,
                    proposal_std=proposal_std,
                    seed=seed,
                )
            else:
                continue
            err = abs(r["mean_estimate"] - exact)
            rows.append({
                "function": func_name,
                "method": method,
                "n": n,
                "eval_count": n,
                "mean_estimate": r["mean_estimate"],
                "std_estimate": r.get("std_estimate", np.nan),
                "exact": exact,
                "mean_abs_error": err,
                "rel_error": _rel_error(err, exact),
                "accept_rate": r.get("accept_rate", np.nan),
                "lag1_autocorr": r.get("lag1_autocorr", np.nan),
                "ess_lag1": r.get("ess_lag1", np.nan),
                "repeat": r.get("repeat", np.nan),
            })
    return pd.DataFrame(rows)


def run_smoothness_report(func_names=None):
    if func_names is None:
        func_names = FUNCTIONS_1D
    rows = []
    for name in func_names:
        info = get_function(name)
        a, b = info["bounds"]
        rows.append(smoothness_analyze(info["func"], a, b, name=name))
    return pd.DataFrame(rows)


def make_representative_table(results):
    rows = []
    for func_name in FUNCTIONS_1D:
        info = get_function(func_name)
        a, b = info["bounds"]
        exact = info["exact"](a, b)
        row = {"function": func_name, "exact": exact}

        quad_df = results[f"{func_name}_quadrature"]
        for method in QUADRATURE_METHODS:
            sub = quad_df[(quad_df["method"] == method) & (quad_df["n"] == REPRESENTATIVE_QUAD_N)]
            if sub.empty:
                row[f"{method}_abs_error"] = np.nan
                row[f"{method}_estimate"] = np.nan
            else:
                row[f"{method}_abs_error"] = float(sub.iloc[0]["abs_error"])
                row[f"{method}_estimate"] = float(sub.iloc[0]["estimate"])

        mc_df = results[f"{func_name}_mc"]
        for method in MC_METHODS:
            sub = mc_df[(mc_df["method"] == method) & (mc_df["n"] == REPRESENTATIVE_MC_N)]
            if sub.empty:
                row[f"{method}_abs_error"] = np.nan
                row[f"{method}_estimate"] = np.nan
                row[f"{method}_std"] = np.nan
            else:
                s = sub.iloc[0]
                row[f"{method}_abs_error"] = float(s["mean_abs_error"])
                row[f"{method}_estimate"] = float(s["mean_estimate"])
                row[f"{method}_std"] = float(s["std_estimate"]) if pd.notna(s["std_estimate"]) else np.nan
                if method == "mcmc":
                    row["mcmc_accept_rate"] = float(s["accept_rate"])
                    row["mcmc_lag1_autocorr"] = float(s["lag1_autocorr"])
                    row["mcmc_ess_lag1"] = float(s["ess_lag1"])
        rows.append(row)
    return pd.DataFrame(rows)


def make_equal_budget_table(budget, seed=42):
    rows = []
    for func_name in FUNCTIONS_1D:
        info = get_function(func_name)
        func = info["func"]
        a, b = info["bounds"]
        exact = info["exact"](a, b)

        for method in QUADRATURE_METHODS:
            n = n_for_eval_budget(method, budget)
            est = integrate_1d(method, func, a, b, n)
            err = abs(est - exact)
            rows.append({
                "function": func_name,
                "method": method,
                "budget": budget,
                "n_or_N": n,
                "eval_count": method_eval_count_1d(method, n),
                "estimate": est,
                "exact": exact,
                "abs_error": err,
                "rel_error": _rel_error(err, exact),
                "std_estimate": np.nan,
                "accept_rate": np.nan,
                "lag1_autocorr": np.nan,
                "ess_lag1": np.nan,
            })

        for method in MC_METHODS:
            if method == "monte_carlo":
                r = monte_carlo_stats(func, a, b, budget, repeat=MC_REPEAT, seed=seed)
            elif method == "quasi_mc":
                r = {"mean_estimate": quasi_mc(func, a, b, budget), "std_estimate": np.nan}
            else:
                proposal_std = (b - a) * MCMC_PROPOSAL_SCALE
                r = mcmc_stats(func, a, b, budget, repeat=MC_REPEAT, burn_in=MCMC_BURN_IN, proposal_std=proposal_std, seed=seed)
            err = abs(r["mean_estimate"] - exact)
            rows.append({
                "function": func_name,
                "method": method,
                "budget": budget,
                "n_or_N": budget,
                "eval_count": budget,
                "estimate": r["mean_estimate"],
                "exact": exact,
                "abs_error": err,
                "rel_error": _rel_error(err, exact),
                "std_estimate": r.get("std_estimate", np.nan),
                "accept_rate": r.get("accept_rate", np.nan),
                "lag1_autocorr": r.get("lag1_autocorr", np.nan),
                "ess_lag1": r.get("ess_lag1", np.nan),
            })
    return pd.DataFrame(rows)


def run_step_sensitivity(seed=42):
    a, b = 0.0, 1.0
    rows = []
    for threshold in STEP_THRESHOLDS:
        func, exact_fn = make_step(threshold)
        exact = exact_fn(a, b)
        for method in QUADRATURE_METHODS:
            est = integrate_1d(method, func, a, b, REPRESENTATIVE_QUAD_N)
            err = abs(est - exact)
            rows.append({
                "threshold": threshold,
                "method": method,
                "n_or_N": REPRESENTATIVE_QUAD_N,
                "eval_count": method_eval_count_1d(method, REPRESENTATIVE_QUAD_N),
                "estimate": est,
                "exact": exact,
                "abs_error": err,
                "std_estimate": np.nan,
                "accept_rate": np.nan,
                "lag1_autocorr": np.nan,
                "ess_lag1": np.nan,
            })
        for method in MC_METHODS:
            if method == "monte_carlo":
                r = monte_carlo_stats(func, a, b, REPRESENTATIVE_MC_N, repeat=MC_REPEAT, seed=seed)
            elif method == "quasi_mc":
                r = {"mean_estimate": quasi_mc(func, a, b, REPRESENTATIVE_MC_N), "std_estimate": np.nan}
            else:
                r = mcmc_stats(func, a, b, REPRESENTATIVE_MC_N, repeat=MC_REPEAT, burn_in=MCMC_BURN_IN, proposal_std=MCMC_PROPOSAL_SCALE, seed=seed)
            err = abs(r["mean_estimate"] - exact)
            rows.append({
                "threshold": threshold,
                "method": method,
                "n_or_N": REPRESENTATIVE_MC_N,
                "eval_count": REPRESENTATIVE_MC_N,
                "estimate": r["mean_estimate"],
                "exact": exact,
                "abs_error": err,
                "std_estimate": r.get("std_estimate", np.nan),
                "accept_rate": r.get("accept_rate", np.nan),
                "lag1_autocorr": r.get("lag1_autocorr", np.nan),
                "ess_lag1": r.get("ess_lag1", np.nan),
            })
    return pd.DataFrame(rows)


def _empirical_rate_from_df(df, error_col, x_col):
    sub = df[[x_col, error_col]].replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub[(sub[error_col] > 1e-15) & (sub[x_col] > 0)].sort_values(x_col)
    if len(sub) < 2:
        return np.nan
    sub = sub.tail(CONVERGENCE_TAIL_POINTS)
    slope, _ = np.polyfit(np.log(sub[x_col].to_numpy(float)), np.log(sub[error_col].to_numpy(float)), 1)
    return float(slope)


def make_empirical_convergence_rates(results):
    rows = []
    for func_name in FUNCTIONS_1D:
        quad_df = results[f"{func_name}_quadrature"]
        for method in QUADRATURE_METHODS:
            sub = quad_df[quad_df["method"] == method]
            rows.append({
                "function": func_name,
                "method": method,
                "x_axis": "eval_count",
                "empirical_rate": _empirical_rate_from_df(sub, "abs_error", "eval_count"),
            })
        mc_df = results[f"{func_name}_mc"]
        for method in MC_METHODS:
            sub = mc_df[mc_df["method"] == method]
            rows.append({
                "function": func_name,
                "method": method,
                "x_axis": "eval_count",
                "empirical_rate": _empirical_rate_from_df(sub, "mean_abs_error", "eval_count"),
            })
    return pd.DataFrame(rows)


def run_all_1d(verbose=True):
    os.makedirs(TABLES_DIR, exist_ok=True)
    results = {}

    if verbose:
        print("\n[1차원] 수치적 거칠기 지표 계산")
    smooth_df = run_smoothness_report()
    smooth_df.to_csv(os.path.join(TABLES_DIR, "smoothness.csv"), index=False)
    results["smoothness"] = smooth_df

    for func_name in FUNCTIONS_1D:
        if verbose:
            print(f"\n[1차원] {func_name}")
        quad_df = run_quadrature_1d(func_name)
        mc_df = run_mc_1d(func_name)
        results[f"{func_name}_quadrature"] = quad_df
        results[f"{func_name}_mc"] = mc_df
        quad_df.to_csv(os.path.join(TABLES_DIR, f"{func_name}_quadrature.csv"), index=False)
        mc_df.to_csv(os.path.join(TABLES_DIR, f"{func_name}_mc.csv"), index=False)

    rep = make_representative_table(results)
    rep.to_csv(os.path.join(TABLES_DIR, "representative_1d.csv"), index=False)
    results["representative_1d"] = rep

    for budget in EQUAL_BUDGET_LIST:
        eq = make_equal_budget_table(budget)
        eq.to_csv(os.path.join(TABLES_DIR, f"equal_budget_{budget}.csv"), index=False)
        results[f"equal_budget_{budget}"] = eq

    step_df = run_step_sensitivity()
    step_df.to_csv(os.path.join(TABLES_DIR, "step_sensitivity.csv"), index=False)
    results["step_sensitivity"] = step_df

    rates = make_empirical_convergence_rates(results)
    rates.to_csv(os.path.join(TABLES_DIR, "empirical_convergence_rates.csv"), index=False)
    results["empirical_convergence_rates"] = rates

    if verbose:
        print(f"\n완료: {TABLES_DIR}")
    return results


if __name__ == "__main__":
    run_all_1d(verbose=True)

"""이미 저장된 CSV에서 보고서용 요약을 다시 만드는 유틸리티."""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import TABLES_DIR, FUNCTIONS_1D, QUADRATURE_METHODS, MC_METHODS, REPRESENTATIVE_QUAD_N, REPRESENTATIVE_MC_N, EQUAL_BUDGET_LIST
from experiments_1d import make_equal_budget_table


def _load(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"필요한 CSV가 없다: {path}")
    return pd.read_csv(path)


def rebuild_representative_1d():
    rows = []
    for func_name in FUNCTIONS_1D:
        quad = _load(os.path.join(TABLES_DIR, f"{func_name}_quadrature.csv"))
        mc = _load(os.path.join(TABLES_DIR, f"{func_name}_mc.csv"))
        row = {"function": func_name}
        exact_vals = []
        if "exact" in quad:
            exact_vals.append(float(quad["exact"].iloc[0]))
        if "exact" in mc:
            exact_vals.append(float(mc["exact"].iloc[0]))
        row["exact"] = exact_vals[0] if exact_vals else np.nan
        for method in QUADRATURE_METHODS:
            sub = quad[(quad["method"] == method) & (quad["n"] == REPRESENTATIVE_QUAD_N)]
            row[f"{method}_abs_error"] = float(sub.iloc[0]["abs_error"]) if not sub.empty else np.nan
        for method in MC_METHODS:
            sub = mc[(mc["method"] == method) & (mc["n"] == REPRESENTATIVE_MC_N)]
            row[f"{method}_abs_error"] = float(sub.iloc[0]["mean_abs_error"]) if not sub.empty else np.nan
            row[f"{method}_std"] = float(sub.iloc[0]["std_estimate"]) if (not sub.empty and pd.notna(sub.iloc[0]["std_estimate"])) else np.nan
            if method == "mcmc" and not sub.empty:
                row["mcmc_accept_rate"] = float(sub.iloc[0]["accept_rate"])
                row["mcmc_lag1_autocorr"] = float(sub.iloc[0]["lag1_autocorr"])
                row["mcmc_ess_lag1"] = float(sub.iloc[0]["ess_lag1"])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "representative_1d.csv"), index=False)
    return df


def rebuild_equal_budget_tables():
    out = {}
    for budget in EQUAL_BUDGET_LIST:
        df = make_equal_budget_table(budget)
        df.to_csv(os.path.join(TABLES_DIR, f"equal_budget_{budget}.csv"), index=False)
        out[budget] = df
    return out


def rebuild_all_report_tables():
    os.makedirs(TABLES_DIR, exist_ok=True)
    rep = rebuild_representative_1d()
    eq = rebuild_equal_budget_tables()
    return {"representative_1d": rep, "equal_budget": eq}


if __name__ == "__main__":
    rebuild_all_report_tables()
    print(f"보고서용 표 재생성 완료: {TABLES_DIR}")

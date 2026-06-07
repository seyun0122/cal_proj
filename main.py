"""실험 실행 진입점."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def mode_check():
    from functions import get_function
    from quadrature import integrate_1d
    from monte_carlo import integrate_mc_1d, mcmc_stats

    info = get_function("sin")
    func = info["func"]
    a, b = info["bounds"]
    exact = info["exact"](a, b)
    print("빠른 검증: sin, 참값", exact)
    for method in ["midpoint", "trapezoidal", "simpson", "gauss"]:
        est = integrate_1d(method, func, a, b, 16)
        print(f"{method:12s} {est:.8f} error={abs(est-exact):.3e}")
    for method in ["monte_carlo", "quasi_mc", "mcmc"]:
        est = integrate_mc_1d(method, func, a, b, 5000, seed=42)
        print(f"{method:12s} {est:.8f} error={abs(est-exact):.3e}")
    diag = mcmc_stats(func, a, b, 5000, repeat=5, seed=42)
    print("MCMC diagnostics:", diag)


def mode_1d():
    from experiments_1d import run_all_1d
    run_all_1d(verbose=True)


def mode_highdim():
    from experiments_high_dim import run_all_highdim
    run_all_highdim(verbose=True)


def mode_plot():
    from plot_results import plot_all
    plot_all(verbose=True)


def mode_report():
    from report_tables import rebuild_all_report_tables
    rebuild_all_report_tables()


def main():
    parser = argparse.ArgumentParser(description="수치적분 비교 실험 확장 버전")
    parser.add_argument("--mode", choices=["all", "1d", "highdim", "plot", "report", "check"], default="all")
    args = parser.parse_args()

    if args.mode == "check":
        mode_check()
    elif args.mode == "1d":
        mode_1d()
    elif args.mode == "highdim":
        mode_highdim()
    elif args.mode == "plot":
        mode_plot()
    elif args.mode == "report":
        mode_report()
    elif args.mode == "all":
        mode_1d()
        mode_highdim()
        mode_plot()


if __name__ == "__main__":
    main()

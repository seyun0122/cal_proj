"""전역 설정 파일."""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

QUADRATURE_N_LIST = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
MC_N_LIST = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
MC_REPEAT = 30

REPRESENTATIVE_QUAD_N = 256
REPRESENTATIVE_MC_N = 50000
EQUAL_BUDGET_LIST = [256, 1024]

MCMC_BURN_IN = 1000
MCMC_PROPOSAL_SCALE = 0.2

STEP_THRESHOLDS = [0.31, 0.37, 0.50, 0.61, 0.73]

HIGH_DIM_LIST = [1, 2, 3, 5, 8, 10]
HIGH_DIM_QUAD_N = {1: 64, 2: 32, 3: 16, 5: 8, 8: 4, 10: 4}
HIGH_DIM_MC_N = 50000
MAX_TENSOR_POINTS = 2_000_000

FUNCTIONS_1D = ["sin", "sin10x", "gaussian_peak", "runge", "step", "steep_sigmoid"]
FUNCTIONS_HD = ["exp_product", "sum_of_squares"]

QUADRATURE_METHODS = ["midpoint", "trapezoidal", "simpson", "gauss"]
MC_METHODS = ["monte_carlo", "quasi_mc", "mcmc"]

FIGURE_DPI = 150
FIGURE_FORMAT = "png"
LOG_ERROR_FLOOR = 1e-16
CONVERGENCE_TAIL_POINTS = 5

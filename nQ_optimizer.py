"""nQ_optimizer.py

Tools for choosing optimal laser pulse intensities for extracting nth-quantum-
coherence (nQ) signals in nonlinear optical spectroscopy.

The core computation is handled by OptimizeIntensities from order_separation.py,
which must be importable from the same directory or from sys.path.

Created by Claude (Anthropic), 2026-03-03.

Usage example
-------------
from nQ_optimizer import nQ_exponential_saturation_model, optimize_nQ_intensities

total_err, optimal_I, err_by_order = optimize_nQ_intensities(
    nQ_exponential_saturation_model,
    noise_over_S_max=1.6e-6 / 0.05,
    ns=[1],
    Ns=[3, 4, 5],
    M=4,
    I_sat=14.0,   # nJ
    I_max=60.0,   # nJ
)
"""

import numpy as np
from scipy.special import factorial, ive, binom, hyp2f1
from tabulate import tabulate
from itertools import product

from order_separation import OptimizeIntensities


# ---------------------------------------------------------------------------
# Built-in model factories
# ---------------------------------------------------------------------------
# Each factory takes the quantum order n and returns a pair (f, f_series):
#   f(x)         : total nQ signal at dimensionless intensity x = I / I_sat
#   f_series(m, x): mth-order term in the power-series expansion of f(x)
# ---------------------------------------------------------------------------

def nQ_exponential_saturation_model(n):
    """Exponential-saturation model for the nQ signal.

    Based on the TA saturation form S(x) = 1 − exp(−x), which gives the nQ
    signal component f(x) = (−1)^n · ive(n, 2x), where x = I / I_sat.

    Parameters
    ----------
    n : int
        nQ order (1 for 1Q, 2 for 2Q, …).

    Returns
    -------
    f : callable  f(x) → float
        nQ signal at dimensionless intensity x = I / I_sat.
    f_series : callable  f_series(m, x) → float
        mth-order term in the power-series expansion of f(x).
    """
    def f(x):
        if n == 0:
            return ive(0, 2.0 * x) - 1.0
        return (-1)**n * ive(n, 2.0 * x)

    def f_series(m, x):
        return (-1)**m * x**m / factorial(m) * binom(2 * m, m - n)

    return f, f_series


def nQ_saturable_absorption_model(n):
    """Saturable-absorption model for the nQ signal.

    Based on the TA saturation form S(x) = x / (x + 1), which gives the nQ
    signal via a hypergeometric function, where x = I / I_sat.

    Parameters
    ----------
    n : int
        nQ order (1 for 1Q, 2 for 2Q, …).

    Returns
    -------
    f, f_series : callables — see nQ_exponential_saturation_model for signatures.
    """
    def f(x):
        return (-1) * (-x)**n * hyp2f1(n + 0.5, n + 1, 2 * n + 1, -4.0 * x)

    def f_series(m, x):
        return -(-x)**m * binom(2 * m, m - n)

    return f, f_series


def ta_exponential_model(n):
    """Direct exponential model of the TA signal (n-independent).

    f(x) = −(1 − exp(−x)), where x = I / I_sat.
    The argument n is accepted for interface compatibility but is ignored.

    Returns
    -------
    f, f_series : callables — see nQ_exponential_saturation_model for signatures.
    """
    def f(x):
        return -(1.0 - np.exp(-x))

    def f_series(m, x):
        if m == 0:
            return np.zeros_like(np.asarray(x, dtype=float))
        return (-x)**m / factorial(m)

    return f, f_series


def ta_saturable_absorption_model(n):
    """Direct saturable-absorption model of the TA signal (n-independent).

    f(x) = −x / (x + 1), where x = I / I_sat.
    The argument n is accepted for interface compatibility but is ignored.

    Returns
    -------
    f, f_series : callables — see nQ_exponential_saturation_model for signatures.
    """
    def f(x):
        return -x / (x + 1.0)

    def f_series(m, x):
        if m == 0:
            return np.zeros_like(np.asarray(x, dtype=float))
        return (-x)**m

    return f, f_series


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def optimize_nQ_intensities(
    get_model,
    noise_over_S_max,
    *,
    ns=(1,),
    Ns=(3,),
    M=3,
    I_sat=1.0,
    I_max=None,
    I0=None,
    fixed_intensities=None,
    print_tables=True,
):
    """Compute optimal laser pulse intensities for nQ spectroscopy.

    Parameters
    ----------
    get_model : callable
        Model factory.  ``get_model(n)`` must return ``(f, f_series)`` for
        quantum order n, where:

            ``f(x)``          signal at dimensionless intensity x = I / I_sat.
            ``f_series(m, x)`` mth-order term in the series expansion of f(x).

        Built-in options: ``nQ_exponential_saturation_model``, ``nQ_saturable_absorption_model``,
        ``ta_exponential_model``, ``ta_saturable_absorption_model``.
        Users may supply their own factory.

    noise_over_S_max : float
        Dimensionless noise level  σ / S_max.

    ns : sequence of int
        Quantum coherence orders to evaluate
        (e.g., ``[1]`` for 1Q only, ``[1, 2]`` for 1Q and 2Q).

    Ns : sequence of int
        Candidate numbers of pulse intensities to compare
        (e.g., ``[3, 4, 5]``).  Ignored when ``fixed_intensities`` is given.

    M : int
        Number of perturbation orders to include in the error calculation.

    I_sat : float
        Saturation intensity in any unit (e.g. nJ or normalized).
        All output intensities are reported in the same unit.

    I_max : float or None
        Maximum achievable intensity (same units as I_sat).
        When provided the optimisation is bounded to [0, I_max].
        When None the optimisation is unconstrained.

    I0 : float or None
        Reference intensity for the Vandermonde matrix (same units as I_sat).
        For signals whose leading-order series term is zero (e.g. 2Q)
        this sets the scale for absolute rather than relative
        errors.  Defaults to I_sat.

    fixed_intensities : array-like or None
        If given (in same units as I_sat), errors are evaluated at these
        exact intensities without any optimisation.  Overrides Ns; N is
        inferred as ``len(fixed_intensities)``.

    print_tables : bool
        If True, print formatted result tables to stdout.

    Returns
    -------
    total_err : ndarray, shape (len(Ns), 1 + 3·len(ns))
        Columns: N, then total / random / systematic error for each n in ns.
        Each error is divided by sqrt(min(N, M)), as given by Eq. 15 in
        Krich et al JPCL 2025.

    optimal_I : ndarray, shape (len(Ns), len(ns), max(Ns))
        Optimal intensities in physical units (same as I_sat), zero-padded.
        Meaningful only when fixed_intensities is None.

    err_by_order : ndarray or None
        Shape (len(ns), 2·M+1): per-order random and systematic errors
        (not divided by sqrt(M)).  Computed only in the single-N case
        (len(Ns) == 1); otherwise None.
    """
    ns = list(ns)
    Ns = list(Ns)

    if I0 is None:
        I0 = I_sat
    I0_dl = I0 / I_sat          # dimensionless, passed to OptimizeIntensities

    if fixed_intensities is not None:
        fixed_dl = np.asarray(fixed_intensities, dtype=float) / I_sat
        Ns = [len(fixed_dl)]

    bounds = [1e-15, I_max / I_sat] if I_max is not None else [1e-15, 1e4]

    single_N = (len(Ns) == 1)
    num_cols = 1 + 3 * len(ns)
    total_err = np.zeros((len(Ns), num_cols))
    total_err[:, 0] = Ns

    optimal_I = np.zeros((len(Ns), len(ns), max(Ns)))

    err_by_order = None
    if single_N:
        err_by_order = np.zeros((len(ns), 2 * M + 1))
        err_by_order[:, 0] = ns

    for (i_N, N), (i_n, n) in product(enumerate(Ns), enumerate(ns)):
        curM = min(N, M)
        f, f_series = get_model(n)
        opt = OptimizeIntensities(f, f_series, noise_over_S_max, I0=I0_dl)
        opt.num = curM

        if fixed_intensities is not None:
            # Evaluate errors at the fixed intensities (no optimisation).
            # Call the array version once; derive scalars from it.
            r_arr, s_arr = opt.get_random_and_systematic_error_arrays(fixed_dl)
            r = np.sqrt(np.sum(r_arr**2))
            s = np.sqrt(np.sum(s_arr**2))
            intensities = fixed_dl
        else:
            intensities, r, s, _ = opt.minimize_errors_single_N(
                curM, N,
                intensity_selection='bounded',
                bounds=bounds,
                optimization_cycles=3,
            )
            optimal_I[i_N, i_n, :N] = intensities * I_sat
            if single_N:
                r_arr, s_arr = opt.get_random_and_systematic_error_arrays(
                    intensities
                )

        k = np.sqrt(curM)
        total_err[i_N, i_n + 1]              = np.sqrt(r**2 + s**2) / k
        total_err[i_N, i_n + 1 + len(ns)]   = r / k
        total_err[i_N, i_n + 1 + 2*len(ns)] = s / k

        if single_N:
            nr = len(r_arr)
            err_by_order[i_n, 1:nr + 1]          = r_arr
            err_by_order[i_n, M + 1:M + 1 + nr]  = s_arr

    if print_tables:
        _print_tables(
            total_err, optimal_I, err_by_order,
            ns, Ns, M, I_max, fixed_intensities, single_N,
        )

    return total_err, optimal_I, err_by_order


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _print_tables(total_err, optimal_I, err_by_order,
                  ns, Ns, M, I_max, fixed_intensities, single_N):
    """Print formatted result tables."""

    # --- Header line: fixed vs optimised, bounded vs unbounded ---
    if fixed_intensities is not None:
        print('Fixed intensities (no optimisation)')
    elif I_max is not None:
        print(f'Optimised intensities, I_max = {I_max:0.5g}')
    else:
        print('Optimised intensities, unbounded')

    # --- Main error table ---
    heads = (
        ['N']
        + [f'{n}Q tot'  for n in ns]
        + [f'{n}Q rand' for n in ns]
        + [f'{n}Q sys'  for n in ns]
    )
    fmts = ['1.0f'] + ['0.5f'] * (3 * len(ns))
    print('\nTotal errors:')
    print(tabulate(total_err, headers=heads, floatfmt=fmts))

    # --- Intensity table ---
    if fixed_intensities is not None:
        print(f'\nFixed intensities: {np.asarray(fixed_intensities)}')
    else:
        for i_n, n in enumerate(ns):
            print(f'\nOptimal intensities for {n}Q (same units as I_sat):')
            rows = []
            for i_N, N in enumerate(Ns):
                vals = optimal_I[i_N, i_n, :N]
                rows.append([N] + [f'{v:.4g}' for v in vals])
            col_heads = ['N'] + [f'I_{i+1}' for i in range(max(Ns))]
            print(tabulate(rows, headers=col_heads, tablefmt='simple'))

    # --- Per-order error table (single-N case only) ---
    if single_N and err_by_order is not None:
        head_err = (
            ['nQ']
            + [f'r({2*i+1})' for i in range(1, M + 1)]
            + [f's({2*i+1})' for i in range(1, M + 1)]
        )
        fmts_err = ['1.0f'] + ['0.4f'] * (2 * M)
        print('\nErrors by perturbation order:')
        print(tabulate(err_by_order, headers=head_err, floatfmt=fmts_err))

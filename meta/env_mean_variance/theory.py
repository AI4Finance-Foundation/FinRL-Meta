"""Closed-form solution of the mean-variance problem in ``MeanVarianceEnv``.

From H. Wang and X. Y. Zhou (2020), Section 3 and Theorems 1-2. These use the true
market parameters, so they are benchmarks for RL agents, not something an agent can
use while learning.
"""

from __future__ import annotations

import numpy as np


def optimal_lagrange_multiplier(z: float, x0: float, rho: float, T: float) -> float:
    """w* = (z e^{rho^2 T} - x0) / (e^{rho^2 T} - 1), which gives E[x_T] = z (Theorem 1)."""
    growth = np.exp(rho**2 * T)
    return (z * growth - x0) / (growth - 1.0)


def classical_policy(x, w: float, rho: float, sigma: float):
    """Optimal allocation of the classical mean-variance problem, u = -rho / sigma (x - w)."""
    return -rho / sigma * (x - w)


def exploratory_policy(t, x, w: float, rho: float, sigma: float, lam: float, T: float):
    """Mean and variance of the optimal Gaussian exploratory policy (eq. 24).

    ``lam`` is the weight of the entropy regularization (the exploration temperature).
    """
    mean = -rho / sigma * (x - w)
    var = lam / (2.0 * sigma**2) * np.exp(rho**2 * (T - t))
    return mean, var


def optimal_terminal_moments(z: float, x0: float, rho: float, T: float):
    """Mean and variance of x_T under the classical optimal policy with w = w*.

    Var[x_T] = (z - x0)^2 / (e^{rho^2 T} - 1) is the efficient frontier of the
    continuous-time mean-variance problem.
    """
    return z, (z - x0) ** 2 / (np.exp(rho**2 * T) - 1.0)

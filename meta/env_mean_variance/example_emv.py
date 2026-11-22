"""Train the EMV agent and compare it with the paper and the efficient frontier.

    python -m meta.env_mean_variance.example_emv [n_episodes]

For each market, trains EMV with the settings of Section 5.1 of Wang and Zhou (2020)
and reports the mean M, variance V and Sharpe ratio (M - 1) / sqrt(V) of the last 2000
terminal wealths, next to the EMV results in Table 1 of the paper and the efficient
frontier of the classical problem (the best possible variance for E[x_T] = z).
"""

from __future__ import annotations

import sys

import numpy as np

from meta.env_mean_variance import theory
from meta.env_mean_variance.emv_agent import EMVAgent
from meta.env_mean_variance.emv_agent import EMVConfig
from meta.env_mean_variance.env_mean_variance import MeanVarianceEnv

# (mu, sigma): (M, V, SR) of EMV in Table 1 of the paper.
PAPER = {
    (-0.3, 0.1): (1.390, 0.016, 3.039),
    (0.1, 0.2): (1.221, 0.314, 0.395),
    (0.3, 0.2): (1.345, 0.062, 1.387),
    (0.5, 0.2): (1.385, 0.027, 2.350),
}


def summary(m: float, v: float) -> str:
    return f"M={m:.3f} V={v:.3f} SR={(m - 1) / np.sqrt(v):.3f}"


def main(n_episodes: int = 20_000) -> None:
    last = min(2_000, n_episodes)
    for (mu, sigma), paper in PAPER.items():
        env = MeanVarianceEnv(mu=mu, sigma=sigma)
        agent = EMVAgent(env, EMVConfig(sign_aware=True, grad_clip=10.0))
        terminal = agent.train(n_episodes, seed=0)[-last:]
        m_star, v_star = theory.optimal_terminal_moments(env.z, env.x0, env.rho, env.T)
        print(
            f"mu={mu:+.0%} sigma={sigma:.0%}"
            f" | EMV: {summary(terminal.mean(), terminal.var(ddof=1))}"
            f" | paper: M={paper[0]:.3f} V={paper[1]:.3f} SR={paper[2]:.3f}"
            f" | frontier: {summary(m_star, v_star)}",
            flush=True,
        )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20_000)

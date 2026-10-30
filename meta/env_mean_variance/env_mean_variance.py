"""Continuous-time mean-variance portfolio selection environment.

Setting of H. Wang and X. Y. Zhou, "Continuous-time mean-variance portfolio selection:
a reinforcement learning framework", Mathematical Finance 30(4), 2020
(https://arxiv.org/abs/1904.11392).

The market has one risky asset following a geometric Brownian motion with drift ``mu``
and volatility ``sigma``, and a riskless asset with rate ``r``. At each step the agent
chooses ``u``, the discounted dollar amount held in the risky asset, and its discounted
wealth evolves as (eq. 2 of the paper, with an Euler step of length ``dt``)

    x_{t + dt} = x_t + sigma * u * (rho * dt + sqrt(dt) * Z),   rho = (mu - r) / sigma.

The goal is the mean-variance problem: minimize Var[x_T] subject to E[x_T] = z.
Its Lagrangian form is to minimize E[(x_T - w)^2] - (w - z)^2 over policies, for a
Lagrange multiplier ``w``, so the reward is -(x_T - w)^2 at the final step and 0
otherwise. ``w`` can be set in the constructor or with ``reset(options={"w": w})``.

Since the market is a GBM, the optimal policy is known in closed form (see
``theory.py``), which makes this environment a benchmark for checking how close an RL
agent gets to the true optimum.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class MeanVarianceEnv(gym.Env):
    """Mean-variance portfolio selection in a geometric Brownian motion market.

    Observation: ``[t, x]``, the current time and discounted wealth. The market
    parameters ``mu``, ``sigma`` and ``r`` are not observed.

    Action: ``[u]``, the discounted dollar amount held in the risky asset (negative
    values are short positions), clipped to ``[-max_position, max_position]``.

    Reward: ``-(x_T - w)^2`` at the final step, 0 otherwise.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        mu: float = 0.3,
        sigma: float = 0.2,
        r: float = 0.02,
        x0: float = 1.0,
        z: float = 1.4,
        T: float = 1.0,
        dt: float = 1.0 / 252,
        w: Optional[float] = None,
        max_position: float = 1e3,
    ):
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if T <= 0 or dt <= 0 or dt > T:
            raise ValueError("T and dt must be positive with dt <= T")
        self.mu = mu
        self.sigma = sigma
        self.r = r
        self.x0 = x0
        self.z = z
        self.T = T
        self.dt = dt
        self.w = z if w is None else w
        self.max_position = max_position
        self.n_steps = int(round(T / dt))
        self.observation_space = spaces.Box(
            low=np.array([0.0, -np.inf]),
            high=np.array([T, np.inf]),
            dtype=np.float64,
        )
        self.action_space = spaces.Box(
            low=-max_position, high=max_position, shape=(1,), dtype=np.float64
        )
        self._step = 0
        self._x = x0

    @property
    def rho(self) -> float:
        """Market price of risk (Sharpe ratio) (mu - r) / sigma."""
        return (self.mu - self.r) / self.sigma

    @property
    def t(self) -> float:
        return self._step * self.dt

    def _obs(self) -> np.ndarray:
        return np.array([self.t, self._x], dtype=np.float64)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if options is not None and "w" in options:
            self.w = float(options["w"])
        self._step = 0
        self._x = self.x0
        return self._obs(), {"wealth": self._x}

    def step(self, action):
        u = float(
            np.clip(
                np.asarray(action).reshape(-1)[0], -self.max_position, self.max_position
            )
        )
        noise = self.np_random.standard_normal()
        self._x = self._x + self.sigma * u * (
            self.rho * self.dt + np.sqrt(self.dt) * noise
        )
        self._step += 1
        terminated = self._step >= self.n_steps
        reward = -((self._x - self.w) ** 2) if terminated else 0.0
        return self._obs(), float(reward), terminated, False, {"wealth": self._x}

"""Exploratory mean-variance (EMV) agent, Algorithm 1 of Wang and Zhou (2020).

The agent learns from wealth paths only, through ``MeanVarianceEnv.step``; it never
uses ``mu``, ``sigma`` or ``r``. It uses the problem setup (``x0``, ``z``, ``T``,
``dt``), which is known to the investor.

Parametrization (Section 4.2):
    value function   V(t, x) = (x - w)^2 exp(-theta3 (T - t)) + theta2 t^2 + theta1 t + theta0  (44)
    policy entropy   H_t = phi1 + phi2 (T - t)
    with theta3 = 2 phi2, and the policy (46)
    pi(u | t, x) = N(-sqrt(2 phi2 / (lam pi)) e^{(2 phi1 - 1) / 2} (x - w),
                     e^{2 phi2 (T - t) + 2 phi1 - 1} / (2 pi)).
Policy evaluation minimizes the discretized squared temporal-difference error (43)
    C = 1/2 sum_i (Vdot_i - lam H_i)^2 dt,  Vdot_i = (V(t_{i+1}, x_{i+1}) - V(t_i, x_i)) / dt,
and the Lagrange multiplier w follows the self-correcting update (52).

Two optional changes, both off by default so that the defaults follow the paper:

- ``sign_aware``: policy (46) assumes rho > 0 and cannot short, so training diverges
  when the stock is falling. With ``sign_aware=True`` the agent estimates sign(rho) from
  its own trades: sum_i u_i (x_{i+1} - x_i) has expectation sigma rho sum_i u_i^2 dt.
- ``grad_clip``: clip each gradient component to [-grad_clip, grad_clip]. In low Sharpe
  ratio markets phi2 can grow until an update overflows, and clipping prevents this.

In the 8 markets of Table 1 of the paper, over 10 seeds, the defaults diverge in all
seeds of 3 markets (mu = -30% with sigma = 10% and 20%, and mu = 10% with sigma = 20%),
while ``sign_aware=True, grad_clip=10`` converges in all 80 runs and matches the paper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from meta.env_mean_variance.env_mean_variance import MeanVarianceEnv


@dataclass
class EMVConfig:
    """Hyperparameters; defaults are those of Section 5.1 of the paper."""

    lam: float = 2.0
    alpha: float = 0.05
    eta_theta: float = 0.0005
    eta_phi: float = 0.0005
    N: int = 10
    # Initial values are not given in the paper; these start close to doing nothing.
    init_phi1: float = 0.0
    init_phi2: float = 0.01
    init_w: Optional[float] = None  # defaults to the target z
    min_phi2: float = 1e-6  # the paper requires phi2 > 0
    grad_clip: Optional[float] = None
    sign_aware: bool = False


class EMVAgent:
    def __init__(self, env: MeanVarianceEnv, config: Optional[EMVConfig] = None):
        self.env = env
        self.config = config if config is not None else EMVConfig()
        c = self.config
        self.T = env.T
        self.dt = env.dt
        self.times = np.arange(env.n_steps + 1) * env.dt
        self.theta1 = 0.0
        self.theta2 = 0.0
        self.phi1 = c.init_phi1
        self.phi2 = c.init_phi2
        self.w = env.z if c.init_w is None else c.init_w
        self._recent_terminal: List[float] = []
        # running sum of u_i (x_{i+1} - x_i), whose expectation is sigma rho sum u_i^2 dt
        self._trade_signal = 0.0

    # --- policy (46) -------------------------------------------------------------------
    @property
    def sign(self) -> float:
        """Estimated sign of rho; policy (46) assumes rho > 0, i.e. a sign of +1."""
        if not self.config.sign_aware:
            return 1.0
        return 1.0 if self._trade_signal >= 0 else -1.0

    def policy(self, t, x):
        """Mean and variance of the current Gaussian policy at time ``t`` and wealth ``x``."""
        c = self.config
        coef = np.sqrt(2.0 * self.phi2 / (c.lam * np.pi)) * np.exp(
            (2.0 * self.phi1 - 1.0) / 2.0
        )
        mean = -self.sign * coef * (x - self.w)
        var = np.exp(2.0 * self.phi2 * (self.T - t) + 2.0 * self.phi1 - 1.0) / (
            2.0 * np.pi
        )
        return mean, var

    def act(self, obs, rng: np.random.Generator, deterministic: bool = False):
        """Sample an action from the policy (or return its mean if ``deterministic``)."""
        mean, var = self.policy(obs[0], obs[1])
        u = mean if deterministic else mean + np.sqrt(var) * rng.standard_normal()
        return np.array([u])

    # --- one episode -------------------------------------------------------------------
    def run_episode(self, rng: np.random.Generator, seed: Optional[int] = None):
        """Simulate one episode in the environment and return the wealth path."""
        obs, _ = self.env.reset(seed=seed, options={"w": self.w})
        x = np.empty(len(self.times))
        x[0] = obs[1]
        signal = 0.0
        for i in range(len(self.times) - 1):
            action = self.act(obs, rng)
            obs, _, terminated, truncated, _ = self.env.step(action)
            x[i + 1] = obs[1]
            signal += action[0] * (x[i + 1] - x[i])
            if terminated or truncated:
                break
        self._trade_signal += signal
        return x

    # --- policy evaluation (43) and its gradients (47)-(50) -----------------------------
    def value(self, t, x, phi2: Optional[float] = None):
        """V(t, x) of eq. (44) without theta0, which cancels in the TD error."""
        phi2 = self.phi2 if phi2 is None else phi2
        return (
            (x - self.w) ** 2 * np.exp(-2.0 * phi2 * (self.T - t))
            + self.theta2 * t**2
            + self.theta1 * t
        )

    def td_errors(self, x: np.ndarray) -> np.ndarray:
        """e_i = Vdot_i - lam H_i for i = 0, ..., n - 1."""
        t = self.times
        v = self.value(t, x)
        vdot = (v[1:] - v[:-1]) / self.dt
        entropy = self.phi1 + self.phi2 * (self.T - t[:-1])
        return vdot - self.config.lam * entropy

    def loss(self, x: np.ndarray) -> float:
        """C(theta, phi) of eq. (43)."""
        return 0.5 * float(np.sum(self.td_errors(x) ** 2) * self.dt)

    def gradients(self, x: np.ndarray) -> Dict[str, float]:
        """Exact gradients of ``loss`` with respect to theta1, theta2, phi1, phi2 (eqs. 47-50)."""
        t, lam, dt, T = self.times, self.config.lam, self.dt, self.T
        e = self.td_errors(x)
        decay = np.exp(-2.0 * self.phi2 * (T - t))
        # d V(t, x) / d phi2, through theta3 = 2 phi2
        dv_dphi2 = -2.0 * (T - t) * (x - self.w) ** 2 * decay
        dvdot_dphi2 = (dv_dphi2[1:] - dv_dphi2[:-1]) / dt
        return {
            "theta1": float(np.sum(e) * dt),
            "theta2": float(np.sum(e * (t[1:] ** 2 - t[:-1] ** 2))),
            "phi1": float(np.sum(e * -lam) * dt),
            "phi2": float(np.sum(e * (dvdot_dphi2 - lam * (T - t[:-1]))) * dt),
        }

    def update(self, x: np.ndarray) -> None:
        """One gradient step on theta and phi from an episode's wealth path."""
        c = self.config
        g = self.gradients(x)
        if c.grad_clip is not None:
            g = {k: float(np.clip(v, -c.grad_clip, c.grad_clip)) for k, v in g.items()}
        self.theta1 -= c.eta_theta * g["theta1"]
        self.theta2 -= c.eta_theta * g["theta2"]
        self.phi1 -= c.eta_phi * g["phi1"]
        self.phi2 = max(self.phi2 - c.eta_phi * g["phi2"], c.min_phi2)

    def update_lagrange_multiplier(self, terminal_wealth: float) -> None:
        """Self-correcting update (52), using the mean of the last N terminal wealths."""
        c = self.config
        self._recent_terminal.append(terminal_wealth)
        if len(self._recent_terminal) == c.N:
            self.w -= c.alpha * (np.mean(self._recent_terminal) - self.env.z)
            self._recent_terminal.clear()

    # --- training loop -------------------------------------------------------------------
    def train(self, n_episodes: int, seed: Optional[int] = None) -> np.ndarray:
        """Run Algorithm 1 for ``n_episodes`` and return the terminal wealth of each episode."""
        # gymnasium seeds the environment like np.random.default_rng(seed), so using the
        # same seed for the agent would make its exploration noise equal to the market
        # noise. Derive two independent seeds instead.
        env_seed, agent_seed = np.random.SeedSequence(seed).generate_state(2)
        rng = np.random.default_rng(agent_seed)
        terminal = np.empty(n_episodes)
        for k in range(n_episodes):
            x = self.run_episode(rng, seed=int(env_seed) if k == 0 else None)
            self.update(x)
            self.update_lagrange_multiplier(x[-1])
            terminal[k] = x[-1]
        return terminal

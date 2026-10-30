# Continuous-Time Mean-Variance Environment

> **Paper**: H. Wang and X. Y. Zhou, "Continuous-time mean-variance portfolio selection: a reinforcement learning framework," *Mathematical Finance* 30(4), 2020. [[arXiv]](https://arxiv.org/abs/1904.11392)

A Gymnasium environment for mean-variance portfolio selection in continuous time, together with the closed-form optimal policy and the paper's exploratory mean-variance (EMV) RL algorithm as a baseline.

Unlike environments built on historical data, this market (one stock following a geometric Brownian motion, plus a riskless asset) has a **known optimal policy**. You can measure exactly how close an RL agent gets to the optimum, and how that changes with the market's Sharpe ratio, its volatility, or a falling market.

## The problem

The investor starts with wealth `x0` and, at each step of length `dt` up to the horizon `T`, chooses `u`, the (discounted) dollar amount held in the stock. Discounted wealth evolves as

```
x_{t+dt} = x_t + sigma * u * (rho * dt + sqrt(dt) * Z),    rho = (mu - r) / sigma
```

The goal is to minimize `Var[x_T]` subject to `E[x_T] = z`. With a Lagrange multiplier `w`, this is equivalent to minimizing `E[(x_T - w)^2]`, which is the environment's reward:

| | |
|---|---|
| Observation | `[t, x]`: time and discounted wealth. `mu`, `sigma`, `r` are hidden. |
| Action | `[u]`: dollar amount in the stock, negative for a short position. |
| Reward | `-(x_T - w)^2` at the last step, 0 otherwise. Set `w` in the constructor or with `env.reset(options={"w": w})`. |

## Files

| File | What it is |
|---|---|
| `env_mean_variance.py` | `MeanVarianceEnv` (Gymnasium) |
| `theory.py` | Closed-form results: optimal `w*`, classical and exploratory optimal policies, efficient frontier `Var[x_T] = (z - x0)^2 / (e^{rho^2 T} - 1)` |
| `emv_agent.py` | `EMVAgent`, Algorithm 1 of the paper. It learns only through `env.step`. |
| `example_emv.py` | Trains EMV in four markets and compares it with the optimal policy |

```python
from meta.env_mean_variance import EMVAgent, EMVConfig, MeanVarianceEnv

env = MeanVarianceEnv(mu=0.3, sigma=0.2)  # target z = 1.4 from x0 = 1 over T = 1 year
agent = EMVAgent(env, EMVConfig(sign_aware=True, grad_clip=10.0))
terminal_wealth = agent.train(n_episodes=20_000, seed=0)
```

Any other agent (e.g. from Stable-Baselines3 or ElegantRL) can be trained on `MeanVarianceEnv` directly, and compared with `theory.classical_policy`.

## EMV and two optional changes

`EMVConfig()` follows the paper (Section 5.1 hyperparameters). Two changes are available and off by default:

- **`sign_aware=True`**: the policy in eq. (46) of the paper can only hold the stock long below the target, so it cannot follow the optimal policy when the stock is falling (`rho < 0`), and training diverges. With this option the agent estimates the sign of `rho` from its own trades (`sum_i u_i (x_{i+1} - x_i)` has expectation `sigma * rho * sum_i u_i^2 dt`), using nothing it does not observe.
- **`grad_clip=10.0`**: in low Sharpe ratio markets, the exploration parameter `phi2` can grow until an update overflows. Clipping each gradient component prevents this.

Over 10 seeds in the 8 markets of Table 1 of the paper, the default EMV diverges in every seed of 3 markets (`mu = -30%` with `sigma = 10%` and `20%`, and `mu = 10%, sigma = 20%`). With both options, all 80 runs converge, and the Sharpe ratios match the paper, e.g. 0.395 ± 0.021 against 0.395 for `mu = 10%, sigma = 20%`. In the 5 other markets both versions give the same results. The full study is at [anveshachuri/emv-rl](https://github.com/anveshachuri/emv-rl).

## Example output

`python -m meta.env_mean_variance.example_emv` (20,000 episodes, last 2,000 terminal wealths; Sharpe ratio is `(M - 1) / sqrt(V)`):

| mu | sigma | EMV (M / V / SR) | Paper, Table 1 | Efficient frontier |
|---|---|---|---|---|
| -30% | 10% | 1.388 / 0.015 / 3.199 | 1.390 / 0.016 / 3.039 | 1.400 / 0.000 / 167.3 |
| 10% | 20% | 1.350 / 0.929 / 0.363 | 1.221 / 0.314 / 0.395 | 1.400 / 0.922 / 0.417 |
| 30% | 20% | 1.370 / 0.073 / 1.372 | 1.345 / 0.062 / 1.387 | 1.400 / 0.026 / 2.470 |
| 50% | 20% | 1.382 / 0.027 / 2.333 | 1.385 / 0.027 / 2.350 | 1.400 / 0.001 / 17.79 |

The efficient frontier is the best variance any policy can reach with `E[x_T] = z` when `mu` and `sigma` are known. EMV stays above it because its Gaussian policy keeps exploring, which is the cost of learning without knowing the market. When the market's Sharpe ratio is high, the frontier variance is close to 0, so even small amounts of exploration make a large difference in Sharpe ratio. When it is low (`mu = 10%, sigma = 20%`), EMV is close to the frontier.

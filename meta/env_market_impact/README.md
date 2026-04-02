# Market Impact Environments for FinRL

> **Paper**: L. R. Abbade and A. H. R. Costa, "Realistic Market Impact Modeling for Reinforcement Learning Trading Environments," arXiv:2603.29086, 2026. [[arXiv]](https://arxiv.org/abs/2603.29086) [[PDF]](https://arxiv.org/pdf/2603.29086)

This directory contains improved versions of FinRL's trading environments that incorporate realistic market impact models. These environments are designed to provide a more accurate simulation of real-world trading by accounting for the costs associated with executing large orders.

## Key Features

1.  **Market Impact Modeling**: Transaction costs use realistic models based on trade size, market volume, and volatility — not a fixed percentage.

2.  **Permanent vs. Temporary Impact**: The models distinguish between temporary impact (immediate cost of demanding liquidity) and permanent impact (persistent price effect, tracked over time with exponential decay).

3.  **Dynamic Position Sizing**: Position sizes are calculated as a percentage of portfolio value, with volume constraints (max % of daily volume per trade).

4.  **Pluggable Impact Models**: All environments accept any `ImpactModel` subclass, enabling apples-to-apples comparison of cost assumptions on the same agent/reward/state.

5.  **Comprehensive Trade Logging**: Every trade records shares, notional, POV (Participation of Volume), and turnover percentile for detailed post-hoc analysis.

6.  **Hyperparameter Optimization**: Optuna-based HPO scripts for each environment, searching over both environment and DRL algorithm hyperparameters.

## Environments

### `env_mace_stock_trading.py` — MACE (our contribution)

MACE (Market-Adjusted Cost Execution) is an enhanced stock trading environment with a Differential Sharpe Ratio (DSR) reward, drawdown penalty, online observation normalization, and full impact model integration. The agent's continuous actions in `[-1, 1]` per stock are converted to share quantities respecting position and volume limits.

**Key features**: DSR reward with drawdown penalty, observation normalizer (`RunningMeanStd`) with train→test state transfer, position limits (% of portfolio), volume constraints, cooldown tracking, risk-free rate accrual, permanent impact in observations.

**Runner**: `example_mace_env.py` | **HPO**: `optuna_optimize.py`

### `env_margin_trader_impact.py` — Margin Trading (Gu et al. 2023)

Faithful adaptation of the "Margin Trader" paper (ICAIF '23). The agent manages long and short positions with leverage and margin constraints. **The original reward function is preserved**: `R_t = λ₁ × profit + λ₂ × rolling_sharpe` (Section 4.2.3), along with the original state representation (Section 4.2.1).

The paper explicitly assumes **zero market impact** and **0.1% (10 bps) transaction cost**. By plugging in the Almgren-Chriss model instead, we measure how much performance degrades under realistic costs.

**Paper parameters preserved**:
- Reward: `λ₁=1e-5`, `λ₂` tuned in {0.0001...0.05}, rolling Sharpe over 5 steps
- Margin rate `k=2` (2x leverage), maintenance margin 30%/40%
- Margin Adjustment Module every 30 steps (Section 4.3.2)
- Maintenance Detection Module with warning/strict levels (Section 4.3.3)
- Algorithms: A2C, PPO, DDPG, SAC (Table 1)

**Added for impact comparison**: pluggable `ImpactModel`, `end_day()` decay, permanent impact in state, trade-level logging (POV, turnover percentile).

**Runner**: `example_margin_trader.py` | **HPO**: `optuna_margin_trader.py`

### `env_portfolio_optimization_impact.py` — Portfolio Optimization (Costa & Costa 2023)

Faithful adaptation of the POE paper (BWAIF '23). The agent outputs target portfolio weights via softmax normalization; the environment computes the rebalancing trades. **The original log-return reward is preserved**: `r_t = ln(V_t / V_{t-1})`.

The paper explicitly assumes **no market impact** and provides an environment framework without running RL experiments. We use their environment design and add impact model integration to measure cost effects.

**Paper design preserved**: softmax action normalization, weight-based rebalancing, log-return reward, portfolio memory tracking.

**Added for impact comparison**: config-dict input (compatible with `MarketDataPreparator`), pluggable `ImpactModel`, `end_day()` decay, permanent impact in state, trade-level logging, flat `Box` observation space for SB3 compatibility.

**Runner**: `example_poe.py` | **HPO**: `optuna_poe.py`

### `impact_models.py` — Market Impact Models

| Model | Description |
|---|---|
| `SqrtImpactModel` | Square-root law impact (empirically validated, default) |
| `ACImpactModel` | Classic Almgren-Chriss with permanent + temporary decomposition |
| `OWImpactModel` | Obizhaeva-Wang with transient impact decay (LOB resilience) |
| `BaselineImpactModel` | Fixed 10 bps fee — the naive assumption used by most RL papers |

All models support permanent impact decay (exponential with configurable half-life).

## Supporting Infrastructure

| File | Purpose |
|---|---|
| `market_data.py` | Data download, preprocessing, caching, train/trade splitting |
| `backtest_config.py` | Shared configuration for stock trading env (`BacktestParams`, `MODEL_KWARGS`, `NET_ARCH`) |
| `backtest_report_generator.py` | Interactive Plotly HTML reports from backtest results |
| `utils.py` | Logger setup, `compute_performance_stats()` (single source of truth for metrics) |

## Usage

### Quick Start

```bash
# Stock trading (our env with DSR reward)
python example_mace_env.py

# Margin trading (Gu et al. 2023 reward, paper params)
python example_margin_trader.py

# Portfolio optimization (Costa & Costa 2023 log-return reward)
python example_poe.py
```

### Hyperparameter Optimization

```bash
# Stock trading — searches over 5 agents × env + model params
python optuna_optimize.py

# Margin trading — searches over env + model params (paper's ranges)
python optuna_margin_trader.py

# Portfolio optimization
python optuna_poe.py
```

### Comparison Protocol

The comparison grid for each environment runs:
1. **Baseline cost model**: `BaselineImpactModel(basis_points=10)` — the 10 bps fixed fee assumed by most papers.
2. **Almgren-Chriss cost model**: `ACImpactModel()` — realistic nonlinear costs based on trade size, volatility, and volume.

Both use the **same reward function, state, and agent hyperparameters**. The only variable is the cost model, isolating the effect of realistic market impact on performance.

## Experimental Setup

- **Universe**: NASDAQ 100 (static ~2021 composition to reduce survivorship bias)
- **Benchmark**: QQEW (equal-weighted NASDAQ 100 ETF, comparable given 2% per-stock cap)
- **Initial capital**: $1B (needed for measurable market impact on large-cap equities)
- **Data**: 2010–2026, 90/10 train/test split. 2025-2026 used for final test only, HPO ran only up to 2025.
- **Algorithms**: A2C, PPO, DDPG, SAC, TD3 (via Stable-Baselines3)
- **HPO**: Optuna with TPE sampling and median pruning; objective = best OOS annualized Sharpe across epochs

## Key Results

### MACE Stock Trading (A2C with Optuna HPO)

- Optimized A2C narrows the IS–OOS gap: IS performance is lower but OOS is higher, indicating better generalization.
- AC model reduces OOS trading costs by ~20%; the agent shifts toward more liquid names.
- Average order POV drops ~20% with optimized params; cash allocation drops from ~10% to ~1.5%.

### Margin Trading (A2C, PPO, DDPG, SAC — default params)

- All agents underperform the 19% QQEW benchmark OOS, though A2C and PPO outperform IS (suggesting overfitting; HPO candidates).
- **PPO**: Best margin result at 15% OOS (baseline), but drops to 9% under AC despite 40% lower trading costs — the nonlinear cost signal disrupts its aggressive policy.
- **DDPG**: Dramatic improvement under AC — Sharpe goes from −2.1 to 0.3, max drawdown from −23% to −6%, with similar trading costs/POV.
- **SAC**: Degrades under AC — Sharpe from −0.5 to −1.2, POV drops (0.26→0.15) yet costs increase ~10%.
- Training dynamics: A2C peaks at epoch ~15 then degrades; PPO IS/OOS diverge with POV spiking mid-training; DDPG and SAC show no epoch-to-epoch evolution.

### Portfolio Optimization (A2C, PPO, DDPG, SAC, TD3 — Optuna HPO)

- All five agents outperform the 19% QQEW benchmark OOS.
- **TD3** is both best and worst depending on cost model: 32% OOS return with AC (best overall), 26% with baseline (worst among agents). Sharpe improves from 0.9 to 1.1 under AC.
- **A2C** ~30% on both models; **SAC** 30%/29%; **DDPG** 28%/27%.
- **PPO** is the only agent hurt by AC: 31% (baseline) → 28% (AC).
- Trading costs ~15% lower under AC for TD3, ~30% lower for other agents.
- **HPO impact on training**: Optimized TD3 converges (IS ↓, OOS ↑ across epochs); baseline TD3 stays flat. PPO baseline params produce runaway POV (IS reaching 1.39); optimized stays flat at 0.36.
- **Cost model impact on convergence**: TD3 OOS return converges upward under AC (reaching 19.35%) but stays flat under baseline (~16%), despite higher turnover/POV under AC — the nonlinear cost signal produces a more generalizable policy.

## References

-   **This work**: Abbade, L. R. & Costa, A. H. R. (2026). Realistic Market Impact Modeling for Reinforcement Learning Trading Environments. [arXiv:2603.29086](https://arxiv.org/abs/2603.29086).
-   **Almgren-Chriss**: Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions. *Journal of Risk*, 3(2), 5–39.
-   **Square-Root Law**: Tóth, B., et al. (2011). Anomalous price impact and the critical nature of liquidity. *Physical Review X*, 1(2), 021006.
-   **Obizhaeva-Wang**: Obizhaeva, A. & Wang, J. (2013). Optimal trading strategy and supply/demand dynamics. *Journal of Financial Markets*, 16(1), 1–32.
-   **Margin Trader**: Gu, J., et al. (2023). Margin Trader: A Reinforcement Learning Framework for Portfolio Management with Margin and Constraints. *ICAIF '23*, pp. 610–618.
-   **POE**: Costa, C. & Costa, A. (2023). POE: A General Portfolio Optimization Environment for FinRL. *BWAIF '23*, pp. 132–143.
-   **DSR**: Moody, J. & Saffell, M. (1998). Reinforcement Learning for Trading. *NeurIPS 11*.
-   **FinRL**: Liu, X.-Y., et al. (2020). FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading. arXiv:2011.09607.
-   **Gymnasium**: Brockman, G., et al. (2016). OpenAI Gym. arXiv:1606.01540.
-   **Optuna**: Akiba, T., et al. (2019). Optuna: A Next-Generation Hyperparameter Optimization Framework. *KDD '19*, pp. 2623–2631.
-   **Stable-Baselines3**: Raffin, A., et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR*, 22(268), 1–8.

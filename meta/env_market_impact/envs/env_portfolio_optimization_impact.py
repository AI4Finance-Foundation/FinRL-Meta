"""
Portfolio Optimization Environment with Market Impact

This environment adapts the POE (Portfolio Optimization Environment) from
Costa & Costa (2023) to the FinRL-Meta framework, incorporating a realistic
market impact model while preserving the original reward function (log return)
and the weight-based action paradigm from the paper.

The impact model integration (permanent impact tracking, ``end_day()`` decay,
trade-level logging) and the runner interface (normalizer stubs, ``total_asset``,
``cash``, etc.) are added on top of the original formulation to enable
consistent backtesting and comparison against the stock-trading environment.

Accounting model
~~~~~~~~~~~~~~~~
The portfolio is tracked via explicit **cash** and **holdings** (share
counts).  Portfolio value is ``cash + sum(holdings * prices)`` using market
prices.  After each rebalancing step, *actual* weights are derived from
holdings so that the next step's price-variation correctly reflects the
real portfolio composition (not just the softmax target).

Reference:
    Costa, C., & Costa, A. (2023). POE: A General Portfolio Optimization
    Environment for FinRL. In Anais do II Brazilian Workshop on Artificial
    Intelligence in Finance (pp. 132-143). SBC.
    https://doi.org/10.5753/bwaif.2023.231144
"""

from typing import Dict
from typing import Optional
from typing import Tuple

import gymnasium as gym
import numpy as np

from .impact_models import ImpactModel
from .impact_models import SqrtImpactModel

EPS = 1e-8


class PortfolioOptimizationImpactEnv(gym.Env):
    """
    A portfolio optimization environment that incorporates market impact.

    Preserves the original Costa & Costa (2023) reward (log return) and the
    softmax weight-based action space, while adding:

    * Config-dict input (compatible with ``MarketDataPreparator``).
    * Pluggable ``ImpactModel`` with ``end_day()`` decay and permanent
      impact tracked via the model.
    * Holdings-based accounting: cash and share counts are tracked
      explicitly so that portfolio value and weights always reflect
      the actual post-trade composition.
    * Trade-level logging (POV, turnover percentile) for post-hoc analysis.
    * Normalizer-state stubs so the runner and Optuna scripts work unchanged.
    * ``total_asset`` / ``cash`` / ``time`` / ``max_step`` attributes
      expected by the backtesting pipeline.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config: Dict,
        initial_amount: float = 1e6,
        reward_scaling: float = 1.0,
        max_trade_volume_pct: float = 0.1,
        impact_model: Optional[ImpactModel] = None,
        initial_portfolio_state: Optional[Dict] = None,
    ) -> None:
        self.date_list = config["date_list"]
        price_array = config["price_array"]
        tech_array = config["tech_array"]
        self.volatility_array = config.get(
            "volatility_array", np.ones_like(price_array) * 0.02
        )
        self.volume_array = config.get("volume_array", np.ones_like(price_array) * 1e6)

        self.price_array = price_array.astype(np.float32)
        self.tech_array = tech_array.astype(np.float32) * 2**-7

        self.stock_dim = self.price_array.shape[1]
        self.stock_symbols = config["tic_list"]
        self._initial_amount = initial_amount
        self.initial_capital = initial_amount  # runner compatibility
        self._reward_scaling = reward_scaling
        self._max_trade_volume_pct = max_trade_volume_pct
        self.impact_model = (
            impact_model if impact_model is not None else SqrtImpactModel()
        )
        self.initial_portfolio_state = initial_portfolio_state

        # Action: raw logits for (cash + N stocks) -> softmax
        action_space_dim = 1 + self.stock_dim

        # State: [price_variation(N), current_weights(1+N),
        #         last_action(1+N), perm_impact(N), tech(T)]
        self.state_dim = (
            self.stock_dim  # price variation
            + action_space_dim  # current weights (cash + stocks)
            + action_space_dim  # last action
            + self.stock_dim  # permanent impact
            + self.tech_array.shape[1]  # tech indicators
        )

        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(action_space_dim,), dtype=np.float32
        )
        self.max_step = self.price_array.shape[0] - 1

    # ── Permanent impact helpers ──────────────────────────────────────

    def _get_perm_impact(self) -> np.ndarray:
        return self.impact_model.get_perm_state_array(self.stock_symbols)

    # ── Normalizer stubs (for runner compatibility) ───────────────────

    def get_normalizer_state(self) -> Optional[Dict]:
        return None

    def set_normalizer_state(self, state: Dict, freeze: bool = True) -> None:
        pass

    # ── Portfolio state transfer (train → OOS continuation) ────────────

    def get_portfolio_state(self) -> Dict:
        """Return current portfolio state for env-to-env transfer.

        Pass the returned dict as ``initial_portfolio_state`` when
        constructing the OOS environment so that the test run continues
        from the exact end-of-training cash and holdings.
        """
        return {
            "cash": self._cash_balance,
            "holdings": self.holdings.copy(),
        }

    # ── Turnover percentile helper ────────────────────────────────────

    def _calculate_turnover_percentiles(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> np.ndarray:
        gross_notional = prices * volumes
        n = len(gross_notional)
        if n <= 1:
            return np.full(n, 50.0)
        sorted_indices = np.argsort(gross_notional)
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(1, n + 1)
        return (ranks - 1) / (n - 1) * 100

    # ── Softmax normalisation (original paper, numerically stable) ────

    def _softmax_normalization(self, actions: np.ndarray) -> np.ndarray:
        """Normalizes the action vector using the softmax function."""
        exp_actions = np.exp(actions - np.max(actions))
        return exp_actions / np.sum(exp_actions)

    # ── Compute actual weights from cash + holdings ───────────────────

    def _compute_weights(self, prices: np.ndarray) -> None:
        """Recompute ``_current_weights`` from explicit cash and holdings."""
        stock_values = self.holdings * prices
        self._portfolio_value = self._cash_balance + stock_values.sum()
        self.total_asset = self._portfolio_value

        if self._portfolio_value > EPS:
            self._current_weights[0] = self._cash_balance / self._portfolio_value
            self._current_weights[1:] = stock_values / self._portfolio_value
        else:
            self._current_weights[0] = 1.0
            self._current_weights[1:] = 0.0

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0

        if self.initial_portfolio_state is not None:
            # Continued-from-training: restore cash and holdings
            ps = self.initial_portfolio_state
            self._cash_balance = ps["cash"]
            self.holdings = ps["holdings"].copy().astype(np.float32)

            # Derive portfolio value from actual state at day-0 prices
            day0_prices = self.price_array[self.time]
            stock_values = self.holdings * day0_prices
            self._portfolio_value = self._cash_balance + stock_values.sum()
        else:
            # Fresh start: 100% cash
            self._cash_balance = self._initial_amount
            self._portfolio_value = self._initial_amount
            self.holdings = np.zeros(self.stock_dim, dtype=np.float32)

        self.total_asset = self._portfolio_value
        self.cash = self._cash_balance

        self._current_weights = np.zeros(1 + self.stock_dim, dtype=np.float32)
        if self.initial_portfolio_state is not None:
            self._compute_weights(self.price_array[self.time])
        else:
            self._current_weights[0] = 1.0  # 100% cash

        self._actions_memory = [self._current_weights.copy()]
        self._final_weights = [self._current_weights.copy()]
        self._asset_memory = {
            "initial": [self._portfolio_value],
            "final": [self._portfolio_value],
        }
        self._portfolio_return_memory = [0.0]
        self._date_memory = [self.date_list[self.time]]

        self.episode_return = 0.0

        if options is None or options.get("reset_impact_model", True):
            self.impact_model.reset()

        return self._build_state(), {}

    # ── Step ──────────────────────────────────────────────────────────

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.time += 1
        terminated = self.time >= self.max_step

        curr_prices = self.price_array[self.time]
        volatility = self.volatility_array[self.time]
        volume = self.volume_array[self.time]

        # ── 1. Mark-to-market at new prices (before rebalancing) ──────
        stock_values_pre = self.holdings * curr_prices
        value_pre = self._cash_balance + stock_values_pre.sum()
        self._asset_memory["initial"].append(value_pre)

        # ── 2. Target weights from softmax ────────────────────────────
        weights = self._softmax_normalization(actions)
        self._actions_memory.append(weights)

        # ── 3. Compute rebalancing trades (in shares) ─────────────────
        target_stock_values = value_pre * weights[1:]
        trades_in_value = target_stock_values - stock_values_pre
        trades_in_shares = np.divide(
            trades_in_value,
            curr_prices,
            out=np.zeros_like(trades_in_value),
            where=curr_prices > 0,
        ).astype(int)

        # Volume constraint
        volume_limit = (volume * self._max_trade_volume_pct).astype(int)
        trades_in_shares = np.clip(trades_in_shares, -volume_limit, volume_limit)

        # Can't sell more shares than currently held (no short selling)
        for i in range(self.stock_dim):
            if trades_in_shares[i] < 0:
                trades_in_shares[i] = max(trades_in_shares[i], -int(self.holdings[i]))

        # Turnover percentiles for logging
        turnover_percentiles = self._calculate_turnover_percentiles(curr_prices, volume)

        # ── 4. Execute trades through impact model ────────────────────
        total_impact_cost = 0.0
        total_traded_value = 0.0
        total_buy_value = 0.0
        total_sell_value = 0.0
        trades = []

        for i in range(self.stock_dim):
            ts = trades_in_shares[i]
            if ts == 0:
                continue

            impact = self.impact_model.apply_trade(
                ts,
                curr_prices[i],
                volatility[i],
                volume[i],
                self.stock_symbols[i],
            )
            total_impact_cost += impact.cost
            notional = abs(ts) * curr_prices[i]
            pov = abs(ts) / volume[i] if volume[i] > 0 else 0.0
            side = "buy" if ts > 0 else "sell"

            if ts > 0:
                total_buy_value += notional
            else:
                total_sell_value += notional
            total_traded_value += notional

            trades.append(
                {
                    "stock_idx": i,
                    "side": side,
                    "shares": abs(ts),
                    "notional": notional,
                    "pov": pov,
                    "turnover_percentile": turnover_percentiles[i],
                }
            )

            # Update cash: buy costs cash, sell generates cash
            self._cash_balance -= ts * curr_prices[i]
            self._cash_balance -= impact.cost
            self.holdings[i] += ts

        # ── 5. Decay permanent impact ─────────────────────────────────
        self.impact_model.end_day(self.date_list[self.time])

        # ── 6. Post-trade portfolio value and actual weights ──────────
        self._compute_weights(curr_prices)
        self.cash = self._cash_balance

        self._final_weights.append(self._current_weights.copy())
        self._asset_memory["final"].append(self._portfolio_value)

        # ── 7. Reward: log return (original paper) ────────────────────
        last_value = self._asset_memory["final"][-2]
        rate_of_return = self._portfolio_value / last_value if last_value > EPS else 1.0
        portfolio_return = rate_of_return - 1
        log_return = np.log(max(rate_of_return, EPS))

        self._portfolio_return_memory.append(portfolio_return)
        self._date_memory.append(self.date_list[self.time])
        reward = log_return * self._reward_scaling

        if terminated:
            self.episode_return = (
                self.total_asset / self._initial_amount
                if self._initial_amount > 0
                else 0.0
            )

        turnover = total_traded_value / self.total_asset if self.total_asset > 0 else 0
        info = {
            "turnover": turnover,
            "cost": total_impact_cost,
            "total_buy_value": total_buy_value,
            "total_sell_value": total_sell_value,
            "cash": self.cash,
            "trades": trades,
        }

        return self._build_state(), reward, terminated, False, info

    # ── State construction ────────────────────────────────────────────

    def _build_state(self) -> np.ndarray:
        """Build the flat observation vector.

        Components (original POE spirit, adapted for config-dict input):
        * price_variation(N): ratio curr_price / prev_price  (1.0 at t=0)
        * current_weights(1+N): actual cash weight + stock weights
        * last_action(1+N): the previous softmax weights (target)
        * permanent_impact(N): cumulative impact from the model
        * tech(T): technical indicators
        """
        if self.time == 0:
            price_var = np.ones(self.stock_dim, dtype=np.float32)
        else:
            prev = self.price_array[self.time - 1]
            curr = self.price_array[self.time]
            price_var = np.where(prev > 0, curr / prev, 1.0).astype(np.float32)

        perm_impact = self._get_perm_impact()

        state_components = [
            price_var,
            self._current_weights,
            self._actions_memory[-1],
            perm_impact,
            self.tech_array[min(self.time, self.max_step)],
        ]
        return np.hstack(state_components).astype(np.float32)

    def render(self, mode="human"):
        return self.total_asset

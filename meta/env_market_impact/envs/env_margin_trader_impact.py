"""
Margin Trading Environment with Market Impact

This environment adapts the "Margin Trader" paper by Gu et al. (2023) to the
FinRL-Meta framework, incorporating a realistic market impact model while
preserving the original reward function and state representation from the paper.

Original reward (Section 4.2.3):
    R_t = lambda_1 * profit_t + lambda_2 * risk_t
    profit_t = e_{t+1} - e_t   (equity change)
    risk_t   = annualized rolling Sharpe ratio over the last ``sharpe_window`` steps

Original state (Section 4.2.1):
    S_t = [b^l_t, l^l_t, e^l_t, l^s_t, b^s_t, e^s_t, p_t, n_t, t_t]

    b^l = available cash balance in long position
    l^l = loan (cash borrowed from broker)
    e^l = long equity = b^l + long_market_value - l^l
    l^s = available limit in short position (remaining shortable market value)
    b^s = credit balance (initial borrowing limit + initial deposit)
    e^s = short equity = b^s - l^s - short_market_value

    We append permanent_impact and cooldown arrays from the impact model.

The action space is ``stock_dim`` (one action per stock).  Positive actions
buy (or cover short then buy long); negative actions sell (or sell long then
open short).  This automatic position-flipping replaces the paper's dual-action
(2N) + Position Fusion Module design while being functionally equivalent.

The impact model integration (permanent impact tracking, ``end_day()`` decay,
trade-level logging) and the runner interface (normalizer stubs, ``total_asset``,
``cash``, etc.) are added on top of the original formulation.

Reference:
    Gu, J., Du, W., Rahman, A. M. M., & Wang, G. (2023).
    Margin Trader: A Reinforcement Learning Framework for Portfolio
    Management with Margin and Constraints.
    In Proceedings of the Fourth ACM International Conference on AI in
    Finance (pp. 610-618).
"""

from collections import deque
from typing import Dict
from typing import Optional
from typing import Tuple

import gymnasium as gym
import numpy as np

from .impact_models import ImpactModel
from .impact_models import SqrtImpactModel

EPS = 1e-8


class MarginTraderImpactEnv(gym.Env):
    """
    A margin trading environment that incorporates a market impact model.

    Preserves the Gu et al. (2023) six-variable margin accounting
    (``cash, loan, long_equity, limit, credit, short_equity``), reward
    function (``lambda_1 * profit + lambda_2 * annualized_rolling_sharpe``),
    Margin Adjustment Module, and Maintenance Detection Module, while adding
    pluggable impact models, trade-level logging, and runner-compatible
    interface attributes.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        config: Dict,
        initial_capital: float = 1e9,
        max_stock_pct: float = 0.02,
        initial_stocks: Optional[np.ndarray] = None,
        # ── Paper margin parameters (Section 4.1) ─────────────────
        margin_rate: float = 2.0,
        long_short_ratio: float = 1.0,
        maintenance_margin: float = 0.3,
        maintenance_warning: float = 0.4,
        # ── Trade constraints ──────────────────────────────────────
        max_trade_volume_pct: float = 0.1,
        impact_model: Optional[ImpactModel] = None,
        # ── Paper reward parameters (Section 4.2.3 & 5.4) ─────────
        lambda_1: float = 1e-5,
        lambda_2: float = 0.01,
        sharpe_window: int = 5,
        # ── Margin adjustment period (Section 4.3.2) ──────────────
        margin_adjust_period: int = 30,
        # ── State transfer (continued-from-training OOS runs) ─────
        initial_margin_state: Optional[Dict] = None,
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
        self.max_stock_pct = max_stock_pct
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(self.stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks.astype(np.float32)
        )

        self.margin_rate = margin_rate
        self.long_short_ratio = long_short_ratio
        self.maintenance_margin = maintenance_margin
        self.maintenance_warning = maintenance_warning
        self.max_trade_volume_pct = max_trade_volume_pct
        self.margin_adjust_period = margin_adjust_period
        self.impact_model = (
            impact_model if impact_model is not None else SqrtImpactModel()
        )

        self.initial_margin_state = initial_margin_state
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.sharpe_window = sharpe_window

        # State: [cash, loan, long_equity, limit, credit, short_equity,
        #         prices(N), holdings(N), perm_impact(N), cooldown(N), tech(T)]
        self.state_dim = 6 + (4 * self.stock_dim) + self.tech_array.shape[1]
        self.action_dim = self.stock_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )
        self.max_step = self.price_array.shape[0] - 1

    # ── Permanent impact helpers ──────────────────────────────────────

    def _get_perm_impact(self) -> np.ndarray:
        return self.impact_model.get_perm_state_array(self.stock_symbols)

    # ── Normalizer stubs (runner compatibility) ───────────────────────

    def get_normalizer_state(self) -> Optional[Dict]:
        return None

    def set_normalizer_state(self, state: Dict, freeze: bool = True) -> None:
        pass

    # ── Margin state transfer (train → OOS continuation) ───────────────

    def get_margin_state(self) -> Dict:
        """Return current margin accounting state for env-to-env transfer.

        Pass the returned dict as ``initial_margin_state`` when constructing
        the OOS environment so that the test run continues from the exact
        end-of-training portfolio composition.
        """
        return {
            "long_cash": self.long_cash,
            "loan": self.loan,
            "long_equity": self.long_equity,
            "short_limit": self.short_limit,
            "short_credit": self.short_credit,
            "short_equity": self.short_equity,
            "stocks": self.stocks.copy(),
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

    # ── Rolling Sharpe (paper Section 4.2.3) ──────────────────────────

    def _rolling_sharpe(self) -> float:
        """Annualized Sharpe ratio over the last ``sharpe_window`` equity
        snapshots, matching the paper's formulation with sqrt(252) scaling."""
        if len(self._equity_history) < 2:
            return 0.0
        equities = np.array(self._equity_history)
        returns = np.diff(equities) / (equities[:-1] + EPS)
        std = returns.std()
        if std < EPS:
            return 0.0
        return float((252**0.5) * returns.mean() / std)

    # ── Maintenance ratio checks (paper Section 4.3.3) ────────────────

    def _check_long_maintenance(self, price: np.ndarray) -> float:
        """equity / long_market_value.  Returns 1.0 when no long holdings."""
        adjusted = price + self._get_perm_impact()
        long_mask = self.stocks > 0
        if not long_mask.any():
            return 1.0
        long_mv = (self.stocks[long_mask] * adjusted[long_mask]).sum()
        if long_mv <= 0:
            return 1.0
        return self.long_equity / long_mv

    def _check_short_maintenance(self, price: np.ndarray) -> float:
        """equity / short_market_value.  Returns 1.0 when no short holdings."""
        adjusted = price + self._get_perm_impact()
        short_mask = self.stocks < 0
        if not short_mask.any():
            return 1.0
        short_mv = abs((self.stocks[short_mask] * adjusted[short_mask]).sum())
        if short_mv <= 0:
            return 1.0
        return self.short_equity / short_mv

    # ── Reset ─────────────────────────────────────────────────────────

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self.time = 0
        price = self.price_array[self.time]

        if self.initial_margin_state is not None:
            # Continued-from-training: restore full margin accounting state
            ms = self.initial_margin_state
            self.long_cash = ms["long_cash"]
            self.loan = ms["loan"]
            self.long_equity = ms["long_equity"]
            self.short_limit = ms["short_limit"]
            self.short_credit = ms["short_credit"]
            self.short_equity = ms["short_equity"]
            self.stocks = ms["stocks"].copy().astype(np.float32)
        else:
            # Fresh start: derive margin accounting from initial_capital
            # (paper Section 4.1)
            r = self.long_short_ratio / (self.long_short_ratio + 1)
            equity_long = r * self.initial_capital
            equity_short = self.initial_capital - equity_long

            # Long account (paper: b^l = k*e_l, l^l = (k-1)*e_l, e^l = e_l)
            self.long_cash = equity_long * self.margin_rate
            self.loan = equity_long * (self.margin_rate - 1)
            self.long_equity = equity_long

            # Short account (paper: l^s = k*e_s, b^s = (k+1)*e_s, e^s = e_s)
            self.short_limit = equity_short * self.margin_rate
            self.short_credit = (self.margin_rate + 1) * equity_short
            self.short_equity = equity_short

            self.stocks = self.initial_stocks.copy()

        self.stocks_cool_down = np.zeros_like(self.stocks)

        if options is None or options.get("reset_impact_model", True):
            self.impact_model.reset()

        self.total_asset = self.long_equity + self.short_equity
        self.initial_total_asset = self.total_asset
        self.cash = max(self.long_cash - self.loan, 0.0)
        self.episode_return = 0.0

        self._equity_history: deque = deque(maxlen=self.sharpe_window + 1)
        self._equity_history.append(self.total_asset)

        return self.get_state(price), {}

    # ── Position sizing (mirrors MACEStockTradingEnv) ────────────────

    def _calculate_trade_shares(
        self,
        actions: np.ndarray,
        prices: np.ndarray,
        volume: np.ndarray,
        total_equity: float,
    ) -> np.ndarray:
        """Compute integer trade-share vector respecting position limits.

        ``max_stock_pct`` defines the **maximum position** (not trade) as a
        fraction of total equity.  This ensures equal notional exposure per
        stock regardless of price and produces trades large enough for
        observable market impact.

        For each stock the target position is clamped to
        ``[-max_shares, +max_shares]`` so that:
        * Going from 0 → max (action = 1) trades ``max_stock_pct`` of equity.
        * A full reversal from max-short → max-long trades 2× that amount.
        * If the position already exceeds the limit (e.g. due to price
          appreciation), a forced reduction is triggered.
        """
        max_position_value = max(total_equity * self.max_stock_pct, 0.0)
        max_shares = np.where(prices > 0, max_position_value / prices, 0).astype(int)

        desired_trade = (actions * max_shares).astype(int)
        trade_shares = np.zeros(self.stock_dim, dtype=int)

        for i in range(self.stock_dim):
            current = int(self.stocks[i])
            limit = int(max_shares[i])
            target_pos = current + desired_trade[i]
            clamped_pos = max(-limit, min(target_pos, limit))
            trade_shares[i] = clamped_pos - current

        volume_limit = (volume * self.max_trade_volume_pct).astype(int)
        trade_shares = np.clip(trade_shares, -volume_limit, volume_limit)
        return trade_shares

    # ── Step ──────────────────────────────────────────────────────────

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        self.time += 1
        self.stocks_cool_down += 1
        done = self.time >= self.max_step

        trade_price = self.price_array[self.time - 1]
        volatility = self.volatility_array[self.time - 1]
        volume = self.volume_array[self.time - 1]

        begin_total_asset = self.long_equity + self.short_equity

        trade_shares = self._calculate_trade_shares(
            actions, trade_price, volume, begin_total_asset
        )

        # Turnover percentiles for logging
        tp_idx = min(self.time, self.max_step)
        turnover_percentiles = self._calculate_turnover_percentiles(
            self.price_array[tp_idx], self.volume_array[tp_idx]
        )

        total_traded_value = 0.0
        total_trade_cost = 0.0
        total_buy_value = 0.0
        total_sell_value = 0.0
        trades = []

        # Process sells first, then buys (frees cash/limit for buying)
        order = np.argsort(trade_shares)
        for i_idx in order:
            i = int(i_idx)
            if trade_shares[i] == 0:
                continue
            traded_value, trade_cost, trade_sides = self._execute_trade(
                i, trade_shares[i], trade_price, volatility, volume
            )
            if traded_value > 0:
                pov = abs(trade_shares[i]) / volume[i] if volume[i] > 0 else 0.0
                for side_info in trade_sides:
                    trades.append(
                        {
                            "stock_idx": i,
                            "side": side_info["side"],
                            "shares": side_info["shares"],
                            "notional": side_info["shares"] * trade_price[i],
                            "pov": pov,
                            "turnover_percentile": turnover_percentiles[i],
                        }
                    )
                if trade_shares[i] > 0:
                    total_buy_value += traded_value
                else:
                    total_sell_value += traded_value
                total_traded_value += traded_value
                total_trade_cost += trade_cost

        # ── Margin handling at trade_price (before day advance) ───────
        # Matches original: margin checks use same-day prices.

        # Post-action maintenance (strict level) OR periodic adjustment
        if self.time % self.margin_adjust_period == 0:
            self._margin_adjust_long(trade_price, volatility, volume)
            self._margin_adjust_short(trade_price, volatility, volume)
        else:
            if self._check_long_maintenance(trade_price) < self.maintenance_margin:
                self._margin_adjust_long(trade_price, volatility, volume)
            if self._check_short_maintenance(trade_price) < self.maintenance_margin:
                self._margin_adjust_short(trade_price, volatility, volume)

        # ── Day advance ───────────────────────────────────────────────
        self.impact_model.end_day(self.date_list[self.time])

        new_price = self.price_array[min(self.time, self.max_step)]
        self._update_equities(new_price)

        current_total_equity = self.long_equity + self.short_equity

        # ── Paper reward: R_t = λ1 * profit + λ2 * risk ──────────────
        profit = current_total_equity - begin_total_asset
        self._equity_history.append(current_total_equity)
        risk = self._rolling_sharpe()
        reward = self.lambda_1 * profit + self.lambda_2 * risk

        self.total_asset = current_total_equity
        self.cash = max(self.long_cash - self.loan, 0.0)

        if done:
            self.episode_return = (
                self.total_asset / self.initial_total_asset
                if self.initial_total_asset > 0
                else 0.0
            )

        turnover = total_traded_value / self.total_asset if self.total_asset > 0 else 0
        info = {
            "turnover": turnover,
            "cost": total_trade_cost,
            "total_buy_value": total_buy_value,
            "total_sell_value": total_sell_value,
            "cash": self.cash,
            "trades": trades,
        }
        return self.get_state(new_price), reward, done, False, info

    # ── Trade execution ───────────────────────────────────────────────

    def _execute_trade(
        self,
        i: int,
        trade_size: int,
        price: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
    ) -> Tuple[float, float, list]:
        """Execute a trade with automatic position flipping.

        Replaces the paper's Position Fusion Module (Section 4.3.1):
        positive trade_size first covers short then buys long;
        negative trade_size first sells long then opens short.
        """
        if trade_size == 0:
            return 0.0, 0.0, []

        total_value = 0.0
        total_cost = 0.0
        trade_sides = []
        current_holding = int(self.stocks[i])

        if trade_size > 0:  # Buy direction: cover short, then buy long
            if current_holding < 0:
                shares_to_cover = min(trade_size, -current_holding)
                val, cost = self._cover_short(
                    i, shares_to_cover, price, volatility, volume
                )
                total_value += val
                total_cost += cost
                if val > 0:
                    trade_sides.append({"side": "cover", "shares": shares_to_cover})
                trade_size -= shares_to_cover
            if trade_size > 0:
                val, cost = self._buy_long(i, trade_size, price, volatility, volume)
                total_value += val
                total_cost += cost
                if val > 0:
                    trade_sides.append({"side": "buy", "shares": trade_size})
        else:  # Sell direction: sell long, then open short
            abs_trade = abs(trade_size)
            if current_holding > 0:
                shares_to_sell = min(abs_trade, current_holding)
                val, cost = self._sell_long(
                    i, shares_to_sell, price, volatility, volume
                )
                total_value += val
                total_cost += cost
                if val > 0:
                    trade_sides.append({"side": "sell", "shares": shares_to_sell})
                abs_trade -= shares_to_sell
            if abs_trade > 0:
                val, cost = self._sell_short(i, abs_trade, price, volatility, volume)
                total_value += val
                total_cost += cost
                if val > 0:
                    trade_sides.append({"side": "short", "shares": abs_trade})

        if total_value > 0:
            self.stocks_cool_down[i] = 0
        return total_value, total_cost, trade_sides

    # ── Long position trades ──────────────────────────────────────────

    def _buy_long(
        self,
        i: int,
        shares: int,
        price: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
    ) -> Tuple[float, float]:
        """Buy long: cash decreases, holdings increase.

        Pre-action constraint (Section 4.3.3): blocked when long maintenance
        ratio falls below the warning level (40%).
        """
        if self._check_long_maintenance(price) <= self.maintenance_warning:
            return 0.0, 0.0
        if price[i] <= 0 or shares <= 0:
            return 0.0, 0.0

        max_affordable = int(self.long_cash / price[i])
        shares = min(shares, max_affordable)
        if shares <= 0:
            return 0.0, 0.0

        impact = self.impact_model.apply_trade(
            shares, price[i], volatility[i], volume[i], self.stock_symbols[i]
        )
        total_cost = shares * price[i] + impact.cost
        if total_cost > self.long_cash:
            return 0.0, 0.0

        self.long_cash -= total_cost
        self.long_equity -= impact.cost
        self.stocks[i] += shares
        return shares * price[i], impact.cost

    def _sell_long(
        self,
        i: int,
        shares: int,
        price: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
    ) -> Tuple[float, float]:
        """Sell long: cash increases, holdings decrease."""
        shares = min(shares, max(int(self.stocks[i]), 0))
        if shares <= 0 or price[i] <= 0:
            return 0.0, 0.0

        impact = self.impact_model.apply_trade(
            -shares, price[i], volatility[i], volume[i], self.stock_symbols[i]
        )
        proceeds = shares * price[i] - impact.cost
        if proceeds <= 0:
            return 0.0, 0.0

        self.long_cash += proceeds
        self.long_equity -= impact.cost
        self.stocks[i] -= shares
        return shares * price[i], impact.cost

    # ── Short position trades ─────────────────────────────────────────

    def _sell_short(
        self,
        i: int,
        shares: int,
        price: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
    ) -> Tuple[float, float]:
        """Open/increase short: limit decreases, holdings go more negative.

        Pre-action constraint (Section 4.3.3): blocked when short maintenance
        ratio falls below the warning level (40%).
        """
        if self._check_short_maintenance(price) <= self.maintenance_warning:
            return 0.0, 0.0
        if price[i] <= 0 or shares <= 0:
            return 0.0, 0.0

        market_value = shares * price[i]
        if market_value > self.short_limit:
            shares = int(self.short_limit / price[i])
            if shares <= 0:
                return 0.0, 0.0
            market_value = shares * price[i]

        impact = self.impact_model.apply_trade(
            -shares, price[i], volatility[i], volume[i], self.stock_symbols[i]
        )

        self.short_limit -= market_value
        self.short_credit -= impact.cost
        self.short_equity -= impact.cost
        self.stocks[i] -= shares
        return market_value, impact.cost

    def _cover_short(
        self,
        i: int,
        shares: int,
        price: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
    ) -> Tuple[float, float]:
        """Close/reduce short: limit increases, holdings go less negative."""
        shares = min(shares, max(int(abs(self.stocks[i])), 0))
        if shares <= 0 or price[i] <= 0:
            return 0.0, 0.0

        impact = self.impact_model.apply_trade(
            shares, price[i], volatility[i], volume[i], self.stock_symbols[i]
        )

        market_value = shares * price[i]
        self.short_limit += market_value
        self.short_credit -= impact.cost
        self.short_equity -= impact.cost
        self.stocks[i] += shares
        return market_value, impact.cost

    # ── Equity accounting (paper Section 4.2.1) ──────────────────────

    def _update_equities(self, price: np.ndarray) -> None:
        """Full equity recalculation from current prices.

        e^l = b^l + m^l - l^l   (cash + long_market - loan)
        e^s = b^s - l^s - m^s   (credit - limit - short_market)
        """
        adjusted = price + self._get_perm_impact()

        long_mask = self.stocks > 0
        long_mv = (
            (self.stocks[long_mask] * adjusted[long_mask]).sum()
            if long_mask.any()
            else 0.0
        )
        self.long_equity = self.long_cash + long_mv - self.loan

        short_mask = self.stocks < 0
        short_mv = (
            abs((self.stocks[short_mask] * adjusted[short_mask]).sum())
            if short_mask.any()
            else 0.0
        )
        self.short_equity = self.short_credit - self.short_limit - short_mv

    def _calculate_total_equity(self, price: np.ndarray) -> float:
        self._update_equities(price)
        return self.long_equity + self.short_equity

    # ── Margin Adjustment Module — Long (paper Section 4.3.2) ─────────

    def _margin_adjust_long(
        self,
        price: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
    ) -> None:
        """Periodic buying-power re-alignment for the long position.

        Compares loan against target (k-1)*equity.
        Scenario 1 — profit: borrow more (increase cash and loan).
        Scenario 2 — loss:  return cash, or liquidate from smallest first.
        """
        self._update_equities(price)
        loan_diff = self.long_equity - self.loan

        if loan_diff > 0:
            # Profit: can borrow more cash
            self.loan = self.long_equity
            self.long_cash += loan_diff
        elif loan_diff < 0:
            # Loss: must reduce loan
            shortfall = abs(loan_diff)
            if self.long_cash >= shortfall:
                self.long_cash -= shortfall
                self.loan -= shortfall
            else:
                remaining = shortfall - self.long_cash
                self.long_cash = 0

                adjusted = price + self._get_perm_impact()
                long_indices = np.where(self.stocks > 0)[0]
                if len(long_indices) > 0:
                    values = self.stocks[long_indices] * adjusted[long_indices]
                    for idx in long_indices[np.argsort(values)]:
                        n_sell = int(self.stocks[idx])
                        if n_sell <= 0:
                            continue
                        self._sell_long(idx, n_sell, price, volatility, volume)
                        if self.long_cash >= remaining:
                            self.long_cash -= remaining
                            break
                        else:
                            remaining -= self.long_cash
                            self.long_cash = 0

                self.loan -= shortfall

        self._update_equities(price)

    # ── Margin Adjustment Module — Short (paper Section 4.3.2) ────────

    def _margin_adjust_short(
        self,
        price: np.ndarray,
        volatility: np.ndarray,
        volume: np.ndarray,
    ) -> None:
        """Periodic credit/limit re-alignment for the short position.

        Compares current borrow usage (limit + short_market) against
        target k * equity.
        Scenario 1 — profit: increase limit and credit.
        Scenario 2 — loss:  reduce limit/credit, or cover from smallest first.
        """
        self._update_equities(price)
        adjusted = price + self._get_perm_impact()

        short_mask = self.stocks < 0
        short_mv = (
            abs((self.stocks[short_mask] * adjusted[short_mask]).sum())
            if short_mask.any()
            else 0.0
        )
        borrow_used = self.short_limit + short_mv
        max_borrow = self.margin_rate * self.short_equity
        borrow_diff = max_borrow - borrow_used

        if borrow_diff > 0:
            # Profit: can short more
            self.short_limit += borrow_diff
            self.short_credit += borrow_diff
        elif borrow_diff < 0:
            # Loss: must reduce borrowing
            shortfall = abs(borrow_diff)
            if self.short_limit >= shortfall:
                self.short_limit -= shortfall
                self.short_credit -= shortfall
            else:
                remaining = shortfall - self.short_limit
                self.short_limit = 0

                short_indices = np.where(self.stocks < 0)[0]
                if len(short_indices) > 0:
                    values = abs(self.stocks[short_indices] * adjusted[short_indices])
                    for idx in short_indices[np.argsort(values)]:
                        n_cover = int(abs(self.stocks[idx]))
                        if n_cover <= 0:
                            continue
                        self._cover_short(idx, n_cover, price, volatility, volume)
                        if self.short_limit >= remaining:
                            self.short_limit -= remaining
                            break
                        else:
                            remaining -= self.short_limit
                            self.short_limit = 0

                self.short_credit -= shortfall

        self._update_equities(price)

    # ── State construction (paper Section 4.2.1) ──────────────────────

    def get_state(self, price: np.ndarray) -> np.ndarray:
        """Build observation vector, scaled by 2**-12.

        S_t = [cash, loan, long_equity, limit, credit, short_equity,
               prices(N), holdings(N), perm_impact(N), cooldown(N), tech(T)]
        """
        scale = 2**-12
        perm_impact = self._get_perm_impact()
        state_components = [
            np.array(
                [
                    self.long_cash,
                    self.loan,
                    self.long_equity,
                    self.short_limit,
                    self.short_credit,
                    self.short_equity,
                ]
            )
            * scale,
            price * scale,
            self.stocks * scale,
            perm_impact * scale,
            self.stocks_cool_down,
            self.tech_array[min(self.time, self.max_step)],
        ]
        return np.hstack(state_components).astype(np.float32)

    def render(self, mode="human"):
        return self.total_asset

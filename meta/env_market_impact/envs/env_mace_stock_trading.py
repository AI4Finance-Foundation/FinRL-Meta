from dataclasses import dataclass
from typing import Dict
from typing import Optional
from typing import Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3.common.running_mean_std import RunningMeanStd

from .impact_models import ImpactModel
from .impact_models import SqrtImpactModel

EPS = 1e-8


@dataclass
class EnvParams:
    """Configuration parameters for :class:`MACEStockTradingEnv`.

    Collects every tuneable knob that controls the environment's behaviour
    into a single container.  Pass an instance to the environment's
    constructor alongside the per-run values (``initial_capital``,
    ``initial_stocks``) that are supplied as direct keyword arguments.

    Attributes
    ----------
    max_stock_pct : float
        Maximum fraction of total portfolio value that may be held in a
        single stock position.
    max_trade_volume_pct : float
        Maximum fraction of a stock's daily volume that may be traded in
        a single step.
    reward_scaling : float
        Multiplicative scaling factor applied to the reward signal before
        it is returned to the agent.
    include_permanent_impact_in_state : bool
        Whether to append the cumulative permanent market-impact vector
        (one value per stock, in basis points) to the observation.
    include_cooldown_in_state : bool
        Whether to append the per-stock cooldown counter (number of steps
        since the stock was last traded, clipped and normalised to [0, 1])
        to the observation.
    include_tbill_in_state : bool
        Whether to append the current risk-free (T-bill) daily rate to
        the observation.
    sharpe_window : int
        Rolling window length (in steps) used when computing the
        exponential-moving-average Sharpe ratio for the reward.
    horizon : int
        Effective horizon for the Differential Sharpe Ratio (DSR) reward.
        Controls the EMA decay via ``alpha = 1 / horizon``.
    eta_dd : float
        Penalty weight applied to increases in drawdown.  Larger values
        make the agent more drawdown-averse.
    use_obs_normalizer : bool
        When *True*, observations are standardised online using a
        running-mean / running-variance estimator
        (``stable_baselines3.common.running_mean_std.RunningMeanStd``)
        and clipped to ``[-obs_clip, obs_clip]``.  When *False*, raw
        features are returned.
    obs_clip : float
        Absolute clipping bound applied after normalisation (ignored when
        ``use_obs_normalizer`` is *False*).
    obs_norm_update : bool
        Whether the running-mean/variance normaliser updates its
        statistics on each observation.  Set to *False* to freeze
        statistics (e.g. during evaluation).
    """

    max_stock_pct: float = 0.02
    max_trade_volume_pct: float = 0.1
    reward_scaling: float = 2**-11
    include_permanent_impact_in_state: bool = True
    include_cooldown_in_state: bool = True
    include_tbill_in_state: bool = True
    sharpe_window: int = 20
    horizon: int = 20
    eta_dd: float = 0.5
    use_obs_normalizer: bool = True
    obs_clip: float = 10.0
    obs_norm_update: bool = True


class MACEStockTradingEnv(gym.Env):
    """
    MACE (Market-Adjusted Cost Execution) is a stock trading environment that incorporates a market impact model.
    This environment simulates single-stock trading, where the agent's actions
    (buying or selling shares) incur costs based on a realistic market
    impact model, affecting the portfolio's value.
    Attributes:
        observation_space (gym.spaces.Box): The observation space.
        action_space (gym.spaces.Box): The action space.
        stock_dim (int): The number of stocks in the environment.
    """

    def __init__(
        self,
        config: Dict,
        params: Optional[EnvParams] = None,
        impact_model: Optional[ImpactModel] = None,
        initial_capital: float = 1e6,
        initial_stocks: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initializes the StockTradingEnvImpact.
        Args:
            config: A dictionary containing the market data arrays.
                Expected keys: "price_array", "tech_array",
                "volatility_array", "volume_array".
            params: An :class:`EnvParams` dataclass with all tuneable
                parameters.  If *None*, default values are used.
            impact_model: An instance of an ImpactModel subclass. If None, a
                default SqrtImpactModel will be created.
            initial_capital: The starting cash balance for the portfolio.
            initial_stocks: Initial share holdings per stock.  If *None*,
                the portfolio starts fully in cash (all zeros).
        """
        if params is None:
            params = EnvParams()
        self.params = params

        self.date_list = config["date_list"]
        price_array = config["price_array"]
        tech_array = config["tech_array"]
        self.volatility_array = config["volatility_array"]
        self.volume_array = config["volume_array"]
        self.adv20_array = config["adv20_array"]
        self.tbill_rates = config["tbill_rates"]

        self.price_array = price_array.astype(np.float32)
        self.tech_array = tech_array.astype(np.float32) * (2**-7)

        self.stock_dim = self.price_array.shape[1]
        self.max_stock_pct = params.max_stock_pct
        self.max_trade_volume_pct = params.max_trade_volume_pct
        self.reward_scaling = params.reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(self.stock_dim, dtype=int)
            if initial_stocks is None
            else initial_stocks
        )
        self.sharpe_window = params.sharpe_window
        self.alpha = 1.0 / params.horizon
        self.eta_dd = params.eta_dd

        self.impact_model = (
            impact_model if impact_model is not None else SqrtImpactModel()
        )
        self.stock_symbols = config["tic_list"]
        self.include_permanent_impact_in_state = (
            params.include_permanent_impact_in_state
        )
        self.include_cooldown_in_state = params.include_cooldown_in_state
        self.include_tbill_in_state = params.include_tbill_in_state

        # State: [cash, price, stocks, tech, adv20]
        # Optional features are appended to the state.
        self.state_dim = (
            1 + (2 * self.stock_dim) + self.tech_array.shape[1] + self.stock_dim
        )
        if self.include_permanent_impact_in_state:
            self.state_dim += self.stock_dim
        if self.include_cooldown_in_state:
            self.state_dim += self.stock_dim
        if self.include_tbill_in_state:
            self.state_dim += 1

        self.action_dim = self.stock_dim
        self._obs_norm_update = params.obs_norm_update
        self.obs_clip = params.obs_clip
        if params.use_obs_normalizer:
            low, high = -params.obs_clip, params.obs_clip
            self.obs_normalizer = RunningMeanStd(shape=(self.state_dim,))
        else:
            low, high = -np.inf, np.inf
            self.obs_normalizer = None

        self.observation_space = gym.spaces.Box(
            low=low, high=high, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )
        self.max_step = self.price_array.shape[0] - 1

    def _get_perm_impact(self) -> np.ndarray:
        """Get the current permanent impact from the impact model."""
        return self.impact_model.get_perm_state_array(self.stock_symbols)

    def get_normalizer_state(self) -> Optional[Dict]:
        """
        Get the current state of the observation normalizer.

        Returns
        -------
        dict or None
            Dictionary containing 'mean', 'var', and 'count' if normalizer exists,
            None otherwise.
        """
        if self.obs_normalizer is None:
            return None
        return {
            "mean": self.obs_normalizer.mean.copy(),
            "var": self.obs_normalizer.var.copy(),
            "count": self.obs_normalizer.count,
        }

    def set_normalizer_state(self, state: Dict, freeze: bool = True) -> None:
        """
        Set the observation normalizer state from a saved state.

        Use this to transfer normalizer statistics from a training environment
        to a test environment, ensuring consistent observation scaling.

        Parameters
        ----------
        state : dict
            Dictionary containing 'mean', 'var', and 'count' from get_normalizer_state().
        freeze : bool (default=True)
            If True, disables normalizer updates so statistics remain fixed during testing.
        """
        if self.obs_normalizer is None:
            return
        self.obs_normalizer.mean = state["mean"].copy()
        self.obs_normalizer.var = state["var"].copy()
        self.obs_normalizer.count = state["count"]
        if freeze:
            self._obs_norm_update = False

    def reset(self, *, seed=None, options=None):
        """Resets the environment to its initial state."""
        super().reset(seed=seed)

        self.time = 0
        self.stocks = self.initial_stocks.copy()
        self.cash = self.initial_capital
        self.stocks_cool_down = np.zeros_like(self.stocks)
        self.mu_prev = 0.0
        self.m2_prev = 1e-6
        self.dd = 0.0
        self.episode_return = 0.0
        if options is None or options.get("reset_impact_model", True):
            self.impact_model.reset()

        price = self.price_array[self.time]
        adjusted_prices = price + self._get_perm_impact()
        self.total_asset = self._calculate_total_asset(adjusted_prices)
        self.peak = self.total_asset
        return self.get_state(adjusted_prices, adjusted_prices, self.total_asset), {}

    def _calculate_turnover_percentiles(
        self, prices: np.ndarray, volumes: np.ndarray
    ) -> np.ndarray:
        """
        Calculate the percentile rank of each stock's market turnover (gross notional).

        Returns an array where percentile[i] is the percentile rank (0-100) of stock i's
        daily gross notional among all stocks in the universe.
        """
        gross_notional = prices * volumes
        # Use scipy-style percentile ranking: (rank - 1) / (n - 1) * 100
        n = len(gross_notional)
        if n <= 1:
            return np.full(n, 50.0)

        # Get ranks (1-based, with average for ties)
        sorted_indices = np.argsort(gross_notional)
        ranks = np.empty_like(sorted_indices, dtype=float)
        ranks[sorted_indices] = np.arange(1, n + 1)

        # Convert to percentile (0-100 scale)
        percentiles = (ranks - 1) / (n - 1) * 100
        return percentiles

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Run one timestep of the environment's dynamics.
        Args:
            actions: An action provided by the agent.
        Returns:
            A tuple containing the new state, reward, terminated flag, truncated flag, and info dict.
        """
        prev_adjusted_prices = self.price_array[self.time] + self._get_perm_impact()

        self.time += 1
        self.stocks_cool_down += 1
        terminated = self.time >= self.max_step

        adjusted_prices = self.price_array[self.time] + self._get_perm_impact()
        volatility = self.volatility_array[self.time]
        volume = self.volume_array[self.time]
        trade_shares = self._calculate_trade_shares(actions, adjusted_prices, volume)

        # Calculate turnover percentiles for all stocks
        turnover_percentiles = self._calculate_turnover_percentiles(
            self.price_array[self.time], self.volume_array[self.time]
        )

        total_traded_value = 0.0
        total_trade_cost = 0.0
        total_buy_value = 0.0
        total_sell_value = 0.0
        trades = []

        # Process sells first to free up cash
        for i in range(len(trade_shares)):
            if trade_shares[i] < 0:
                sell_shares = -trade_shares[i]
                traded_value, trade_cost = self._sell_stock(
                    i, sell_shares, adjusted_prices[i], volatility[i], volume[i]
                )
                if traded_value > 0:
                    gross_notional = adjusted_prices[i] * sell_shares
                    pov = sell_shares / volume[i] if volume[i] > 0 else 0.0
                    trades.append(
                        {
                            "stock_idx": i,
                            "side": "sell",
                            "shares": sell_shares,
                            "notional": gross_notional,
                            "pov": pov,
                            "turnover_percentile": turnover_percentiles[i],
                        }
                    )
                total_sell_value += traded_value
                total_traded_value += traded_value
                total_trade_cost += trade_cost

        for i in range(len(trade_shares)):
            if trade_shares[i] > 0:
                buy_shares = trade_shares[i]
                traded_value, trade_cost = self._buy_stock(
                    i, buy_shares, adjusted_prices[i], volatility[i], volume[i]
                )
                if traded_value > 0:
                    gross_notional = adjusted_prices[i] * buy_shares
                    pov = buy_shares / volume[i] if volume[i] > 0 else 0.0
                    trades.append(
                        {
                            "stock_idx": i,
                            "side": "buy",
                            "shares": buy_shares,
                            "notional": gross_notional,
                            "pov": pov,
                            "turnover_percentile": turnover_percentiles[i],
                        }
                    )
                total_buy_value += traded_value
                total_traded_value += traded_value
                total_trade_cost += trade_cost

        # Accrue interest on cash
        self.cash += self.cash * self._calc_rf_rate()
        self.impact_model.end_day(self.date_list[self.time])
        adjusted_prices_post = self.price_array[self.time] + self._get_perm_impact()
        end_total_asset = self._calculate_total_asset(adjusted_prices_post)
        self.total_asset = end_total_asset
        if terminated:
            state = np.zeros(self.state_dim, dtype=np.float32)
            self.episode_return = self.total_asset / self.initial_capital
        else:
            state = self.get_state(
                adjusted_prices_post, prev_adjusted_prices, end_total_asset
            )
            # Calculate reward based on next day's total asset
            tomorrow_adjusted_prices = (
                self.price_array[self.time + 1] + self._get_perm_impact()
            )
            end_total_asset = self._calculate_total_asset(tomorrow_adjusted_prices)

        reward = self._calculate_reward(end_total_asset)
        turnover = total_traded_value / self.total_asset if self.total_asset > 0 else 0
        info = {
            "turnover": turnover,
            "cost": total_trade_cost,
            "total_buy_value": total_buy_value,
            "total_sell_value": total_sell_value,
            "cash": self.cash,
            "trades": trades,
        }
        return state, reward, terminated, False, info

    def _calculate_trade_shares(
        self, actions: np.ndarray, adjusted_prices: np.ndarray, volume: np.ndarray
    ) -> np.ndarray:
        # Determine the desired trade size while respecting max exposure constraints
        max_stocks_per_position = self._calculate_max_stock_per_position(
            adjusted_prices
        )
        desired_shares = (actions * max_stocks_per_position).astype(int)
        trade_shares = np.zeros_like(desired_shares)
        for i in range(len(desired_shares)):
            limit = max_stocks_per_position[i]
            current = self.stocks[i]
            if desired_shares[i] < 0:
                # Cap the sell amount to how many shares we have
                trade_shares[i] = -min(-desired_shares[i], current)
            elif current > limit:
                # Force a sell if current exposure already exceeds the limit
                trade_shares[i] = limit - current
            else:
                # Cap the buy amount so we don't breach the limit
                trade_shares[i] = min(desired_shares[i], limit - current)

        volume_limit = (volume * self.max_trade_volume_pct).astype(int)
        trade_shares = np.clip(trade_shares, -volume_limit, volume_limit)
        return trade_shares

    def _calc_rf_rate(self) -> float:
        """Calculates the risk-free rate."""
        return (1 + self.tbill_rates[self.time] / 100.0) ** (1.0 / 252.0) - 1.0

    def _differential_sharpe(self, r_t: float):
        """
        One-step update of the Differential Sharpe Ratio (DSR).

        Parameters
        ----------
        r_t : float
            Return at time t (already net of transaction costs, risk-free rate, etc.).

        Returns
        -------
        dsr_t : float
            Reward signal at t.

        Notes
        -----
        Let α = 1 / horizon.  Update the first and second moments via

            μ_t   = (1-α) μ_{t-1} + α r_t
            m²_t  = (1-α) m²_{t-1} + α r_t²

        Then

            σ_t²  = max(m²_t – μ_t², eps)
            SR_{t-1} = μ_{t-1} / σ_{t-1}
            x     = (r_t – μ_{t-1}) / σ_t
            DSR_t = x – 0.5 · SR_{t-1} · x²

        The DSR is the stochastic gradient of the Sharpe ratio, giving a dense,
        risk-adjusted reward suitable for online RL optimisation.

        Reference
        ---------
        J. Moody & M. Saffell, “Reinforcement Learning for Trading,” *Advances in
        Neural Information Processing Systems 11* (NeurIPS 1998).
        """
        # update moments
        mu_next = (1 - self.alpha) * self.mu_prev + self.alpha * r_t
        m2_next = (1 - self.alpha) * self.m2_prev + self.alpha * (r_t**2)

        var_next = max(m2_next - mu_next**2, EPS)
        sigma_next = np.sqrt(var_next)

        var_prev = max(self.m2_prev - self.mu_prev**2, EPS)
        sigma_prev = np.sqrt(var_prev)
        sr_prev = self.mu_prev / (sigma_prev + EPS)

        x = (r_t - self.mu_prev) / sigma_next
        dsr_t = x - 0.5 * sr_prev * x**2

        self.mu_prev = mu_next
        self.m2_prev = m2_next

        return dsr_t

    def _calculate_reward(self, end_total_asset: float) -> float:
        """Calculates the reward for the current step."""
        r_t = (end_total_asset / self.total_asset) - 1 if self.total_asset != 0 else 0.0
        dsr_reward = self._differential_sharpe(r_t)
        self.peak = max(self.peak, end_total_asset)
        dd_new = (self.peak - end_total_asset) / self.peak
        delta_dd = max(0, dd_new - self.dd)
        self.dd = dd_new
        return (dsr_reward - self.eta_dd * delta_dd**2) * self.reward_scaling

    def _sell_stock(
        self,
        index: int,
        sell_shares: int,
        price: float,
        volatility: float,
        volume: float,
    ) -> Tuple[float, float]:
        """Executes a sell trade and returns the traded value and cost."""
        if price > 0 and sell_shares > 0:
            impact_result = self.impact_model.apply_trade(
                -sell_shares,
                price,
                volatility,
                volume,
                self.stock_symbols[index],
            )
            self.stocks[index] -= sell_shares
            total_value = price * sell_shares - impact_result.cost
            self.cash += total_value
            self.stocks_cool_down[index] = 0
            return total_value, impact_result.cost
        return 0.0, 0.0

    def _buy_stock(
        self,
        index: int,
        buy_shares: int,
        price: float,
        volatility: float,
        volume: float,
    ) -> Tuple[float, float]:
        """Executes a buy trade and returns the traded value and cost."""
        if price > 0 and buy_shares > 0:
            impact_result = self.impact_model.apply_trade(
                buy_shares,
                price,
                volatility,
                volume,
                self.stock_symbols[index],
            )
            total_cost = price * buy_shares + impact_result.cost
            if total_cost <= self.cash:
                self.stocks[index] += buy_shares
                self.cash -= total_cost
                self.stocks_cool_down[index] = 0
                return total_cost, impact_result.cost
        return 0.0, 0.0

    def _calculate_total_asset(self, adjusted_prices: np.ndarray) -> float:
        """Calculates the total portfolio value."""
        r = self.cash + (self.stocks * adjusted_prices).sum()
        return r if r > 0 else EPS

    def _calculate_max_stock_per_position(
        self, current_prices: np.ndarray
    ) -> np.ndarray:
        """Calculates the max number of shares per stock based on portfolio percentage."""
        portfolio_value = self.cash + (self.stocks * current_prices).sum()
        max_position_value = portfolio_value * self.max_stock_pct
        return np.where(
            current_prices > 0, max_position_value / current_prices, 0
        ).astype(int)

    def get_state(
        self,
        adjusted_prices: np.ndarray,
        prev_adjusted_prices: np.ndarray,
        end_total_asset: float,
    ) -> np.ndarray:
        """Build and return the current observation vector.

        Computes internally all features that depend on the current env state and prices,
        given the current and previous adjusted prices.
        """
        # Price feature: 1-day log return
        price_ret_1d = np.log((adjusted_prices + EPS) / (prev_adjusted_prices + EPS))
        position_value_pct = (self.stocks * adjusted_prices) / end_total_asset
        shares_over_adv = self.stocks / (self.adv20_array[self.time] + EPS)
        perm_impact = self._get_perm_impact()
        impact_bps = perm_impact / (adjusted_prices + EPS) * 1e4
        cash_pct = self.cash / end_total_asset

        state_components = [
            np.array([cash_pct], dtype=np.float32),
            price_ret_1d,
            position_value_pct,
            self.tech_array[self.time],
            shares_over_adv,
        ]
        if self.include_permanent_impact_in_state:
            state_components.append(impact_bps)
        if self.include_cooldown_in_state:
            state_components.append(
                (np.clip(self.stocks_cool_down, 0, 10.0) / 10.0).astype(np.float32)
            )
        if self.include_tbill_in_state:
            state_components.append(np.array([self._calc_rf_rate()], dtype=np.float32))

        obs = np.hstack(state_components).astype(np.float32)
        if self.obs_normalizer is not None:
            if self._obs_norm_update:
                self.obs_normalizer.update(obs[None, :])

            mean = self.obs_normalizer.mean.astype(np.float32)
            var = self.obs_normalizer.var.astype(np.float32)
            obs = (obs - mean) / np.sqrt(var + 1e-8)
            obs = np.clip(obs, -self.obs_clip, self.obs_clip)

        return obs

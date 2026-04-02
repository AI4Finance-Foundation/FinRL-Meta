from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd


@dataclass
class TradeImpact:
    """
    Encapsulates the result of an impact model's trade.

    Attributes:
    -----------
    cost : float
        Total notional cost for executing the trade (impact + commission).
    price_shift : float
        Permanent price shift ΔP to apply to the mid-price for subsequent trades.
    """

    cost: float
    price_shift: float


class ImpactModel(ABC):

    def __init__(self, perm_half_life_days: float = 5.0):
        """
        Initialize the impact model.

        Parameters
        ----------
        perm_half_life_days : float (default=5.0)
            Half-life for permanent impact decay in trading days. The permanent
            impact decays exponentially as the market absorbs the information.
            Empirical research suggests 1-5 days for large-cap stocks (default 5.0).
            Use longer values (5-20 days) for less liquid securities.

        References
        ----------
        Bouchaud, J.P., Farmer, J.D., & Lillo, F. (2009). How markets slowly digest
            changes in supply and demand. In Handbook of Financial Markets.
        """
        self.perm_half_life_days = perm_half_life_days
        # Compute decay rate: (1 - decay_rate)^half_life = 0.5
        if perm_half_life_days > 0:
            self.perm_decay_rate = 1 - 0.5 ** (1.0 / perm_half_life_days)
        else:
            self.perm_decay_rate = 0.0

        self.reset()

    @abstractmethod
    def apply_trade(
        self,
        trade_size: float,
        price: float,
        volatility: float,
        volume: float,
        symbol: str,
    ) -> TradeImpact:
        """
        Execute a trade and return its impact.

        Parameters
        ----------
        trade_size : float
            Signed number of shares to trade.
        price : float
            Execution price per share (e.g., VWAP).
        volatility : float
            Asset's daily return volatility.
        volume : float
            Available daily volume in shares.
        symbol : str
            Asset identifier to track per-symbol state.

        Returns
        -------
        TradeImpact
            An object containing the calculated cost and permanent price shift.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def end_day(self, date_str: str) -> None:
        """
        Apply decay and record the permanent impact state at the end of a trading day.

        Parameters
        ----------
        date_str : str
            The date of the trading day being ended.

        Notes
        -----
        This method first applies exponential decay to the permanent impact state
        (simulating market absorption of information), then records the state for
        each symbol into the impact history dataframe for later analysis.

        The decay follows: impact_new = impact_old * (1 - decay_rate)
        where decay_rate is computed from perm_half_life_days.
        """
        # Apply exponential decay to permanent impact (market absorbs information)
        if self.perm_decay_rate > 0:
            decay_factor = 1 - self.perm_decay_rate
            for symbol in self._perm_state:
                self._perm_state[symbol] *= decay_factor

        for symbol, impact in self._perm_state.items():
            self._impact_records.append(
                {"date": date_str, "symbol": symbol, "permanent_impact": impact}
            )

    def get_impact_history(self) -> pd.DataFrame:
        """
        Get the recorded impact history as a DataFrame.

        Returns
        -------
        pd.DataFrame
            The impact history with columns ['date', 'symbol', 'permanent_impact'].
        """
        if not self._impact_records:
            return pd.DataFrame(columns=["date", "symbol", "permanent_impact"])
        return pd.DataFrame(self._impact_records)

    def reset(self) -> None:
        """
        Reset the impact model state.
        """
        # Track cumulative permanent price shift per symbol
        self._perm_state = defaultdict(float)
        self._impact_records = []

    def get_perm_state(self, symbol: str) -> float:
        """Get the permanent impact state for a given symbol."""
        return self._perm_state.get(symbol, 0.0)

    def get_perm_state_array(self, symbols: list) -> np.ndarray:
        """Get the permanent impact state as an array for a list of symbols."""
        return np.array(
            [self._perm_state.get(s, 0.0) for s in symbols], dtype=np.float32
        )


class SqrtImpactModel(ImpactModel):
    """
    Market impact model based on the empirically-supported square-root model of price impact.

    This model is a modern alternative to the original Almgren-Chriss model. The
    classic AC model assumes the permanent price impact is a linear function of the
    trading rate (leading to a quadratic cost), while temporary impact is linear.
    Instead, this implementation follows the "square-root law" of market impact,
    which has been extensively validated in empirical studies of market data.

    The total impact is proportional to the square root of the trade size, scaled by
    the asset's volatility. The cost is decomposed into a permanent component and a
    temporary (or "immediate") component.

    The total impact as a fraction of the price is given by:

        I(x) = Y * σ * sqrt(|x| / V)

    where:
    - x: trade size (number of shares)
    - V: total daily volume
    - σ: daily price volatility
    - Y: an empirical coefficient, typically around 0.6 for single stocks

    The permanent impact is a fraction of the total impact:

        ΔP = perm_fraction * I(x)

    The cost of the trade is calculated from the permanent and temporary components.

    Parameters
    ----------
    Y : float (default=0.6)
        Empirical square-root impact coefficient, typically 0.5–1.0.
    perm_fraction : float (default=0.25)
        Fraction of total impact assumed permanent, ~25% empirically.

    References
    ----------
    Gatheral, J. (2010). No-Dynamic-Arbitrage and Market Impact.
        *Quantitative Finance*, 10(7), 749-759.
    Tóth, B., Lempérière, Y., Deremble, C., de Lataillade, J., Kockelkoren, J., & Bouchaud, J. P. (2011).
        Anomalous Price Impact and the Critical Nature of Liquidity in Financial Markets.
        *Physical Review X*, 1(2), 021006.
    Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
        *Journal of Risk*, 3(2), 5-39. (For historical context on the original model)
    """

    def __init__(
        self,
        Y: float = 0.6,
        perm_fraction: float = 0.25,
        perm_half_life_days: float = 5.0,
    ) -> None:
        super().__init__(perm_half_life_days=perm_half_life_days)
        self.Y = Y
        self.perm_fraction = perm_fraction
        self._k = 2.0 / 3.0

    def __str__(self):
        return "Square-Root Law Impact Model"

    def __repr__(self):
        return f"SqrtImpactModel(Y={self.Y}, perm_fraction={self.perm_fraction}, perm_half_life_days={self.perm_half_life_days})"

    def apply_trade(
        self,
        trade_size: float,
        price: float,
        volatility: float,
        volume: float,
        symbol: str,
    ) -> TradeImpact:
        if volume <= 0.0 or trade_size == 0.0:
            return TradeImpact(cost=0.0, price_shift=0.0)

        participation = abs(trade_size) / volume
        peak_frac = self.Y * volatility * np.sqrt(participation)
        perm_frac = self.perm_fraction * peak_frac
        price_shift = np.sign(trade_size) * perm_frac * price
        cost = self._k * peak_frac * abs(trade_size) * price
        self._perm_state[symbol] += price_shift
        return TradeImpact(cost=cost, price_shift=price_shift)


class ACImpactModel(ImpactModel):
    """
    Classic Almgren-Chriss market impact model with full cost decomposition.

    This model captures both permanent and temporary market impact following
    the original Almgren-Chriss framework. Total execution cost is decomposed
    into three components:

    1. **Permanent Impact Cost** (quadratic in trade size):
       The price moves against you as you trade, creating a triangular cost:
           C_perm = 0.5 * α * σ * (x/V) * |x| * P

    2. **Temporary Impact Cost - Linear** (spread/fixed execution cost):
       Fixed cost per share representing bid-ask spread and execution fees:
           C_spread = ε * |x| * P

    3. **Temporary Impact Cost - Quadratic** (market depth depletion):
       Cost that grows with participation rate as you consume liquidity:
           C_depth = β * σ * (x/V) * |x| * P

    The permanent price shift affects future valuations:
        ΔP = α * σ * (x/V) * P

    Total cost:
        Cost = C_perm + C_spread + C_depth
             = [0.5 * α * σ * (x/V) + ε + β * σ * (x/V)] * |x| * P

    Parameters
    ----------
    alpha : float (default=1.0)
        Permanent impact coefficient. Scales the persistent price shift.
        Higher values = more price impact that doesn't revert.
    epsilon : float (default=0.0005)
        Linear temporary impact (half-spread) as a fraction of price.
        Default 5 bps represents typical S&P 500 half-spread.
        Set to 0 to disable spread costs.
    beta : float (default=1.0)
        Quadratic temporary impact coefficient. Scales the depth-depletion
        cost that grows with participation rate.

    Notes
    -----
    Default parameters are calibrated for S&P 500 / large-cap US equities:
    - ε = 5 bps (typical half-spread for liquid large-caps)
    - At 10% participation, 2% vol: ~35 bps permanent + depth, +5 bps spread

    The three-component model behaves as:
    - Small trades (1% ADV): spread cost dominates (~5 bps)
    - Medium trades (5% ADV): mixed (~15 bps spread+depth, ~10 bps perm)
    - Large trades (10% ADV): depth cost dominates (~25 bps depth, ~10 bps perm)

    For less liquid stocks, increase all three parameters.

    References
    ----------
    Almgren, R., & Chriss, N. (2001). Optimal execution of portfolio transactions.
        *Journal of Risk*, 3(2), 5-39.
    Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). Direct estimation of
        equity market impact. *Risk*, 18(7), 58-62.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        epsilon: float = 0.0005,
        perm_half_life_days: float = 5.0,
    ) -> None:
        super().__init__(perm_half_life_days=perm_half_life_days)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon

    def __str__(self):
        return "Almgren-Chriss Impact Model"

    def __repr__(self):
        return f"ACImpactModel(alpha={self.alpha}, beta={self.beta}, epsilon={self.epsilon}, perm_half_life_days={self.perm_half_life_days})"

    def apply_trade(
        self,
        trade_size: float,
        price: float,
        volatility: float,
        volume: float,
        symbol: str,
    ) -> TradeImpact:
        if volume <= 0.0 or trade_size == 0.0:
            return TradeImpact(cost=0.0, price_shift=0.0)

        participation = abs(trade_size) / volume

        # 1. Permanent impact: price shift that persists
        # Cost is triangular (average of 0 and full shift) = 0.5 * shift * shares
        eta = self.alpha * volatility
        price_shift = eta * (trade_size / volume) * price
        perm_cost = 0.5 * abs(price_shift) * abs(trade_size)

        # 2. Linear temporary impact: spread/execution cost (fixed per share)
        spread_cost = self.epsilon * abs(trade_size) * price

        # 3. Quadratic temporary impact: depth depletion (grows with participation)
        gamma = self.beta * volatility
        depth_cost = gamma * participation * abs(trade_size) * price

        total_cost = perm_cost + spread_cost + depth_cost
        self._perm_state[symbol] += price_shift

        return TradeImpact(cost=total_cost, price_shift=price_shift)


class OWImpactModel(ImpactModel):
    """
    Obizhaeva–Wang market impact model with transient decay.

    This model extends the basic impact framework by adding a transient component
    that captures the temporary price displacement from past trades, which decays
    exponentially as the limit order book (LOB) resilience restores liquidity.

    The model decomposes impact into three components:

    1. **Permanent Impact**: A fraction of each trade's impact that persists indefinitely.
       ΔP_perm = perm_fraction * I(x) * sign(x) * P
       where I(x) = Y * σ * sqrt(|x|/V) is the total instantaneous impact.

    2. **Temporary Impact**: The immediate cost of crossing the spread and walking
       the book, which does not persist.
       C_temp = (1 - perm_fraction) * I(x) * |x| * P

    3. **Transient Impact**: A decaying memory of past trades' impact on liquidity.
       The transient state S_t evolves as:
           S_t = S_{t-1} * exp(-κ) + I(x_t)
       The transient cost is: C_trans = S_{t-1} * |x_t| * P
       This captures that trading into a recently-depleted order book is more expensive.

    Parameters
    ----------
    Y : float (default=0.6)
        Square-root impact coefficient. Empirically validated range: 0.5–1.0.
    perm_fraction : float (default=0.25)
        Fraction of instantaneous impact that becomes permanent (~25% empirically).
    half_life_days : float (default=0.08)
        Half-life of transient impact in trading days. Default ~30 minutes
        (30/390 ≈ 0.077 days). Typical range: 15–60 minutes for liquid stocks.

    Notes
    -----
    The half-life parameter controls LOB resilience speed:
    - Liquid large-caps: ~15-30 min (half_life_days ≈ 0.04-0.08)
    - Mid-caps: ~30-60 min (half_life_days ≈ 0.08-0.15)
    - Small-caps/illiquid: can be hours (half_life_days ≈ 0.25-1.0)

    For daily trading (one trade per day), transient effects largely wash out
    overnight. Set half_life_days < 0.5 for realistic daily simulation.

    References
    ----------
    Obizhaeva, A. & Wang, J. (2013). Optimal trading strategy and supply/demand
        dynamics. Journal of Financial Markets, 16(1), 1–32.
    Gatheral, J. & Schied, A. (2013). Dynamical Models of Market Impact and
        Algorithms for Order Execution. Handbook on Systemic Risk, pp. 579–599.
    """

    def __init__(
        self,
        Y: float = 0.6,
        perm_fraction: float = 0.25,
        half_life_days: float = 0.08,
        perm_half_life_days: float = 5.0,
    ):
        super().__init__(perm_half_life_days=perm_half_life_days)
        self.Y = Y
        self.perm_fraction = perm_fraction
        self.half_life_days = half_life_days
        # Decay rate: κ such that exp(-κ) = 0.5 after half_life_days
        self.kappa = np.log(2) / half_life_days if half_life_days > 0 else np.inf
        # Transient state per symbol: accumulated impact that decays over time
        self._transient_states = defaultdict(float)

    def __str__(self):
        return "Obizhaeva-Wang Impact Model"

    def __repr__(self):
        return f"OWImpactModel(Y={self.Y}, perm_fraction={self.perm_fraction}, half_life_days={self.half_life_days}, perm_half_life_days={self.perm_half_life_days})"

    def reset(self) -> None:
        """Reset both permanent and transient state."""
        super().reset()
        self._transient_states = defaultdict(float)

    def apply_trade(
        self,
        trade_size: float,
        price: float,
        volatility: float,
        volume: float,
        symbol: str,
    ) -> TradeImpact:
        """
        Execute a trade and compute cost and permanent price shift.

        The transient state decays by exp(-κ) per time step (assumed to be 1 day).
        If trading intraday, adjust half_life_days accordingly.
        """
        if volume <= 0.0 or trade_size == 0.0:
            return TradeImpact(cost=0.0, price_shift=0.0)

        participation = abs(trade_size) / volume

        # Instantaneous impact fraction: I(x) = Y * σ * sqrt(participation)
        instant_impact_frac = self.Y * volatility * np.sqrt(participation)

        # 1. Permanent impact (persists forever)
        perm_frac = self.perm_fraction * instant_impact_frac
        price_shift = np.sign(trade_size) * perm_frac * price
        # Cost: integral of linear price path → triangular area = 0.5 * shift * shares
        perm_cost = 0.5 * abs(price_shift) * abs(trade_size)

        # 2. Temporary impact (immediate, non-persistent)
        temp_frac = (1 - self.perm_fraction) * instant_impact_frac
        temp_cost = temp_frac * abs(trade_size) * price

        # 3. Transient impact from past trades (decayed)
        # Decay previous transient state by one time step
        prev_transient = self._transient_states[symbol] * np.exp(-self.kappa)
        # Transient cost: trading into depleted book costs more
        # The transient state represents "residual impact fraction" from past trades
        trans_cost = prev_transient * abs(trade_size) * price

        # Update transient state: add this trade's instantaneous impact
        self._transient_states[symbol] = prev_transient + instant_impact_frac

        # Total cost
        total_cost = perm_cost + temp_cost + trans_cost
        self._perm_state[symbol] += price_shift

        return TradeImpact(cost=total_cost, price_shift=price_shift)


class BaselineImpactModel(ImpactModel):
    """
    A simple baseline impact model that charges a fixed 10 basis point fee.

    This model provides a non-zero cost baseline to compare against more
    sophisticated models like Almgren-Chriss. It assumes a fixed transaction
    cost of 0.10% of the trade's notional value and no permanent impact.
    """

    def __init__(self, basis_points=10, perm_half_life_days: float = 5.0):
        super().__init__(perm_half_life_days=perm_half_life_days)
        self.basis_points = basis_points
        self.fee_rate = basis_points / 10000  # Convert bps to a rate

    def __str__(self):
        return "Baseline Impact Model"

    def __repr__(self):
        return f"BaselineImpactModel(basis_points={self.basis_points})"

    def apply_trade(
        self,
        trade_size: float,
        price: float,
        volatility: float,  # Unused, for interface compatibility
        volume: float,  # Unused, for interface compatibility
        symbol: str,
    ) -> TradeImpact:
        """
        Calculates a fixed cost based on trade size and price.

        Parameters
        ----------
        trade_size : float
            Number of shares traded.
        price : float
            Price per share.

        Returns
        -------
        TradeImpact
            cost : notional cost of the trade (10 bps of total value)
            price_shift : permanent mid-price shift (always 0 for this model)
        """
        cost = self.fee_rate * abs(trade_size) * price

        # No permanent impact in the baseline model
        price_shift = 0.0

        return TradeImpact(cost=cost, price_shift=price_shift)

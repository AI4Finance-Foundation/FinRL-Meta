import logging
import sys

import numpy as np
import pandas as pd


def get_logger():
    logger = logging.getLogger("env_market_impact")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] - %(filename)s:%(lineno)d - [%(levelname)s]: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def rolling_sharpe(returns, window: int = 252, min_periods: int = 180):
    """Compute rolling annualized Sharpe ratio.

    Parameters
    ----------
    returns : array-like
        Daily return series.
    window : int
        Rolling window size in trading days.
    min_periods : int
        Minimum observations required for a valid value.

    Returns
    -------
    pd.Series
        Rolling annualized Sharpe ratio.
    """
    returns = pd.Series(returns, dtype=float)
    return returns.rolling(window=window, min_periods=min_periods).apply(
        lambda x: (x.mean() * 252) / (x.std() * (252**0.5)) if x.std() != 0 else 0
    )


def compute_performance_stats(
    portfolio_values,
    turnovers,
    costs,
    trades_df=None,
    rewards=None,
) -> dict:
    """Compute comprehensive performance statistics from simulation results.

    This is the single source of truth for all backtest performance metrics.
    Used by the epoch-evaluation loop, the report-generator performance table,
    and any other place that needs these numbers.

    Parameters
    ----------
    portfolio_values : array-like
        Daily portfolio values.
    turnovers : array-like
        Daily turnovers as fractions (0-1 scale).
    costs : array-like
        Daily trading costs in dollars.
    trades_df : DataFrame, optional
        Trade-level data with columns ``'pov'``, ``'notional'``, and
        ``'turnover_percentile'``.
    rewards : array-like, optional
        Per-step reward values from the environment.

    Returns
    -------
    dict
        Keys (all values are plain floats):

        * ``total_return`` -- total return (%).
        * ``annualized_return`` -- annualized return (%).
        * ``total_sharpe`` -- total (non-annualized) Sharpe ratio.
        * ``annualized_sharpe`` -- annualized Sharpe ratio.
        * ``annualized_volatility`` -- annualized daily-return volatility (%).
        * ``sortino_ratio`` -- annualized Sortino ratio (downside deviation only).
        * ``calmar_ratio`` -- annualized return / |max drawdown|.
        * ``avg_daily_turnover`` -- average daily turnover (%).
        * ``max_drawdown`` -- maximum drawdown (%, negative).
        * ``total_trading_cost`` -- total trading cost ($).
        * ``avg_daily_trading_cost`` -- average daily trading cost ($).
        * ``avg_order_pov`` -- simple average order POV (%).
        * ``wtd_avg_turnover_percentile`` -- notional-weighted avg turnover percentile.
        * ``avg_step_reward`` -- average per-step reward (only when *rewards* is provided).
    """
    portfolio_values = pd.Series(portfolio_values, dtype=float)
    turnovers = pd.Series(turnovers, dtype=float)
    costs = pd.Series(costs, dtype=float)

    _ZEROS = {
        "total_return": 0.0,
        "annualized_return": 0.0,
        "total_sharpe": 0.0,
        "annualized_sharpe": 0.0,
        "annualized_volatility": 0.0,
        "sortino_ratio": 0.0,
        "calmar_ratio": 0.0,
        "avg_daily_turnover": 0.0,
        "max_drawdown": 0.0,
        "total_trading_cost": 0.0,
        "avg_daily_trading_cost": 0.0,
        "avg_order_pov": 0.0,
        "wtd_avg_turnover_percentile": 0.0,
        "avg_step_reward": 0.0,
    }

    if portfolio_values.empty:
        return _ZEROS.copy()

    initial_value = portfolio_values.iloc[0]
    final_value = portfolio_values.iloc[-1]
    days = len(portfolio_values)

    # Total return (%)
    total_return = ((final_value - initial_value) / initial_value) * 100

    # Annualized return (%)
    annualized_return = (
        ((final_value / initial_value) ** (252 / days) - 1) * 100 if days > 0 else 0.0
    )

    # Sharpe ratios & volatility
    daily_returns = portfolio_values.pct_change().dropna()
    std = daily_returns.std()
    if std != 0:
        total_sharpe = daily_returns.mean() / std
        annualized_sharpe = total_sharpe * (252**0.5)
    else:
        total_sharpe = 0.0
        annualized_sharpe = 0.0

    # Annualized volatility (%)
    annualized_volatility = std * (252**0.5) * 100 if std != 0 else 0.0

    # Sortino ratio (annualized, downside deviation only)
    negative_returns = daily_returns[daily_returns < 0]
    downside_std = negative_returns.std()
    if downside_std != 0 and len(negative_returns) > 1:
        sortino_ratio = (daily_returns.mean() / downside_std) * (252**0.5)
    else:
        sortino_ratio = 0.0

    # Avg daily turnover (%)
    avg_daily_turnover = turnovers.mean() * 100

    # Max drawdown (%)
    rolling_max = portfolio_values.expanding(min_periods=1).max()
    drawdown = (portfolio_values - rolling_max) / rolling_max * 100
    max_drawdown = drawdown.min()

    # Calmar ratio (annualized return / |max drawdown|)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0

    # Trading costs
    total_trading_cost = costs.sum()
    avg_daily_trading_cost = costs.mean()

    # Trade-level metrics
    if trades_df is not None and not trades_df.empty:
        avg_order_pov = trades_df["pov"].mean() * 100
        total_notional = trades_df["notional"].sum()
        if total_notional > 0:
            wtd_avg_turnover_percentile = (
                trades_df["turnover_percentile"] * trades_df["notional"]
            ).sum() / total_notional
        else:
            wtd_avg_turnover_percentile = 0.0
    else:
        avg_order_pov = 0.0
        wtd_avg_turnover_percentile = 0.0

    # Reward metrics
    if rewards is not None:
        rewards = pd.Series(rewards, dtype=float)
        avg_step_reward = float(rewards.mean()) if not rewards.empty else 0.0
    else:
        avg_step_reward = 0.0

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized_return),
        "total_sharpe": float(total_sharpe),
        "annualized_sharpe": float(annualized_sharpe),
        "annualized_volatility": float(annualized_volatility),
        "sortino_ratio": float(sortino_ratio),
        "calmar_ratio": float(calmar_ratio),
        "avg_daily_turnover": float(avg_daily_turnover),
        "max_drawdown": float(max_drawdown),
        "total_trading_cost": float(total_trading_cost),
        "avg_daily_trading_cost": float(avg_daily_trading_cost),
        "avg_order_pov": float(avg_order_pov),
        "wtd_avg_turnover_percentile": float(wtd_avg_turnover_percentile),
        "avg_step_reward": float(avg_step_reward),
    }

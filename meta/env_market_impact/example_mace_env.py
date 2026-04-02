"""
Example script for backtesting DRL agents with the MACEStockTradingEnv.

This script demonstrates how to:
1.  Download and preprocess real market data for NASDAQ 100 stocks using FinRL's utilities.
2.  Configure and instantiate the MACEStockTradingEnv with different market impact models
    (e.g., SqrtImpactModel, ACImpactModel, BaselineImpactModel).
3.  Set up and train various DRL agents (A2C, PPO, DDPG) from the Stable Baselines3 library
    using a unified DRLAgent wrapper.
4.  Run backtesting simulations for different combinations of DRL agents, impact models,
    and initial capital levels.
5.  Provide a framework for comparing the performance of different trading strategies
    under realistic market impact conditions.
"""

import json
import os
import sys
import uuid

import numpy as np
import pandas as pd
from finrl.config import INDICATORS
from finrl.config_tickers import NAS_100_TICKER
from finrl.config_tickers import SP_500_TICKER

# Ensure the project root is in the path to resolve imports
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from meta.env_market_impact.envs.env_mace_stock_trading import (
    MACEStockTradingEnv,
    EnvParams,
)
from meta.env_market_impact.envs.impact_models import (
    SqrtImpactModel,
    ACImpactModel,
    BaselineImpactModel,
    ImpactModel,
)
from meta.env_market_impact.envs.market_data import MarketDataPreparator, Split
from meta.env_market_impact.backtest_report_generator import BacktestReportGenerator
from meta.env_market_impact.backtest_config import BacktestParams, MODEL_KWARGS
from meta.env_market_impact.envs.utils import get_logger, compute_performance_stats
from agents.stablebaselines3_models import DRLAgent, TensorboardCallback
from stable_baselines3.common.vec_env import DummyVecEnv

log = get_logger()


def run_and_log_simulation(
    env: MACEStockTradingEnv,
    model,
    dates: list,
    benchmark_df: pd.DataFrame,
    reset_impact_model: bool = True,
    initial_benchmark_value: float = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs a simulation and returns the results and trades as DataFrames."""
    state, _ = env.reset(options={"reset_impact_model": reset_impact_model})
    results_log = []
    trades_log = []
    last_asset_value = env.total_asset

    # Use initial_benchmark_value if provided, otherwise use env's initial capital.
    start_value = (
        initial_benchmark_value
        if initial_benchmark_value is not None
        else env.initial_capital
    )

    # Calculate benchmark cumulative return
    benchmark_cumulative_return = benchmark_df["close"] / benchmark_df["close"].iloc[0]
    benchmark_value_series = start_value * benchmark_cumulative_return

    # Log the initial state at time=0 (the reset point) so the first test
    # entry is continuous with the last training entry.
    results_log.append(
        {
            "date": dates[env.time],
            "step": -1,
            "portfolio_value": env.total_asset,
            "pnl": 0.0,
            "reward": 0.0,
            "turnover": 0.0,
            "cost": 0.0,
            "total_buy_value": 0.0,
            "total_sell_value": 0.0,
            "benchmark_value": benchmark_value_series.iloc[env.time],
            "cash": env.cash,
        }
    )

    max_steps = env.max_step
    for step in range(max_steps):
        actions, _ = model.predict(state, deterministic=True)
        next_state, reward, done, _, info = env.step(actions)
        state = next_state

        current_asset_value = env.total_asset
        pnl = current_asset_value - last_asset_value
        last_asset_value = current_asset_value

        results_log.append(
            {
                "date": dates[env.time],
                "step": step,
                "portfolio_value": current_asset_value,
                "pnl": pnl,
                "reward": reward,
                "turnover": info["turnover"],
                "cost": info["cost"],
                "total_buy_value": info["total_buy_value"],
                "total_sell_value": info["total_sell_value"],
                "benchmark_value": benchmark_value_series.iloc[env.time],
                "cash": info["cash"],
            }
        )

        # Log individual trades
        for trade in info.get("trades", []):
            trade["date"] = dates[env.time]
            trade["step"] = step
            trades_log.append(trade)

        if done:
            break

    return pd.DataFrame(results_log), pd.DataFrame(trades_log)


def train_with_epoch_evaluation(
    model,
    train_env: MACEStockTradingEnv,
    train_config: dict,
    trade_config: dict,
    train_benchmark_df: pd.DataFrame,
    trade_benchmark_df: pd.DataFrame,
    params: BacktestParams,
    num_epochs: int,
    num_training_days: int,
    tb_log_name: str,
) -> tuple:
    """Train a model epoch-by-epoch with per-epoch evaluation.

    After every epoch (one full pass through the training data), the model is
    evaluated deterministically on both the training set and a blank-slate OOS
    test set.  Performance stats are computed via the shared
    ``compute_performance_stats`` utility.

    Parameters
    ----------
    model
        A Stable-Baselines3 model (already initialised with the training env).
    train_env : MACEStockTradingEnv
        The *actual* training environment (used only to snapshot the normaliser
        state -- evaluation runs on fresh copies).
    train_config, trade_config : dict
        Environment config dicts produced by ``MarketDataPreparator.create_env_config``.
    train_benchmark_df, trade_benchmark_df : DataFrame
        Benchmark price data aligned with train / trade periods.
    params : BacktestParams
        The backtest configuration for this run.  Used to create fresh
        evaluation environments with the correct impact model and env kwargs.
    num_epochs : int
        Number of training epochs.
    num_training_days : int
        Number of environment steps per epoch.
    tb_log_name : str
        TensorBoard log name.

    Returns
    -------
    tuple[model, list[dict], list[dict]]
        ``(trained_model, epoch_stats_train, epoch_stats_test_blank)``
        where each stats list contains one dict per epoch with keys from
        ``compute_performance_stats`` plus an ``"epoch"`` key.
    """
    epoch_stats_train: list[dict] = []
    epoch_stats_test_blank: list[dict] = []

    for epoch in range(num_epochs):
        model.learn(
            total_timesteps=num_training_days,
            reset_num_timesteps=(epoch == 0),
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )
        log.info(f"Epoch {epoch + 1}/{num_epochs} complete. Evaluating...")

        # Snapshot normaliser state for eval envs
        normalizer_state_for_eval = train_env.get_normalizer_state()

        # --- Evaluate on training set ---
        eval_train_env = MACEStockTradingEnv(
            config=train_config,
            params=params.env_params,
            impact_model=params.impact_model_class(),
            initial_capital=params.initial_capital,
        )
        if normalizer_state_for_eval is not None:
            eval_train_env.set_normalizer_state(normalizer_state_for_eval, freeze=True)
        eval_train_results, eval_train_trades = run_and_log_simulation(
            eval_train_env,
            model,
            train_config["date_list"],
            train_benchmark_df,
            reset_impact_model=True,
        )
        train_stats = compute_performance_stats(
            eval_train_results["portfolio_value"],
            eval_train_results["turnover"],
            eval_train_results["cost"],
            eval_train_trades,
            rewards=eval_train_results["reward"],
        )
        train_stats["epoch"] = epoch + 1
        epoch_stats_train.append(train_stats)

        # --- Evaluate on blank-slate test set ---
        eval_trade_env = MACEStockTradingEnv(
            config=trade_config,
            params=params.env_params,
            impact_model=params.impact_model_class(),
            initial_capital=params.initial_capital,
        )
        if normalizer_state_for_eval is not None:
            eval_trade_env.set_normalizer_state(normalizer_state_for_eval, freeze=True)
        eval_trade_results, eval_trade_trades = run_and_log_simulation(
            eval_trade_env,
            model,
            trade_config["date_list"],
            trade_benchmark_df,
            reset_impact_model=True,
        )
        test_blank_stats = compute_performance_stats(
            eval_trade_results["portfolio_value"],
            eval_trade_results["turnover"],
            eval_trade_results["cost"],
            eval_trade_trades,
            rewards=eval_trade_results["reward"],
        )
        test_blank_stats["epoch"] = epoch + 1
        epoch_stats_test_blank.append(test_blank_stats)

        log.info(
            f"  Train: Ann. Return={train_stats['annualized_return']:.2f}%, "
            f"Sharpe={train_stats['annualized_sharpe']:.3f}"
        )
        log.info(
            f"  Test:  Ann. Return={test_blank_stats['annualized_return']:.2f}%, "
            f"Sharpe={test_blank_stats['annualized_sharpe']:.3f}"
        )

    log.info("Training complete.")
    return model, epoch_stats_train, epoch_stats_test_blank


def train_and_backtest(
    data_prep: MarketDataPreparator,
    backtest_grid: list[BacktestParams],
    num_epochs: int = 20,
) -> str:
    """
    Trains agents on the training split and runs OOS backtests on the trade split.

    Parameters
    ----------
    data_prep : MarketDataPreparator
        MarketDataPreparator instance.
    backtest_grid : list[BacktestParams]
        Pre-generated grid of all parameter combinations to evaluate.
        Use ``BacktestParams.generate_grid(...)`` to build this.
    num_epochs : int
        Number of training epochs per configuration.
    """
    # Create configurations for training and trading environments
    train_config = data_prep.create_env_config(Split.TRAIN)
    trade_config = data_prep.create_env_config(Split.TRADE)

    run_id = str(uuid.uuid4())
    results_dir = f"backtest_results/{run_id}"
    os.makedirs(results_dir)

    all_backtests_metadata = []
    train_benchmark_df = data_prep.get_benchmark_df(Split.TRAIN)
    trade_benchmark_df = data_prep.get_benchmark_df(Split.TRADE)
    benchmark_ticker = data_prep.benchmark_ticker

    log.info(f"Running {len(backtest_grid)} backtest configuration(s)...")

    for idx, params in enumerate(backtest_grid, 1):
        impact_model_instance = params.impact_model_class()
        base_filename = params.base_filename

        log.info(
            f"[{idx}/{len(backtest_grid)}] "
            f"{params.model_name.upper()} | {params.impact_model_name} | "
            f"capital=${params.initial_capital:,.0f}"
        )

        # --- Train the agent ---
        train_env = MACEStockTradingEnv(
            config=train_config,
            params=params.env_params,
            impact_model=impact_model_instance,
            initial_capital=params.initial_capital,
        )
        vec_train_env = DummyVecEnv([lambda: train_env])
        agent = DRLAgent(env=vec_train_env)
        model = agent.get_model(
            params.model_name,
            model_kwargs=params.get_model_kwargs(),
            policy_kwargs=params.policy_kwargs,
            seed=42,
        )

        num_training_days = len(train_config["date_list"])
        total_timesteps = num_training_days * num_epochs
        log.info(
            f"Training for {num_epochs} epochs ({total_timesteps} total timesteps)."
        )

        trained_model, epoch_stats_train, epoch_stats_test_blank = (
            train_with_epoch_evaluation(
                model=model,
                train_env=train_env,
                train_config=train_config,
                trade_config=trade_config,
                train_benchmark_df=train_benchmark_df,
                trade_benchmark_df=trade_benchmark_df,
                params=params,
                num_epochs=num_epochs,
                num_training_days=num_training_days,
                tb_log_name=base_filename,
            )
        )

        # --- Run in-sample (training) simulation ---
        log.info("--- Running In-Sample Simulation ---")
        train_results_df, train_trades_df = run_and_log_simulation(
            train_env,
            trained_model,
            train_config["date_list"],
            train_benchmark_df,
            reset_impact_model=True,
        )
        train_csv_filename = f"{results_dir}/{base_filename}_train.csv"
        train_trades_csv_filename = f"{results_dir}/{base_filename}_train_trades.csv"
        train_results_df.to_csv(train_csv_filename, index=False)
        train_trades_df.to_csv(train_trades_csv_filename, index=False)
        log.info(f"Training results saved to {train_csv_filename}")
        normalizer_state = train_env.get_normalizer_state()

        # --- Run OOS backtest (continued from training) ---
        log.info("--- Running OOS Backtest ---")
        trade_env = MACEStockTradingEnv(
            config=trade_config,
            params=params.env_params,
            impact_model=impact_model_instance,
            initial_capital=train_env.cash,
            initial_stocks=train_env.stocks,
        )
        if normalizer_state is not None:
            trade_env.set_normalizer_state(normalizer_state, freeze=True)
        last_train_benchmark_value = train_results_df["benchmark_value"].iloc[-1]
        trade_results_df, trade_trades_df = run_and_log_simulation(
            trade_env,
            trained_model,
            trade_config["date_list"],
            trade_benchmark_df,
            reset_impact_model=False,
            initial_benchmark_value=last_train_benchmark_value,
        )
        test_csv_filename = f"{results_dir}/{base_filename}_test.csv"
        test_trades_csv_filename = f"{results_dir}/{base_filename}_test_trades.csv"
        trade_results_df.to_csv(test_csv_filename, index=False)
        trade_trades_df.to_csv(test_trades_csv_filename, index=False)
        log.info(f"Test results saved to {test_csv_filename}")

        # --- Run blank-slate OOS backtest ---
        log.info("--- Running OOS Backtest (blank slate) ---")
        trade_env_blank = MACEStockTradingEnv(
            config=trade_config,
            params=params.env_params,
            impact_model=impact_model_instance,
            initial_capital=params.initial_capital,
        )
        if normalizer_state is not None:
            trade_env_blank.set_normalizer_state(normalizer_state, freeze=True)
        trade_results_blank_df, trade_blank_trades_df = run_and_log_simulation(
            trade_env_blank,
            trained_model,
            trade_config["date_list"],
            trade_benchmark_df,
            reset_impact_model=True,
        )
        test_blank_csv_filename = f"{results_dir}/{base_filename}_test_blank.csv"
        test_blank_trades_csv_filename = (
            f"{results_dir}/{base_filename}_test_blank_trades.csv"
        )
        trade_results_blank_df.to_csv(test_blank_csv_filename, index=False)
        trade_blank_trades_df.to_csv(test_blank_trades_csv_filename, index=False)
        log.info(f"Test (blank slate) results saved to {test_blank_csv_filename}")

        effective_model_kwargs = params.get_model_kwargs()
        all_backtests_metadata.append(
            {
                "drl_agent": params.model_name,
                "impact_model": params.impact_model_name,
                "initial_capital": params.initial_capital,
                "results_csv_train": train_csv_filename,
                "results_csv_test": test_csv_filename,
                "results_csv_test_blank": test_blank_csv_filename,
                "trades_csv_train": train_trades_csv_filename,
                "trades_csv_test": test_trades_csv_filename,
                "trades_csv_test_blank": test_blank_trades_csv_filename,
                "with_perm": params.env_params.include_permanent_impact_in_state,
                "with_cooldown": params.env_params.include_cooldown_in_state,
                "with_tbill": params.env_params.include_tbill_in_state,
                "eta_dd": params.env_params.eta_dd,
                "use_obs_normalizer": params.env_params.use_obs_normalizer,
                "reward_scaling": params.env_params.reward_scaling,
                "horizon": params.env_params.horizon,
                "obs_clip": params.env_params.obs_clip,
                # Agent hyperparameters (flattened for report differentiation)
                "learning_rate": effective_model_kwargs.get("learning_rate"),
                "gamma": effective_model_kwargs.get("gamma"),
                "ent_coef": effective_model_kwargs.get("ent_coef"),
                "net_arch": (
                    str(params.policy_kwargs.get("net_arch"))
                    if params.policy_kwargs
                    else None
                ),
                "model_kwargs": effective_model_kwargs,
                "policy_kwargs": params.policy_kwargs,
                "epoch_stats_train": epoch_stats_train,
                "epoch_stats_test_blank": epoch_stats_test_blank,
            }
        )

    # Save all metadata to a single JSON file for the run
    summary_filename = f"{results_dir}/backtest_summary.json"
    summary_data = {
        "benchmark_ticker": benchmark_ticker,
        "backtests": all_backtests_metadata,
    }
    with open(summary_filename, "w") as f:
        json.dump(summary_data, f, indent=4, default=str)
    log.info(f"All backtest metadata saved to {summary_filename}")
    return summary_filename


def run_example():
    """Run a simple example with the impact environment using real data."""
    np.random.seed(42)
    log.info("Preparing tickers data...")

    start_date = "2010-01-01"
    end_date = "2026-01-01"
    num_epochs = 20

    # Using equal-weighted index ETFs for the benchmark given max exposure constraints
    data_prep = MarketDataPreparator(
        tickers=NAS_100_TICKER,
        # tickers=SP_500_TICKER,
        start_date=start_date,
        end_date=end_date,
        tech_indicators=INDICATORS,
        train_ratio=0.9,
        benchmark_ticker="QQEW",
        # benchmark_ticker="RSP",
    )

    # # --- Build the parameter grid (Cartesian product) ---
    # # perm=True, cooldown=True, tbill=True, eta_dd=0.5, use_obs_normalizer=False seems to be the best combination
    # backtest_grid = BacktestParams.generate_grid(
    #     models_to_run=["a2c"],
    #     # models_to_run=["a2c", "ddpg", "ppo", "sac", "td3"],
    #     initial_capitals=[1e9],
    #     # initial_capitals=[1e7, 1e8, 1e9],
    #     impact_model_classes=[BaselineImpactModel, ACImpactModel],
    #     # impact_model_classes=[BaselineImpactModel, SqrtImpactModel, ACImpactModel],
    #     num_stocks=data_prep.universe_size,
    #     include_permanent_impact_in_state=[True],
    #     # include_permanent_impact_in_state=[False, True],
    #     include_cooldown_in_state=[True],
    #     # include_cooldown_in_state=[False, True],
    #     include_tbill_in_state=[True],
    #     # include_tbill_in_state=[False, True],
    #     use_obs_normalizer=[False],
    #     # use_obs_normalizer=[False, True],
    #     eta_dd=[0.5],
    #     # eta_dd=[0.5, 1.0, 2.0],
    # )

    # --- 20 runs: 5 agents x 2 impact models x 2 hyperparam sets (default / HPO) ---
    backtest_grid = BacktestParams.from_explicit(
        configs=[
            # ── A2C ──────────────────────────────────────────────────
            # A2C | Baseline | default params
            {
                "model_name": "a2c",
                "impact_model_class": BaselineImpactModel,
                "initial_capital": 1e9,
            },
            # A2C | ACImpact | default params
            {
                "model_name": "a2c",
                "impact_model_class": ACImpactModel,
                "initial_capital": 1e9,
            },
            # A2C | Baseline | HPO
            {
                "model_name": "a2c",
                "impact_model_class": BaselineImpactModel,
                "initial_capital": 1e9,
                "model_kwargs": {
                    "learning_rate": 9.345473014207437e-05,
                    "n_steps": 50,
                    "ent_coef": 0.0007879004401207801,
                    "gamma": 0.9088236029155305,
                    "gae_lambda": 0.8085192755608721,
                },
                "policy_kwargs": {"net_arch": [256, 128, 64]},
                "eta_dd": 0.5920255491056137,
                "horizon": 40,
                "include_cooldown_in_state": False,
                "include_permanent_impact_in_state": False,
                "include_tbill_in_state": True,
                "obs_clip": 7.43605870267289,
                "reward_scaling": 0.000230613280155283,
                "use_obs_normalizer": True,
            },
            # A2C | ACImpact | HPO (OOS Sharpe 1.7923)
            {
                "model_name": "a2c",
                "impact_model_class": ACImpactModel,
                "initial_capital": 1e9,
                "model_kwargs": {
                    "learning_rate": 9.345473014207437e-05,
                    "n_steps": 50,
                    "ent_coef": 0.0007879004401207801,
                    "gamma": 0.9088236029155305,
                    "gae_lambda": 0.8085192755608721,
                },
                "policy_kwargs": {"net_arch": [256, 128, 64]},
                "eta_dd": 0.5920255491056137,
                "horizon": 40,
                "include_cooldown_in_state": False,
                "include_permanent_impact_in_state": False,
                "include_tbill_in_state": True,
                "obs_clip": 7.43605870267289,
                "reward_scaling": 0.000230613280155283,
                "use_obs_normalizer": True,
            },
            # ── PPO ──────────────────────────────────────────────────
            # PPO | Baseline | default params
            {
                "model_name": "ppo",
                "impact_model_class": BaselineImpactModel,
                "initial_capital": 1e9,
            },
            # PPO | ACImpact | default params
            {
                "model_name": "ppo",
                "impact_model_class": ACImpactModel,
                "initial_capital": 1e9,
            },
            # PPO | Baseline | HPO
            {
                "model_name": "ppo",
                "impact_model_class": BaselineImpactModel,
                "initial_capital": 1e9,
                "model_kwargs": {
                    "learning_rate": 1.4157895169765918e-05,
                    "n_steps": 1024,
                    "batch_size": 128,
                    "ent_coef": 0.00017806504001840215,
                    "gamma": 0.9341189226765384,
                    "gae_lambda": 0.9244743605610781,
                    "clip_range": 0.1474194639040638,
                    "n_epochs": 10,
                },
                "policy_kwargs": {"net_arch": [256, 128, 64]},
                "eta_dd": 0.33303259024402543,
                "horizon": 80,
                "include_cooldown_in_state": True,
                "include_permanent_impact_in_state": True,
                "include_tbill_in_state": True,
                "reward_scaling": 0.0018445391435152074,
                "use_obs_normalizer": False,
            },
            # PPO | ACImpact | HPO (OOS Sharpe 1.6372)
            {
                "model_name": "ppo",
                "impact_model_class": ACImpactModel,
                "initial_capital": 1e9,
                "model_kwargs": {
                    "learning_rate": 1.4157895169765918e-05,
                    "n_steps": 1024,
                    "batch_size": 128,
                    "ent_coef": 0.00017806504001840215,
                    "gamma": 0.9341189226765384,
                    "gae_lambda": 0.9244743605610781,
                    "clip_range": 0.1474194639040638,
                    "n_epochs": 10,
                },
                "policy_kwargs": {"net_arch": [256, 128, 64]},
                "eta_dd": 0.33303259024402543,
                "horizon": 80,
                "include_cooldown_in_state": True,
                "include_permanent_impact_in_state": True,
                "include_tbill_in_state": True,
                "reward_scaling": 0.0018445391435152074,
                "use_obs_normalizer": False,
            },
            # ── DDPG ─────────────────────────────────────────────────
            # DDPG | Baseline | default params
            {
                "model_name": "ddpg",
                "impact_model_class": BaselineImpactModel,
                "initial_capital": 1e9,
            },
            # DDPG | ACImpact | default params
            {
                "model_name": "ddpg",
                "impact_model_class": ACImpactModel,
                "initial_capital": 1e9,
            },
            # DDPG | Baseline | HPO
            {
                "model_name": "ddpg",
                "impact_model_class": BaselineImpactModel,
                "initial_capital": 1e9,
                "model_kwargs": {
                    "learning_rate": 0.0014255481008962508,
                    "batch_size": 64,
                    "buffer_size": 50000,
                    "gamma": 0.9667387593404914,
                    "tau": 0.020555160629308206,
                },
                "policy_kwargs": {"net_arch": [256, 128, 64]},
                "eta_dd": 2.9533866835352414,
                "horizon": 80,
                "include_cooldown_in_state": False,
                "include_permanent_impact_in_state": True,
                "include_tbill_in_state": False,
                "reward_scaling": 0.0002598848592778177,
                "use_obs_normalizer": False,
            },
            # DDPG | ACImpact | HPO (OOS Sharpe 1.3408)
            {
                "model_name": "ddpg",
                "impact_model_class": ACImpactModel,
                "initial_capital": 1e9,
                "model_kwargs": {
                    "learning_rate": 0.0014255481008962508,
                    "batch_size": 64,
                    "buffer_size": 50000,
                    "gamma": 0.9667387593404914,
                    "tau": 0.020555160629308206,
                },
                "policy_kwargs": {"net_arch": [256, 128, 64]},
                "eta_dd": 2.9533866835352414,
                "horizon": 80,
                "include_cooldown_in_state": False,
                "include_permanent_impact_in_state": True,
                "include_tbill_in_state": False,
                "reward_scaling": 0.0002598848592778177,
                "use_obs_normalizer": False,
            },
            # ── SAC ──────────────────────────────────────────────────
            # SAC | Baseline | default params
            {
                "model_name": "sac",
                "impact_model_class": BaselineImpactModel,
                "initial_capital": 1e9,
            },
            # SAC | ACImpact | default params
            {
                "model_name": "sac",
                "impact_model_class": ACImpactModel,
                "initial_capital": 1e9,
            },
            # SAC | Baseline | HPO
            {
                "model_name": "sac",
                "impact_model_class": BaselineImpactModel,
                "initial_capital": 1e9,
                "model_kwargs": {
                    "learning_rate": 0.00011684631812979796,
                    "batch_size": 256,
                    "buffer_size": 10000,
                    "gamma": 0.9989066452742179,
                    "tau": 0.015505359206935673,
                    "ent_coef": "auto",
                },
                "policy_kwargs": {"net_arch": [64, 64]},
                "eta_dd": 0.29359454642025673,
                "horizon": 10,
                "include_cooldown_in_state": True,
                "include_permanent_impact_in_state": False,
                "include_tbill_in_state": True,
                "obs_clip": 19.694697495218435,
                "reward_scaling": 0.00015558508759195628,
                "use_obs_normalizer": True,
            },
            # SAC | ACImpact | HPO (OOS Sharpe 1.7192)
            {
                "model_name": "sac",
                "impact_model_class": ACImpactModel,
                "initial_capital": 1e9,
                "model_kwargs": {
                    "learning_rate": 0.00011684631812979796,
                    "batch_size": 256,
                    "buffer_size": 10000,
                    "gamma": 0.9989066452742179,
                    "tau": 0.015505359206935673,
                    "ent_coef": "auto",
                },
                "policy_kwargs": {"net_arch": [64, 64]},
                "eta_dd": 0.29359454642025673,
                "horizon": 10,
                "include_cooldown_in_state": True,
                "include_permanent_impact_in_state": False,
                "include_tbill_in_state": True,
                "obs_clip": 19.694697495218435,
                "reward_scaling": 0.00015558508759195628,
                "use_obs_normalizer": True,
            },
            # ── TD3 ──────────────────────────────────────────────────
            # TD3 | Baseline | default params
            {
                "model_name": "td3",
                "impact_model_class": BaselineImpactModel,
                "initial_capital": 1e9,
            },
            # TD3 | ACImpact | default params
            {
                "model_name": "td3",
                "impact_model_class": ACImpactModel,
                "initial_capital": 1e9,
            },
            # TD3 | Baseline | HPO
            {
                "model_name": "td3",
                "impact_model_class": BaselineImpactModel,
                "initial_capital": 1e9,
                "model_kwargs": {
                    "learning_rate": 0.00015265277410336577,
                    "batch_size": 128,
                    "buffer_size": 50000,
                    "gamma": 0.974062775740949,
                    "tau": 0.04060258378797207,
                },
                "policy_kwargs": {"net_arch": [256, 128, 64]},
                "eta_dd": 1.08555509926195,
                "horizon": 10,
                "include_cooldown_in_state": True,
                "include_permanent_impact_in_state": False,
                "include_tbill_in_state": True,
                "obs_clip": 17.900488642468652,
                "reward_scaling": 0.00032703896615777903,
                "use_obs_normalizer": True,
            },
            # TD3 | ACImpact | HPO (OOS Sharpe 1.5991)
            {
                "model_name": "td3",
                "impact_model_class": ACImpactModel,
                "initial_capital": 1e9,
                "model_kwargs": {
                    "learning_rate": 0.00015265277410336577,
                    "batch_size": 128,
                    "buffer_size": 50000,
                    "gamma": 0.974062775740949,
                    "tau": 0.04060258378797207,
                },
                "policy_kwargs": {"net_arch": [256, 128, 64]},
                "eta_dd": 1.08555509926195,
                "horizon": 10,
                "include_cooldown_in_state": True,
                "include_permanent_impact_in_state": False,
                "include_tbill_in_state": True,
                "obs_clip": 17.900488642468652,
                "reward_scaling": 0.00032703896615777903,
                "use_obs_normalizer": True,
            },
        ],
        num_stocks=data_prep.universe_size,
    )
    log.info(f"Generated {len(backtest_grid)} backtest configurations.")

    summary_path = train_and_backtest(data_prep, backtest_grid, num_epochs)
    BacktestReportGenerator(summary_path).generate_report()
    log.info("\nBacktests complete.")


if __name__ == "__main__":
    run_example()

"""
Example script for backtesting DRL agents with the PortfolioOptimizationImpactEnv.

Mirrors the workflow of ``example_mace_env.py`` but uses the POE
(weight-based portfolio optimization) environment.  The original paper's
reward function (log return) and softmax action normalization are preserved.
"""
import json
import os
import sys
import uuid

import numpy as np
import pandas as pd

from finrl.config_tickers import NAS_100_TICKER
from finrl.config import INDICATORS

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from meta.env_market_impact.envs.env_portfolio_optimization_impact import (
    PortfolioOptimizationImpactEnv,
)
from meta.env_market_impact.envs.impact_models import (
    SqrtImpactModel,
    ACImpactModel,
    BaselineImpactModel,
)
from meta.env_market_impact.envs.market_data import MarketDataPreparator, Split
from meta.env_market_impact.backtest_report_generator import BacktestReportGenerator
from meta.env_market_impact.envs.utils import get_logger, compute_performance_stats
from meta.env_market_impact.backtest_config import MODEL_KWARGS, NET_ARCH
from agents.stablebaselines3_models import DRLAgent, TensorboardCallback
from stable_baselines3.common.vec_env import DummyVecEnv

log = get_logger()


# ── Simulation helper ─────────────────────────────────────────────────

def run_and_log_simulation(
    env: PortfolioOptimizationImpactEnv,
    model,
    dates: list,
    benchmark_df: pd.DataFrame,
    reset_impact_model: bool = True,
    initial_benchmark_value: float = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Runs a deterministic simulation and returns results + trades."""
    state, _ = env.reset(options={"reset_impact_model": reset_impact_model})
    results_log, trades_log = [], []
    last_asset_value = env.total_asset

    start_value = (
        initial_benchmark_value
        if initial_benchmark_value is not None
        else env.initial_capital
    )
    bench_cum = benchmark_df["close"] / benchmark_df["close"].iloc[0]
    bench_value = start_value * bench_cum

    results_log.append({
        "date": dates[env.time], "step": -1,
        "portfolio_value": env.total_asset, "pnl": 0.0, "reward": 0.0,
        "turnover": 0.0, "cost": 0.0,
        "total_buy_value": 0.0, "total_sell_value": 0.0,
        "benchmark_value": bench_value.iloc[env.time], "cash": env.cash,
    })

    for step in range(env.max_step):
        actions, _ = model.predict(state, deterministic=True)
        state, reward, done, _, info = env.step(actions)

        current = env.total_asset
        pnl = current - last_asset_value
        last_asset_value = current

        results_log.append({
            "date": dates[env.time], "step": step,
            "portfolio_value": current, "pnl": pnl, "reward": reward,
            "turnover": info["turnover"], "cost": info["cost"],
            "total_buy_value": info["total_buy_value"],
            "total_sell_value": info["total_sell_value"],
            "benchmark_value": bench_value.iloc[env.time],
            "cash": info["cash"],
        })
        for trade in info.get("trades", []):
            trade["date"] = dates[env.time]
            trade["step"] = step
            trades_log.append(trade)

        if done:
            break

    return pd.DataFrame(results_log), pd.DataFrame(trades_log)


# ── Epoch training with evaluation ────────────────────────────────────

def train_with_epoch_evaluation(
    model,
    train_env,
    train_config, trade_config,
    train_bench, trade_bench,
    env_kwargs: dict,
    impact_model_class: type,
    num_epochs: int,
    num_training_days: int,
    tb_log_name: str,
) -> tuple:
    epoch_stats_train, epoch_stats_test = [], []

    for epoch in range(num_epochs):
        model.learn(
            total_timesteps=num_training_days,
            reset_num_timesteps=(epoch == 0),
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )
        log.info(f"Epoch {epoch + 1}/{num_epochs} complete.")

        # Eval on train set
        eval_train = PortfolioOptimizationImpactEnv(
            config=train_config, impact_model=impact_model_class(), **env_kwargs
        )
        res_train, trades_train = run_and_log_simulation(
            eval_train, model, train_config["date_list"], train_bench,
        )
        ts = compute_performance_stats(
            res_train["portfolio_value"], res_train["turnover"],
            res_train["cost"], trades_train, rewards=res_train["reward"],
        )
        ts["epoch"] = epoch + 1
        epoch_stats_train.append(ts)

        # Eval on blank-slate test set
        eval_test = PortfolioOptimizationImpactEnv(
            config=trade_config, impact_model=impact_model_class(), **env_kwargs
        )
        res_test, trades_test = run_and_log_simulation(
            eval_test, model, trade_config["date_list"], trade_bench,
        )
        tt = compute_performance_stats(
            res_test["portfolio_value"], res_test["turnover"],
            res_test["cost"], trades_test, rewards=res_test["reward"],
        )
        tt["epoch"] = epoch + 1
        epoch_stats_test.append(tt)

        log.info(
            f"  Train Sharpe={ts['annualized_sharpe']:.3f} | "
            f"Test Sharpe={tt['annualized_sharpe']:.3f}"
        )

    return model, epoch_stats_train, epoch_stats_test


# ── Main orchestration ────────────────────────────────────────────────

def train_and_backtest(
    data_prep: MarketDataPreparator,
    configs: list[dict],
    num_epochs: int = 20,
) -> str:
    train_config = data_prep.create_env_config(Split.TRAIN)
    trade_config = data_prep.create_env_config(Split.TRADE)

    run_id = str(uuid.uuid4())
    results_dir = f"backtest_results/poe_{run_id}"
    os.makedirs(results_dir)

    all_meta = []
    train_bench = data_prep.get_benchmark_df(Split.TRAIN)
    trade_bench = data_prep.get_benchmark_df(Split.TRADE)

    for idx, cfg in enumerate(configs, 1):
        model_name = cfg["model_name"]
        impact_cls = cfg["impact_model_class"]
        env_kwargs = cfg.get("env_kwargs", {})
        mk = cfg.get("model_kwargs")
        pk = cfg.get("policy_kwargs")
        initial_amount = env_kwargs.get("initial_amount", 1e9)

        impact_inst = impact_cls()
        base = f"poe_{model_name}_{str(impact_inst).replace(' ', '_')}_{idx}"
        log.info(f"[{idx}/{len(configs)}] {model_name.upper()} | {impact_inst}")

        train_env = PortfolioOptimizationImpactEnv(
            config=train_config, impact_model=impact_inst, **env_kwargs
        )
        vec = DummyVecEnv([lambda: train_env])
        agent = DRLAgent(env=vec)
        eff_mk = mk if mk else MODEL_KWARGS[model_name]
        mdl = agent.get_model(model_name, model_kwargs=eff_mk, policy_kwargs=pk, seed=42)

        n_days = len(train_config["date_list"])
        trained, es_train, es_test = train_with_epoch_evaluation(
            mdl, train_env, train_config, trade_config,
            train_bench, trade_bench,
            env_kwargs, impact_cls, num_epochs, n_days, base,
        )

        # In-sample
        res_is, trades_is = run_and_log_simulation(
            train_env, trained, train_config["date_list"], train_bench,
        )
        res_is.to_csv(f"{results_dir}/{base}_train.csv", index=False)
        trades_is.to_csv(f"{results_dir}/{base}_train_trades.csv", index=False)

        # OOS continued from training (picks up end-of-train portfolio state)
        portfolio_state = train_env.get_portfolio_state()
        test_env = PortfolioOptimizationImpactEnv(
            config=trade_config, impact_model=impact_inst,
            initial_portfolio_state=portfolio_state, **env_kwargs,
        )
        last_train_bench = res_is["benchmark_value"].iloc[-1]
        res_oos, trades_oos = run_and_log_simulation(
            test_env, trained, trade_config["date_list"], trade_bench,
            reset_impact_model=False, initial_benchmark_value=last_train_bench,
        )
        res_oos.to_csv(f"{results_dir}/{base}_test.csv", index=False)
        trades_oos.to_csv(f"{results_dir}/{base}_test_trades.csv", index=False)

        # OOS blank slate
        test_env_blank = PortfolioOptimizationImpactEnv(
            config=trade_config, impact_model=impact_cls(), **env_kwargs,
        )
        res_blank, trades_blank = run_and_log_simulation(
            test_env_blank, trained, trade_config["date_list"], trade_bench,
        )
        res_blank.to_csv(f"{results_dir}/{base}_test_blank.csv", index=False)
        trades_blank.to_csv(f"{results_dir}/{base}_test_blank_trades.csv", index=False)

        meta = {
            "drl_agent": model_name,
            "impact_model": str(impact_inst),
            "initial_capital": initial_amount,
            "run_type": cfg.get("run_type", "baseline"),
            "results_csv_train": f"{results_dir}/{base}_train.csv",
            "results_csv_test": f"{results_dir}/{base}_test.csv",
            "results_csv_test_blank": f"{results_dir}/{base}_test_blank.csv",
            "trades_csv_train": f"{results_dir}/{base}_train_trades.csv",
            "trades_csv_test": f"{results_dir}/{base}_test_trades.csv",
            "trades_csv_test_blank": f"{results_dir}/{base}_test_blank_trades.csv",
            "env_type": "poe",
            "model_kwargs": eff_mk, "policy_kwargs": pk,
            "epoch_stats_train": es_train,
            "epoch_stats_test_blank": es_test,
        }
        for ek in ("reward_scaling", "max_trade_volume_pct"):
            if ek in env_kwargs:
                meta[ek] = env_kwargs[ek]
        if mk:
            for k, v in mk.items():
                meta[k] = v
        if cfg.get("net_arch_label"):
            meta["net_arch"] = cfg["net_arch_label"]
        all_meta.append(meta)

    summary = f"{results_dir}/backtest_summary.json"
    with open(summary, "w") as f:
        json.dump({"benchmark_ticker": data_prep.benchmark_ticker, "backtests": all_meta}, f, indent=4, default=str)
    log.info(f"Saved to {summary}")
    return summary


def run_example():
    np.random.seed(42)

    start_date = "2010-01-01"
    end_date = "2026-01-01"
    num_epochs = 20

    data_prep = MarketDataPreparator(
        tickers=NAS_100_TICKER, start_date=start_date, end_date=end_date,
        tech_indicators=INDICATORS, train_ratio=0.9, benchmark_ticker="QQEW",
    )

    # ── Baseline configs (default params) ────────────────────────────
    baseline_configs = [
        {"model_name": "a2c", "impact_model_class": BaselineImpactModel,
         "env_kwargs": {"initial_amount": 1e9}, "run_type": "baseline"},
        {"model_name": "a2c", "impact_model_class": ACImpactModel,
         "env_kwargs": {"initial_amount": 1e9}, "run_type": "baseline"},
        {"model_name": "ppo", "impact_model_class": BaselineImpactModel,
         "env_kwargs": {"initial_amount": 1e9}, "run_type": "baseline"},
        {"model_name": "ppo", "impact_model_class": ACImpactModel,
         "env_kwargs": {"initial_amount": 1e9}, "run_type": "baseline"},
        {"model_name": "ddpg", "impact_model_class": BaselineImpactModel,
         "env_kwargs": {"initial_amount": 1e9}, "run_type": "baseline"},
        {"model_name": "ddpg", "impact_model_class": ACImpactModel,
         "env_kwargs": {"initial_amount": 1e9}, "run_type": "baseline"},
        {"model_name": "sac", "impact_model_class": BaselineImpactModel,
         "env_kwargs": {"initial_amount": 1e9}, "run_type": "baseline"},
        {"model_name": "sac", "impact_model_class": ACImpactModel,
         "env_kwargs": {"initial_amount": 1e9}, "run_type": "baseline"},
        {"model_name": "td3", "impact_model_class": BaselineImpactModel,
         "env_kwargs": {"initial_amount": 1e9}, "run_type": "baseline"},
        {"model_name": "td3", "impact_model_class": ACImpactModel,
         "env_kwargs": {"initial_amount": 1e9}, "run_type": "baseline"},
    ]

    # ── HPO-optimized configs (Optuna best params, both impact models) ──
    _hpo_agent_params = {
        "a2c": {  # OOS Sharpe=1.1703, trial #87
            "env_kwargs_extra": {"reward_scaling": 8.0107, "max_trade_volume_pct": 0.1934},
            "model_kwargs": {"learning_rate": 8.428e-4, "n_steps": 10,
                             "ent_coef": 1.961e-4, "gamma": 0.9535,
                             "gae_lambda": 0.8869},
            "net_arch": "large",
        },
        "ppo": {  # OOS Sharpe=1.2065, trial #89
            "env_kwargs_extra": {"reward_scaling": 0.1009, "max_trade_volume_pct": 0.1210},
            "model_kwargs": {"learning_rate": 2.111e-3, "n_steps": 1024,
                             "batch_size": 128, "ent_coef": 2.154e-4,
                             "gamma": 0.9550, "gae_lambda": 0.9933,
                             "clip_range": 0.2937, "n_epochs": 20},
            "net_arch": "large",
        },
        "ddpg": {  # OOS Sharpe=1.1046, trial #93
            "env_kwargs_extra": {"reward_scaling": 0.1623, "max_trade_volume_pct": 0.1400},
            "model_kwargs": {"learning_rate": 1.619e-5, "batch_size": 256,
                             "buffer_size": 10_000, "gamma": 0.9976,
                             "tau": 3.459e-3},
            "net_arch": "wide",
        },
        "sac": {  # OOS Sharpe=1.1716, trial #80
            "env_kwargs_extra": {"reward_scaling": 1.3005, "max_trade_volume_pct": 0.0971},
            "model_kwargs": {"learning_rate": 1.033e-3, "batch_size": 128,
                             "buffer_size": 50_000, "gamma": 0.9820,
                             "tau": 7.623e-3, "ent_coef": "auto"},
            "net_arch": "wide",
        },
        "td3": {  # OOS Sharpe=1.1451, trial #41
            "env_kwargs_extra": {"reward_scaling": 0.1197, "max_trade_volume_pct": 0.1440},
            "model_kwargs": {"learning_rate": 4.053e-5, "batch_size": 64,
                             "buffer_size": 50_000, "gamma": 0.9487,
                             "tau": 1.251e-2},
            "net_arch": "large",
        },
    }

    hpo_configs = []
    for agent_name, hp in _hpo_agent_params.items():
        for impact_cls in (BaselineImpactModel, ACImpactModel):
            hpo_configs.append({
                "model_name": agent_name,
                "impact_model_class": impact_cls,
                "env_kwargs": {"initial_amount": 1e9, **hp["env_kwargs_extra"]},
                "model_kwargs": hp["model_kwargs"],
                "policy_kwargs": {"net_arch": NET_ARCH[hp["net_arch"]]},
                "net_arch_label": hp["net_arch"],
                "run_type": "optimized",
            })

    configs = baseline_configs + hpo_configs

    summary = train_and_backtest(data_prep, configs, num_epochs=num_epochs)
    BacktestReportGenerator(summary).generate_report()
    log.info("POE backtests complete.")


if __name__ == "__main__":
    run_example()

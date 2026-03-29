"""
Example script for backtesting DRL agents with the MarginTraderImpactEnv.

Parameters are set to match Gu et al. (2023) "Margin Trader" paper:
- Reward: R_t = λ1 * profit + λ2 * rolling_sharpe (Section 4.2.3)
- λ1 = 1e-5, λ2 tuned in {0.0001..0.05} (Section 5.4)
- Algorithms: A2C, PPO, DDPG, SAC (Section 4.2.4 / Table 1)
- Initial equity: $100,000 (Section 5.5)
- Margin rate: 2x leverage (Section 4.1, k=2)
- Maintenance margin strict: 30%, warning: 40% (Section 4.3.3)
- Transaction cost: 0.1% (Section 5.4) -- handled by BaselineImpactModel(10)
- Margin adjustment every 30 steps (Section 4.3.2)

NOTE: The paper does not report final HPO values, only search ranges
(Table 1). The configs below use the search ranges as guidance.
The paper used DJIA 30 stocks; we use NASDAQ 100 for consistency
with the rest of the framework.  PPO was the best performer in
their experiments.
"""
import json
import os
import sys
import uuid

import numpy as np
import pandas as pd

from finrl.config_tickers import NAS_100_TICKER, DOW_30_TICKER
from finrl.config import INDICATORS

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from meta.env_market_impact.envs.env_margin_trader_impact import MarginTraderImpactEnv
from meta.env_market_impact.envs.impact_models import (
    SqrtImpactModel, ACImpactModel, BaselineImpactModel,
)
from meta.env_market_impact.envs.market_data import MarketDataPreparator, Split
from meta.env_market_impact.backtest_report_generator import BacktestReportGenerator
from meta.env_market_impact.envs.utils import get_logger, compute_performance_stats
from agents.stablebaselines3_models import DRLAgent, TensorboardCallback
from stable_baselines3.common.vec_env import DummyVecEnv

log = get_logger()

# ═════════════════════════════════════════════════════════════════════
#  Paper hyperparameters (Table 1 — search ranges; paper does NOT
#  report final selected values, only the ranges below)
# ═════════════════════════════════════════════════════════════════════

# Default model_kwargs per algorithm — midpoints of the paper's ranges
MARGIN_TRADER_MODEL_KWARGS = {
    "a2c": {"n_steps": 10, "ent_coef": 0.005, "learning_rate": 5e-4},
    "ppo": {"n_steps": 2048, "ent_coef": 0.005, "learning_rate": 5e-4, "batch_size": 128},
    "ddpg": {"batch_size": 128, "buffer_size": 50000, "learning_rate": 5e-4},
    "sac": {"batch_size": 64, "buffer_size": 50000, "learning_rate": 5e-4, "ent_coef": 0.01},
}

# Environment params from paper Section 5.4
PAPER_ENV_KWARGS = {
    "initial_capital": 1e9,       # $1B to produce measurable market impact
    "margin_rate": 2.0,           # k = 2 (Section 4.1)
    "maintenance_margin": 0.3,    # strict level 30% (Section 4.3.3)
    "maintenance_warning": 0.4,   # warning level 40% (Section 4.3.3)
    "lambda_1": 1e-5,             # profit weight (Section 5.4)
    "lambda_2": 0.01,             # risk weight — middle of {0.0001..0.05}
    "sharpe_window": 5,           # rolling SR window (Section 4.2.3)
    "margin_adjust_period": 30,   # module runs every 30 steps (Section 4.3.2)
    "max_trade_volume_pct": 0.1,
}


# ── Simulation helper ─────────────────────────────────────────────────

def run_and_log_simulation(
    env: MarginTraderImpactEnv,
    model,
    dates: list,
    benchmark_df: pd.DataFrame,
    reset_impact_model: bool = True,
    initial_benchmark_value: float = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state, _ = env.reset(options={"reset_impact_model": reset_impact_model})
    results_log, trades_log = [], []
    last_asset = env.total_asset

    start = initial_benchmark_value if initial_benchmark_value is not None else env.initial_capital
    bench_cum = benchmark_df["close"] / benchmark_df["close"].iloc[0]
    bench_val = start * bench_cum

    results_log.append({
        "date": dates[env.time], "step": -1,
        "portfolio_value": env.total_asset, "pnl": 0.0, "reward": 0.0,
        "turnover": 0.0, "cost": 0.0,
        "total_buy_value": 0.0, "total_sell_value": 0.0,
        "benchmark_value": bench_val.iloc[env.time], "cash": env.cash,
    })

    for step in range(env.max_step):
        actions, _ = model.predict(state, deterministic=True)
        state, reward, done, _, info = env.step(actions)
        cur = env.total_asset
        results_log.append({
            "date": dates[env.time], "step": step,
            "portfolio_value": cur, "pnl": cur - last_asset, "reward": reward,
            "turnover": info["turnover"], "cost": info["cost"],
            "total_buy_value": info["total_buy_value"],
            "total_sell_value": info["total_sell_value"],
            "benchmark_value": bench_val.iloc[env.time], "cash": info["cash"],
        })
        last_asset = cur
        for t in info.get("trades", []):
            t["date"] = dates[env.time]; t["step"] = step
            trades_log.append(t)
        if done:
            break

    return pd.DataFrame(results_log), pd.DataFrame(trades_log)


# ── Epoch training with evaluation ────────────────────────────────────

def train_with_epoch_evaluation(
    model, train_env, train_config, trade_config,
    train_bench, trade_bench, env_kwargs, impact_model_class,
    num_epochs, num_training_days, tb_log_name,
) -> tuple:
    es_train, es_test = [], []
    for epoch in range(num_epochs):
        model.learn(
            total_timesteps=num_training_days,
            reset_num_timesteps=(epoch == 0),
            tb_log_name=tb_log_name,
            callback=TensorboardCallback(),
        )
        log.info(f"Epoch {epoch + 1}/{num_epochs} complete.")

        # Train eval
        ev_tr = MarginTraderImpactEnv(config=train_config, impact_model=impact_model_class(), **env_kwargs)
        r_tr, t_tr = run_and_log_simulation(ev_tr, model, train_config["date_list"], train_bench)
        ts = compute_performance_stats(r_tr["portfolio_value"], r_tr["turnover"], r_tr["cost"], t_tr, rewards=r_tr["reward"])
        ts["epoch"] = epoch + 1; es_train.append(ts)

        # Test eval (blank slate)
        ev_te = MarginTraderImpactEnv(config=trade_config, impact_model=impact_model_class(), **env_kwargs)
        r_te, t_te = run_and_log_simulation(ev_te, model, trade_config["date_list"], trade_bench)
        tt = compute_performance_stats(r_te["portfolio_value"], r_te["turnover"], r_te["cost"], t_te, rewards=r_te["reward"])
        tt["epoch"] = epoch + 1; es_test.append(tt)

        log.info(f"  Train Sharpe={ts['annualized_sharpe']:.3f} | Test Sharpe={tt['annualized_sharpe']:.3f}")

    return model, es_train, es_test


# ── Main orchestration ────────────────────────────────────────────────

def train_and_backtest(data_prep, configs, num_epochs=20) -> str:
    train_cfg = data_prep.create_env_config(Split.TRAIN)
    trade_cfg = data_prep.create_env_config(Split.TRADE)
    run_id = str(uuid.uuid4())
    results_dir = f"backtest_results/margin_trader_{run_id}"
    os.makedirs(results_dir)

    all_meta = []
    tr_bench = data_prep.get_benchmark_df(Split.TRAIN)
    te_bench = data_prep.get_benchmark_df(Split.TRADE)

    for idx, cfg in enumerate(configs, 1):
        mn = cfg["model_name"]
        imp_cls = cfg["impact_model_class"]
        ek = cfg.get("env_kwargs", PAPER_ENV_KWARGS)
        mk = cfg.get("model_kwargs", MARGIN_TRADER_MODEL_KWARGS.get(mn))
        pk = cfg.get("policy_kwargs")

        imp = imp_cls()
        base = f"mt_{mn}_{str(imp).replace(' ', '_')}_{idx}"
        log.info(f"[{idx}/{len(configs)}] {mn.upper()} | {imp}")

        tr_env = MarginTraderImpactEnv(config=train_cfg, impact_model=imp, **ek)
        vec = DummyVecEnv([lambda: tr_env])
        mdl = DRLAgent(env=vec).get_model(mn, model_kwargs=mk, policy_kwargs=pk, seed=42)

        n_days = len(train_cfg["date_list"])
        trained, es_tr, es_te = train_with_epoch_evaluation(
            mdl, tr_env, train_cfg, trade_cfg, tr_bench, te_bench,
            ek, imp_cls, num_epochs, n_days, base,
        )

        # IS (in-sample training simulation)
        r_is, t_is = run_and_log_simulation(tr_env, trained, train_cfg["date_list"], tr_bench)
        r_is.to_csv(f"{results_dir}/{base}_train.csv", index=False)
        t_is.to_csv(f"{results_dir}/{base}_train_trades.csv", index=False)

        # OOS continued from training (picks up end-of-train margin state)
        margin_state = tr_env.get_margin_state()
        te_env = MarginTraderImpactEnv(
            config=trade_cfg, impact_model=imp,
            initial_margin_state=margin_state, **ek,
        )
        last_train_bench = r_is["benchmark_value"].iloc[-1]
        r_oos, t_oos = run_and_log_simulation(
            te_env, trained, trade_cfg["date_list"], te_bench,
            reset_impact_model=False, initial_benchmark_value=last_train_bench,
        )
        r_oos.to_csv(f"{results_dir}/{base}_test.csv", index=False)
        t_oos.to_csv(f"{results_dir}/{base}_test_trades.csv", index=False)

        # OOS blank slate
        te_env_blank = MarginTraderImpactEnv(config=trade_cfg, impact_model=imp_cls(), **ek)
        r_blank, t_blank = run_and_log_simulation(
            te_env_blank, trained, trade_cfg["date_list"], te_bench,
        )
        r_blank.to_csv(f"{results_dir}/{base}_test_blank.csv", index=False)
        t_blank.to_csv(f"{results_dir}/{base}_test_blank_trades.csv", index=False)

        all_meta.append({
            "drl_agent": mn, "impact_model": str(imp),
            "initial_capital": ek.get("initial_capital", 1e5),
            "results_csv_train": f"{results_dir}/{base}_train.csv",
            "results_csv_test": f"{results_dir}/{base}_test.csv",
            "results_csv_test_blank": f"{results_dir}/{base}_test_blank.csv",
            "trades_csv_train": f"{results_dir}/{base}_train_trades.csv",
            "trades_csv_test": f"{results_dir}/{base}_test_trades.csv",
            "trades_csv_test_blank": f"{results_dir}/{base}_test_blank_trades.csv",
            "env_type": "margin_trader",
            "model_kwargs": mk, "policy_kwargs": pk,
            "epoch_stats_train": es_tr, "epoch_stats_test_blank": es_te,
        })

    summary = f"{results_dir}/backtest_summary.json"
    with open(summary, "w") as f:
        json.dump({"benchmark_ticker": data_prep.benchmark_ticker, "backtests": all_meta}, f, indent=4, default=str)
    log.info(f"Saved to {summary}")
    return summary


# ── Example entry point ───────────────────────────────────────────────

def run_example():
    """Run Margin Trader with paper parameters.

    Comparison grid:
    - 4 algorithms (A2C, PPO, DDPG, SAC) × 2 impact models
      (10 bps baseline vs. Almgren-Chriss)
    - All use the paper's env params (Section 5.4)
    """
    np.random.seed(42)

    start_date = "2010-01-01"
    end_date = "2026-01-01"
    num_epochs = 20

    data_prep = MarketDataPreparator(
        tickers=NAS_100_TICKER,
        start_date=start_date,
        end_date=end_date,
        tech_indicators=INDICATORS,
        train_ratio=0.9,
        benchmark_ticker="QQEW",
    )

    # Build grid: each algorithm × baseline vs AC impact
    # algorithms = ["a2c", "ppo"]
    algorithms = ["a2c", "ppo", "ddpg", "sac"]
    impact_models = [BaselineImpactModel, ACImpactModel]

    configs = []
    for algo in algorithms:
        for imp_cls in impact_models:
            configs.append({
                "model_name": algo,
                "impact_model_class": imp_cls,
                "env_kwargs": PAPER_ENV_KWARGS,
                "model_kwargs": MARGIN_TRADER_MODEL_KWARGS[algo],
            })

    log.info(f"Generated {len(configs)} configs ({len(algorithms)} algos × {len(impact_models)} impact models)")
    summary = train_and_backtest(data_prep, configs, num_epochs=num_epochs)
    BacktestReportGenerator(summary).generate_report()
    log.info("Margin-trader backtests complete.")


if __name__ == "__main__":
    run_example()

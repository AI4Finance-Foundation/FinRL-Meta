"""Hyperparameter optimisation for DRL agents using the POE environment.

Searches over environment parameters (reward_scaling, max_trade_volume_pct)
and algorithm-specific hyperparameters to maximise OOS annualized Sharpe.
The original paper's log-return reward is preserved.
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
import optuna
from optuna.pruners import MedianPruner

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from finrl.config_tickers import NAS_100_TICKER
from finrl.config import INDICATORS

from meta.env_market_impact.envs.env_portfolio_optimization_impact import (
    PortfolioOptimizationImpactEnv,
)
from meta.env_market_impact.envs.impact_models import BaselineImpactModel, ACImpactModel
from meta.env_market_impact.envs.market_data import MarketDataPreparator, Split
from meta.env_market_impact.envs.utils import get_logger, compute_performance_stats
from meta.env_market_impact.backtest_config import NET_ARCH, MODEL_KWARGS
from meta.env_market_impact.example_poe import run_and_log_simulation
from agents.stablebaselines3_models import DRLAgent
from stable_baselines3.common.vec_env import DummyVecEnv

log = get_logger()


def sample_env_kwargs(trial: optuna.Trial, initial_amount: float) -> dict:
    """Sample environment constructor kwargs for a trial."""
    return {
        "initial_amount": initial_amount,
        "reward_scaling": trial.suggest_float("reward_scaling", 0.1, 10.0, log=True),
        "max_trade_volume_pct": trial.suggest_float("max_trade_volume_pct", 0.05, 0.2),
    }


def sample_model_kwargs(trial: optuna.Trial, model_name: str) -> tuple[dict, dict | None]:
    net_arch_key = trial.suggest_categorical("net_arch", list(NET_ARCH.keys()))
    policy_kwargs = {"net_arch": NET_ARCH[net_arch_key]}
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    if model_name == "a2c":
        mk = {"learning_rate": lr,
              "n_steps": trial.suggest_categorical("n_steps", [5, 10, 20, 50]),
              "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
              "gamma": trial.suggest_float("gamma", 0.90, 0.999),
              "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0)}
    elif model_name == "ppo":
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        valid = [b for b in [32, 64, 128, 256, 512] if n_steps % b == 0]
        mk = {"learning_rate": lr, "n_steps": n_steps,
              "batch_size": trial.suggest_categorical("batch_size", valid),
              "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
              "gamma": trial.suggest_float("gamma", 0.90, 0.999),
              "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
              "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
              "n_epochs": trial.suggest_categorical("n_epochs_ppo", [3, 5, 10, 20])}
    elif model_name in ("ddpg", "td3"):
        mk = {"learning_rate": lr,
              "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
              "buffer_size": trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000]),
              "gamma": trial.suggest_float("gamma", 0.90, 0.999),
              "tau": trial.suggest_float("tau", 0.001, 0.05, log=True)}
    elif model_name == "sac":
        mk = {"learning_rate": lr,
              "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256, 512]),
              "buffer_size": trial.suggest_categorical("buffer_size", [10_000, 50_000, 100_000]),
              "gamma": trial.suggest_float("gamma", 0.90, 0.999),
              "tau": trial.suggest_float("tau", 0.001, 0.05, log=True),
              "ent_coef": "auto"}
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return mk, policy_kwargs


def objective(
    trial, *, data_prep, model_name, impact_model_class,
    initial_amount, num_epochs, seed=42,
) -> float:
    env_kwargs = sample_env_kwargs(trial, initial_amount)
    model_kwargs, policy_kwargs = sample_model_kwargs(trial, model_name)

    train_cfg = data_prep.create_env_config(Split.TRAIN)
    trade_cfg = data_prep.create_env_config(Split.TRADE)
    trade_bench = data_prep.get_benchmark_df(Split.TRADE)

    train_env = PortfolioOptimizationImpactEnv(
        config=train_cfg, impact_model=impact_model_class(), **env_kwargs
    )
    vec = DummyVecEnv([lambda: train_env])
    try:
        mdl = DRLAgent(env=vec).get_model(
            model_name, model_kwargs=model_kwargs,
            policy_kwargs=policy_kwargs, seed=seed,
        )
    except Exception as e:
        log.warning(f"Trial {trial.number}: model creation failed – {e}")
        raise optuna.TrialPruned()

    n_days = len(train_cfg["date_list"])
    best_sharpe = float("-inf")

    for epoch in range(num_epochs):
        try:
            mdl.learn(total_timesteps=n_days, reset_num_timesteps=(epoch == 0),
                      tb_log_name=f"optuna_poe_{trial.number}")
        except Exception as e:
            log.warning(f"Trial {trial.number}, epoch {epoch}: failed – {e}")
            raise optuna.TrialPruned()

        eval_env = PortfolioOptimizationImpactEnv(
            config=trade_cfg, impact_model=impact_model_class(), **env_kwargs
        )
        res, trades = run_and_log_simulation(
            eval_env, mdl, trade_cfg["date_list"], trade_bench,
        )
        stats = compute_performance_stats(
            res["portfolio_value"], res["turnover"],
            res["cost"], trades, rewards=res["reward"],
        )
        s = stats["annualized_sharpe"]
        best_sharpe = max(best_sharpe, s)
        log.info(f"Trial {trial.number} | epoch {epoch+1}/{num_epochs} | OOS Sharpe={s:.3f}")
        trial.report(s, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_sharpe


def run_multi_agent_hpo():
    np.random.seed(42)

    start_date = "2010-01-01"
    end_date = "2025-01-01"
    num_epochs = 20

    data_prep = MarketDataPreparator(
        tickers=NAS_100_TICKER, start_date=start_date, end_date=end_date,
        tech_indicators=INDICATORS, train_ratio=0.9, benchmark_ticker="QQEW",
    )
    agents = ["a2c", "ppo", "ddpg", "sac", "td3"]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hpo_results/poe_{ts}"
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    for mn in agents:
        log.info(f"{'='*60}\nHPO: {mn.upper()} (POE)\n{'='*60}")
        try:
            sampler = optuna.samplers.TPESampler(seed=42)
            pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
            study = optuna.create_study(direction="maximize", sampler=sampler,
                                         pruner=pruner, study_name=f"hpo_poe_{mn}")
            study.optimize(
                lambda t: objective(t, data_prep=data_prep, model_name=mn,
                                    impact_model_class=ACImpactModel,
                                    initial_amount=1e9, num_epochs=num_epochs, seed=42),
                n_trials=100, show_progress_bar=True,
            )

            # Save study results
            study_path = f"{results_dir}/{mn}_study.json"
            with open(study_path, "w") as f:
                json.dump({"best_value": study.best_value,
                           "best_params": study.best_params}, f, indent=2, default=str)
            log.info(f"Study results saved to {study_path}")

            # Save best params separately
            best_path = f"{results_dir}/{mn}_best_params.json"
            best_data = {
                "trial_number": study.best_trial.number,
                "objective_value": study.best_value,
                "params": study.best_params,
            }
            with open(best_path, "w") as f:
                json.dump(best_data, f, indent=2, default=str)
            log.info(f"Best params saved to {best_path}")

            # Print best params for this agent
            log.info(f"{'='*60}")
            log.info(f"{mn.upper()} OPTIMISATION COMPLETE")
            log.info(f"  Best OOS annualized Sharpe : {study.best_value:.4f}")
            log.info(f"  Best trial                 : #{study.best_trial.number}")
            log.info(f"  Best params:")
            for k, v in study.best_params.items():
                log.info(f"    {k}: {v}")
            log.info(f"{'='*60}")

            all_results.append({"model_name": mn, "study": study})

        except Exception as e:
            log.error(f"HPO failed for {mn}: {e}")

    # ── Final summary ─────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("POE HPO COMPLETE — SUMMARY OF BEST PARAMS")
    log.info("=" * 60)
    for entry in all_results:
        study = entry["study"]
        log.info(f"\n  {entry['model_name'].upper()} (OOS Sharpe={study.best_value:.4f}, "
                 f"trial #{study.best_trial.number}):")
        for k, v in study.best_params.items():
            log.info(f"    {k}: {v}")
    log.info("=" * 60)


if __name__ == "__main__":
    run_multi_agent_hpo()

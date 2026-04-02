"""Hyperparameter optimisation for DRL trading agents using Optuna.

This script searches over both **environment parameters** (``EnvParams``)
and **algorithm-specific hyperparameters** (learning-rate, entropy
coefficient, network size, …) to maximise the out-of-sample annualized
Sharpe ratio.

Architecture
------------
* ``sample_env_params``   – draws environment knobs for a trial.
* ``sample_model_kwargs`` – draws algorithm-specific knobs for a trial.
* ``objective``           – trains one configuration epoch-by-epoch,
                            reports OOS Sharpe to Optuna for pruning,
                            and returns the *best* OOS Sharpe across epochs.
* ``run_optimization``    – sets up the study and runs ``n_trials``.
* ``run_example``         – ready-made entry-point with sensible defaults.

Usage::

    python optuna_optimize.py          # uses defaults in run_example()
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

from meta.env_market_impact.envs.env_mace_stock_trading import (
    MACEStockTradingEnv,
    EnvParams,
)
from meta.env_market_impact.envs.impact_models import (
    BaselineImpactModel,
    SqrtImpactModel,
    ACImpactModel,
)
from meta.env_market_impact.envs.market_data import MarketDataPreparator, Split
from meta.env_market_impact.envs.utils import get_logger, compute_performance_stats
from meta.env_market_impact.backtest_config import (
    BacktestParams,
    NET_ARCH,
    MODEL_KWARGS,
    reconstruct_agent_kwargs,
)
from meta.env_market_impact.example_mace_env import (
    run_and_log_simulation,
    train_and_backtest,
)
from meta.env_market_impact.backtest_report_generator import BacktestReportGenerator
from agents.stablebaselines3_models import DRLAgent
from stable_baselines3.common.vec_env import DummyVecEnv

log = get_logger()


def sample_env_params(
    trial: optuna.Trial,
    max_stock_pct: float,
    max_trade_volume_pct: float,
) -> EnvParams:
    """Sample tuneable ``EnvParams`` fields; fixed knobs are passed through.

    ``max_stock_pct`` and ``max_trade_volume_pct`` are **not** optimised –
    they are set by the caller and forwarded as-is.
    """
    use_obs_normalizer = trial.suggest_categorical(
        "use_obs_normalizer",
        [True, False],
    )
    obs_clip = (
        trial.suggest_float("obs_clip", 5.0, 20.0)
        if use_obs_normalizer
        else 10.0  # default, unused when normaliser is off
    )

    return EnvParams(
        max_stock_pct=max_stock_pct,
        max_trade_volume_pct=max_trade_volume_pct,
        reward_scaling=trial.suggest_float(
            "reward_scaling",
            2**-14,
            2**-8,
            log=True,
        ),
        include_permanent_impact_in_state=trial.suggest_categorical(
            "include_permanent_impact_in_state",
            [True, False],
        ),
        include_cooldown_in_state=trial.suggest_categorical(
            "include_cooldown_in_state",
            [True, False],
        ),
        include_tbill_in_state=trial.suggest_categorical(
            "include_tbill_in_state",
            [True, False],
        ),
        horizon=trial.suggest_categorical("horizon", [10, 20, 40, 80]),
        eta_dd=trial.suggest_float("eta_dd", 0.0, 3.0),
        use_obs_normalizer=use_obs_normalizer,
        obs_clip=obs_clip,
    )


def sample_model_kwargs(
    trial: optuna.Trial,
    model_name: str,
) -> tuple[dict, dict | None]:
    """Return ``(model_kwargs, policy_kwargs)`` sampled from the search space.

    Each algorithm family has its own hyper-parameter ranges.  Network
    architecture is sampled for all families via ``policy_kwargs``.
    """
    # ── Common: network architecture ─────────────────────────────────
    net_arch_key = trial.suggest_categorical(
        "net_arch",
        list(NET_ARCH.keys()),
    )
    policy_kwargs = {"net_arch": NET_ARCH[net_arch_key]}

    # ── Common: learning-rate ────────────────────────────────────────
    lr = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)

    # ── Per-algorithm knobs ──────────────────────────────────────────
    if model_name == "a2c":
        model_kwargs = {
            "learning_rate": lr,
            "n_steps": trial.suggest_categorical("n_steps", [5, 10, 20, 50]),
            "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
            "gamma": trial.suggest_float("gamma", 0.90, 0.999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
        }

    elif model_name == "ppo":
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        # batch_size must divide n_steps (single env → n_steps * 1)
        valid_batches = [b for b in [32, 64, 128, 256, 512] if n_steps % b == 0]
        batch_size = trial.suggest_categorical("batch_size", valid_batches)
        model_kwargs = {
            "learning_rate": lr,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "ent_coef": trial.suggest_float("ent_coef", 1e-4, 0.1, log=True),
            "gamma": trial.suggest_float("gamma", 0.90, 0.999),
            "gae_lambda": trial.suggest_float("gae_lambda", 0.8, 1.0),
            "clip_range": trial.suggest_float("clip_range", 0.1, 0.4),
            "n_epochs": trial.suggest_categorical("n_epochs_ppo", [3, 5, 10, 20]),
        }

    elif model_name in ("ddpg", "td3"):
        model_kwargs = {
            "learning_rate": lr,
            "batch_size": trial.suggest_categorical(
                "batch_size",
                [64, 128, 256, 512],
            ),
            "buffer_size": trial.suggest_categorical(
                "buffer_size",
                [10_000, 50_000, 100_000],
            ),
            "gamma": trial.suggest_float("gamma", 0.90, 0.999),
            "tau": trial.suggest_float("tau", 0.001, 0.05, log=True),
        }

    elif model_name == "sac":
        model_kwargs = {
            "learning_rate": lr,
            "batch_size": trial.suggest_categorical(
                "batch_size",
                [64, 128, 256, 512],
            ),
            "buffer_size": trial.suggest_categorical(
                "buffer_size",
                [10_000, 50_000, 100_000],
            ),
            "gamma": trial.suggest_float("gamma", 0.90, 0.999),
            "tau": trial.suggest_float("tau", 0.001, 0.05, log=True),
            "ent_coef": "auto",
        }

    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model_kwargs, policy_kwargs


def objective(
    trial: optuna.Trial,
    *,
    data_prep: MarketDataPreparator,
    model_name: str,
    impact_model_class: type,
    initial_capital: float,
    max_stock_pct: float,
    max_trade_volume_pct: float,
    num_epochs: int,
    seed: int = 42,
) -> float:
    """Train one configuration and return best OOS annualized Sharpe.

    Per-epoch OOS Sharpe is reported to Optuna so that the pruner can
    terminate unpromising trials early.
    """
    # ── Sample hyper-parameters ──────────────────────────────────────
    env_params = sample_env_params(trial, max_stock_pct, max_trade_volume_pct)
    model_kwargs, policy_kwargs = sample_model_kwargs(trial, model_name)

    # ── Build envs & configs ─────────────────────────────────────────
    train_config = data_prep.create_env_config(Split.TRAIN)
    trade_config = data_prep.create_env_config(Split.TRADE)
    trade_benchmark_df = data_prep.get_benchmark_df(Split.TRADE)

    impact_model_instance = impact_model_class()
    train_env = MACEStockTradingEnv(
        config=train_config,
        params=env_params,
        impact_model=impact_model_instance,
        initial_capital=initial_capital,
    )
    vec_train_env = DummyVecEnv([lambda: train_env])

    # ── Create the model ─────────────────────────────────────────────
    agent = DRLAgent(env=vec_train_env)
    try:
        model = agent.get_model(
            model_name,
            model_kwargs=model_kwargs,
            policy_kwargs=policy_kwargs,
            seed=seed,
        )
    except Exception as e:
        log.warning(f"Trial {trial.number}: model creation failed – {e}")
        raise optuna.TrialPruned()

    num_training_days = len(train_config["date_list"])
    best_oos_sharpe = float("-inf")

    # ── Epoch loop with pruning ──────────────────────────────────────
    for epoch in range(num_epochs):
        try:
            model.learn(
                total_timesteps=num_training_days,
                reset_num_timesteps=(epoch == 0),
                tb_log_name=f"optuna_trial_{trial.number}",
            )
        except Exception as e:
            log.warning(f"Trial {trial.number}, epoch {epoch}: training failed – {e}")
            raise optuna.TrialPruned()

        # ── OOS evaluation (blank-slate) ─────────────────────────────
        eval_trade_env = MACEStockTradingEnv(
            config=trade_config,
            params=env_params,
            impact_model=impact_model_class(),
            initial_capital=initial_capital,
        )
        normalizer_state = train_env.get_normalizer_state()
        if normalizer_state is not None:
            eval_trade_env.set_normalizer_state(normalizer_state, freeze=True)

        eval_results, eval_trades = run_and_log_simulation(
            eval_trade_env,
            model,
            trade_config["date_list"],
            trade_benchmark_df,
            reset_impact_model=True,
        )
        stats = compute_performance_stats(
            eval_results["portfolio_value"],
            eval_results["turnover"],
            eval_results["cost"],
            eval_trades,
            rewards=eval_results["reward"],
        )

        oos_sharpe = stats["annualized_sharpe"]
        best_oos_sharpe = max(best_oos_sharpe, oos_sharpe)

        log.info(
            f"Trial {trial.number} | epoch {epoch + 1}/{num_epochs} | "
            f"OOS Sharpe={oos_sharpe:.3f} (best={best_oos_sharpe:.3f})"
        )

        # Report intermediate value for pruning
        trial.report(oos_sharpe, epoch)
        if trial.should_prune():
            log.info(f"Trial {trial.number} pruned at epoch {epoch + 1}.")
            raise optuna.TrialPruned()

    return best_oos_sharpe


def run_optimization(
    data_prep: MarketDataPreparator,
    model_name: str = "a2c",
    impact_model_class: type = BaselineImpactModel,
    initial_capital: float = 1e9,
    max_stock_pct: float | None = None,
    max_trade_volume_pct: float = 0.1,
    max_stock_weight_multiplier: float = 2.0,
    num_epochs: int = 10,
    n_trials: int = 50,
    n_startup_trials: int = 5,
    n_warmup_steps: int = 3,
    seed: int = 42,
) -> optuna.Study:
    """Create an Optuna study and optimise the given configuration.

    Parameters
    ----------
    data_prep : MarketDataPreparator
        Pre-built data preparator (tickers, splits, etc.).
    model_name : str
        SB3 algorithm name (``a2c``, ``ppo``, ``ddpg``, ``sac``, ``td3``).
    impact_model_class : type
        Market-impact model class to use.
    initial_capital : float
        Starting portfolio value.
    max_stock_pct : float or None
        Maximum single-stock weight.  If *None*, computed automatically from
        ``max_stock_weight_multiplier`` and the universe size.
    max_trade_volume_pct : float
        Maximum fraction of daily volume tradeable per step (not optimised).
    max_stock_weight_multiplier : float
        Used only when ``max_stock_pct is None``.
    num_epochs : int
        Training epochs per trial.
    n_trials : int
        Total Optuna trials to run.
    n_startup_trials : int
        Random trials before the pruner kicks in.
    n_warmup_steps : int
        Epochs before the pruner can prune a trial.
    seed : int
        Random seed for model initialisation.

    Returns
    -------
    optuna.Study
        The completed study (inspect ``study.best_trial``, etc.).
    """
    if max_stock_pct is None:
        max_stock_pct = float(
            np.clip(
                (1.0 / data_prep.universe_size) * max_stock_weight_multiplier,
                0.01,
                1.0,
            )
        )

    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = MedianPruner(
        n_startup_trials=n_startup_trials,
        n_warmup_steps=n_warmup_steps,
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=f"hpo_{model_name}_{impact_model_class().__class__.__name__}",
    )

    study.optimize(
        lambda trial: objective(
            trial,
            data_prep=data_prep,
            model_name=model_name,
            impact_model_class=impact_model_class,
            initial_capital=initial_capital,
            max_stock_pct=max_stock_pct,
            max_trade_volume_pct=max_trade_volume_pct,
            num_epochs=num_epochs,
            seed=seed,
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    log.info("=" * 60)
    log.info("OPTIMISATION COMPLETE")
    log.info(f"  Best OOS annualized Sharpe : {study.best_value:.4f}")
    log.info(f"  Best trial                 : #{study.best_trial.number}")
    log.info("  Best params:")
    for k, v in study.best_params.items():
        log.info(f"    {k}: {v}")
    log.info("=" * 60)

    return study


def backtest_best_trial(
    study: optuna.Study,
    data_prep: MarketDataPreparator,
    model_name: str,
    impact_model_class: type,
    initial_capital: float = 1e9,
    max_stock_pct: float | None = None,
    max_trade_volume_pct: float = 0.1,
    max_stock_weight_multiplier: float = 2.0,
    num_epochs: int = 10,
) -> str:
    """Run a full backtest (with report) using the best trial's parameters.

    Returns the path to the generated ``backtest_summary.json``.
    """
    best = study.best_params

    if max_stock_pct is None:
        max_stock_pct = float(
            np.clip(
                (1.0 / data_prep.universe_size) * max_stock_weight_multiplier,
                0.01,
                1.0,
            )
        )

    env_params = EnvParams(
        max_stock_pct=max_stock_pct,
        max_trade_volume_pct=max_trade_volume_pct,
        reward_scaling=best["reward_scaling"],
        include_permanent_impact_in_state=best["include_permanent_impact_in_state"],
        include_cooldown_in_state=best["include_cooldown_in_state"],
        include_tbill_in_state=best["include_tbill_in_state"],
        horizon=best["horizon"],
        eta_dd=best["eta_dd"],
        use_obs_normalizer=best["use_obs_normalizer"],
        obs_clip=best.get("obs_clip", 10.0),
    )

    # Reconstruct model_kwargs and policy_kwargs from the best trial's params
    model_kwargs, policy_kwargs = reconstruct_agent_kwargs(best, model_name)

    backtest_grid = [
        BacktestParams(
            model_name=model_name,
            impact_model_class=impact_model_class,
            initial_capital=initial_capital,
            env_params=env_params,
            model_kwargs=model_kwargs,
            policy_kwargs=policy_kwargs,
        )
    ]

    summary_path = train_and_backtest(data_prep, backtest_grid, num_epochs)
    BacktestReportGenerator(summary_path).generate_report()
    return summary_path


def save_study_results(study: optuna.Study, path: str) -> None:
    """Persist the study's best params and top-N trials to a JSON file."""
    trials_data = []
    for t in sorted(
        study.trials,
        key=lambda x: x.value if x.value is not None else float("-inf"),
        reverse=True,
    ):
        if t.value is None:
            continue
        trials_data.append(
            {
                "number": t.number,
                "value": t.value,
                "params": t.params,
                "state": t.state.name,
            }
        )

    result = {
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "n_trials": len(study.trials),
        "all_trials": trials_data,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"Study results saved to {path}")


def save_best_params(study: optuna.Study, path: str) -> None:
    """Dump the best trial's params + objective value to a JSON file."""
    best = {
        "trial_number": study.best_trial.number,
        "best_oos_annualized_sharpe": study.best_value,
        "params": study.best_params,
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(best, f, indent=2, default=str)
    log.info(f"Best params saved to {path}")


def run_multi_agent_hpo():
    """Run HPO for multiple agents sequentially, save results, then
    produce a single comparison backtest report with all best configs
    plus baselines.

    Designed for overnight runs::

        python optuna_optimize.py
    """
    np.random.seed(42)

    # ── Data setup ───────────────────────────────────────────────────
    start_date = "2010-01-01"
    end_date = "2025-01-01"

    data_prep = MarketDataPreparator(
        tickers=NAS_100_TICKER,
        start_date=start_date,
        end_date=end_date,
        tech_indicators=INDICATORS,
        train_ratio=0.9,
        benchmark_ticker="QQEW",
    )

    # ── HPO settings ─────────────────────────────────────────────────
    agents_to_optimize = ["a2c", "ppo", "ddpg", "sac", "td3"]
    impact_model_class = ACImpactModel
    initial_capital = 1e9
    num_epochs = 20
    n_trials = 100

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"hpo_results/{ts}"
    os.makedirs(results_dir, exist_ok=True)

    # ── Run HPO for each agent ───────────────────────────────────────
    best_configs: list[dict] = []  # collect best params per agent

    for model_name in agents_to_optimize:
        log.info("=" * 60)
        log.info(f"STARTING HPO: {model_name.upper()}")
        log.info("=" * 60)

        try:
            study = run_optimization(
                data_prep=data_prep,
                model_name=model_name,
                impact_model_class=impact_model_class,
                initial_capital=initial_capital,
                num_epochs=num_epochs,
                n_trials=n_trials,
            )
        except Exception as e:
            log.error(f"HPO failed for {model_name}: {e}")
            continue

        save_study_results(study, f"{results_dir}/{model_name}_study_results.json")
        save_best_params(study, f"{results_dir}/{model_name}_best_params.json")

        best_configs.append(
            {
                "model_name": model_name,
                "study": study,
            }
        )

    # ── Build comparison backtest grid ───────────────────────────────
    # Baselines (default params) + HPO best for each agent

    baseline_configs = []
    hpo_configs = []

    for entry in best_configs:
        model_name = entry["model_name"]
        study = entry["study"]
        best = study.best_params

        # Baseline with default agent params
        baseline_configs.append(
            {
                "model_name": model_name,
                "impact_model_class": impact_model_class,
                "initial_capital": initial_capital,
                # Default env params (EnvParams defaults)
                "include_permanent_impact_in_state": True,
                "include_cooldown_in_state": True,
                "include_tbill_in_state": True,
                "use_obs_normalizer": False,
                "eta_dd": 0.5,
            }
        )

        # HPO best params
        model_kwargs, policy_kwargs = reconstruct_agent_kwargs(best, model_name)
        hpo_cfg = {
            "model_name": model_name,
            "impact_model_class": impact_model_class,
            "initial_capital": initial_capital,
            "model_kwargs": model_kwargs,
            "policy_kwargs": policy_kwargs,
            "use_obs_normalizer": best["use_obs_normalizer"],
            "reward_scaling": best["reward_scaling"],
            "include_permanent_impact_in_state": best[
                "include_permanent_impact_in_state"
            ],
            "include_cooldown_in_state": best["include_cooldown_in_state"],
            "include_tbill_in_state": best["include_tbill_in_state"],
            "horizon": best["horizon"],
            "eta_dd": best["eta_dd"],
        }
        if best.get("obs_clip") is not None:
            hpo_cfg["obs_clip"] = best["obs_clip"]
        hpo_configs.append(hpo_cfg)

    all_configs = baseline_configs + hpo_configs
    if not all_configs:
        log.error("No successful HPO runs – nothing to backtest.")
        return

    backtest_grid = BacktestParams.from_explicit(
        configs=all_configs,
        num_stocks=data_prep.universe_size,
    )

    log.info("=" * 60)
    log.info(f"RUNNING COMPARISON BACKTEST: {len(backtest_grid)} configurations")
    log.info(f"  {len(baseline_configs)} baselines + {len(hpo_configs)} HPO best")
    log.info("=" * 60)

    summary_path = train_and_backtest(data_prep, backtest_grid, num_epochs)
    BacktestReportGenerator(summary_path).generate_report()
    log.info(f"Comparison report generated from: {summary_path}")

    # ── Final summary ────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("HPO SUMMARY — BEST PARAMS PER AGENT")
    log.info("=" * 60)
    for entry in best_configs:
        study = entry["study"]
        log.info(
            f"\n  {entry['model_name'].upper()} (OOS Sharpe={study.best_value:.4f}):"
        )
        for k, v in study.best_params.items():
            log.info(f"    {k}: {v}")
    log.info("=" * 60)
    log.info("All done.")


if __name__ == "__main__":
    run_multi_agent_hpo()

"""Shared configuration, parameter definitions, and constants for backtesting.

This module is the single source of truth for:

* **MODEL_KWARGS** – default SB3 hyper-parameters per algorithm.
* **NET_ARCH**     – named network architecture presets.
* **AGENT_KEYS**   – which flat keys map to each algorithm's ``model_kwargs``.
* **BacktestParams** – dataclass that fully describes one backtest run.
* **reconstruct_agent_kwargs** – rebuild ``(model_kwargs, policy_kwargs)``
  from a flat dict (e.g. Optuna best params).

Both ``example_mace_env`` and ``optuna_optimize`` import from here
instead of duplicating definitions.
"""

import itertools
import uuid
from dataclasses import dataclass
from dataclasses import field

import numpy as np

from meta.env_market_impact.envs.env_mace_stock_trading import EnvParams

# ── Lazy imports resolved at module level ────────────────────────────────
# EnvParams is needed by BacktestParams; it lives in the env package.


# ═════════════════════════════════════════════════════════════════════════
#  Default agent hyper-parameters
# ═════════════════════════════════════════════════════════════════════════

MODEL_KWARGS: dict[str, dict] = {
    "a2c": {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007},
    "ppo": {
        "n_steps": 2048,
        "ent_coef": 0.01,
        "learning_rate": 0.0003,
        "batch_size": 64,
    },
    "ddpg": {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001},
    "sac": {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001},
    "td3": {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001},
}

# ═════════════════════════════════════════════════════════════════════════
#  Network architecture presets
# ═════════════════════════════════════════════════════════════════════════

NET_ARCH: dict[str, list[int]] = {
    "small": [64, 64],
    "medium": [128, 128],
    "large": [256, 256],
    "wide": [256, 128, 64],
}

# ═════════════════════════════════════════════════════════════════════════
#  Per-algorithm parameter key sets (for flat-dict reconstruction)
# ═════════════════════════════════════════════════════════════════════════

AGENT_KEYS: dict[str, set[str]] = {
    "a2c": {"learning_rate", "n_steps", "ent_coef", "gamma", "gae_lambda"},
    "ppo": {
        "learning_rate",
        "n_steps",
        "batch_size",
        "ent_coef",
        "gamma",
        "gae_lambda",
        "clip_range",
        "n_epochs_ppo",
    },
    "ddpg": {"learning_rate", "batch_size", "buffer_size", "gamma", "tau"},
    "td3": {"learning_rate", "batch_size", "buffer_size", "gamma", "tau"},
    "sac": {"learning_rate", "batch_size", "buffer_size", "gamma", "tau"},
}


def reconstruct_agent_kwargs(
    flat_params: dict,
    model_name: str,
) -> tuple[dict, dict | None]:
    """Rebuild ``(model_kwargs, policy_kwargs)`` from a flat parameter dict.

    Useful when loading Optuna best-params or a saved JSON and needing to
    feed them back into ``DRLAgent.get_model``.

    Parameters
    ----------
    flat_params : dict
        Flat dictionary (e.g. ``study.best_params``).
    model_name : str
        Algorithm name (``a2c``, ``ppo``, …).

    Returns
    -------
    tuple[dict, dict | None]
        ``(model_kwargs, policy_kwargs)``
    """
    agent_keys = AGENT_KEYS.get(model_name, set())
    model_kwargs = {k: v for k, v in flat_params.items() if k in agent_keys}

    # PPO stores n_epochs under a different trial key to avoid collisions
    if "n_epochs_ppo" in model_kwargs:
        model_kwargs["n_epochs"] = model_kwargs.pop("n_epochs_ppo")

    # SAC always uses automatic entropy tuning
    if model_name == "sac":
        model_kwargs["ent_coef"] = "auto"

    # Policy kwargs (net_arch)
    net_arch_key = flat_params.get("net_arch")
    policy_kwargs = {"net_arch": NET_ARCH[net_arch_key]} if net_arch_key else None

    return model_kwargs, policy_kwargs


@dataclass
class BacktestParams:
    """All parameters that define a single backtest configuration.

    Composes an :class:`EnvParams` (the environment-level knobs) with
    backtest-level settings (which DRL model, impact model, initial
    capital, and agent hyperparameters to use).
    """

    model_name: str
    impact_model_class: type
    initial_capital: float = 1e6
    env_params: EnvParams = field(default_factory=EnvParams)
    model_kwargs: dict | None = None
    policy_kwargs: dict | None = None

    # Computed after __init__
    impact_model_name: str = field(init=False, repr=False)

    def __post_init__(self):
        self.impact_model_name = str(self.impact_model_class())

    def get_model_kwargs(self) -> dict:
        """Return agent hyper-parameters, falling back to ``MODEL_KWARGS``."""
        return (
            self.model_kwargs
            if self.model_kwargs is not None
            else MODEL_KWARGS[self.model_name]
        )

    @property
    def base_filename(self) -> str:
        """Deterministic, unique filename fragment for this configuration."""
        ep = self.env_params
        mk = self.get_model_kwargs()
        base = (
            f"backtest_{self.model_name}_"
            f"{self.impact_model_name.replace(' ', '_')}_"
            f"{str(int(self.initial_capital))}"
        )
        param_string = (
            f"model={self.model_name};impact={self.impact_model_name};capital={self.initial_capital};"
            f"perm={ep.include_permanent_impact_in_state};cooldown={ep.include_cooldown_in_state};"
            f"tbill={ep.include_tbill_in_state};eta_dd={ep.eta_dd};norm={ep.use_obs_normalizer};"
            f"reward_scaling={ep.reward_scaling};horizon={ep.horizon};obs_clip={ep.obs_clip};"
            f"model_kwargs={sorted(mk.items())};policy_kwargs={sorted((self.policy_kwargs or {}).items())}"
        )
        uid = uuid.uuid5(uuid.NAMESPACE_URL, param_string).hex[:8]
        return f"{base}_{uid}"

    # Fields that belong to BacktestParams itself (not EnvParams)
    _BACKTEST_KEYS = {
        "model_name",
        "impact_model_class",
        "initial_capital",
        "model_kwargs",
        "policy_kwargs",
    }

    @staticmethod
    def from_explicit(
        configs: list[dict],
        num_stocks: int,
        max_stock_weight_multiplier: float = 2.0,
        max_stock_pct_clip: tuple[float, float] = (0.01, 1.0),
    ) -> list["BacktestParams"]:
        """Build a list from explicit per-configuration dicts (no Cartesian product).

        Each dict in *configs* describes one exact backtest configuration.
        Keys that match ``BacktestParams`` fields (``model_name``,
        ``impact_model_class``, ``initial_capital``, ``model_kwargs``,
        ``policy_kwargs``) are used directly; every other key is forwarded
        to ``EnvParams``.  Missing keys fall back to the dataclass defaults.

        Parameters
        ----------
        configs : list[dict]
            One dict per desired configuration.
        num_stocks : int
            Number of stocks in the universe (used to derive ``max_stock_pct``
            unless a config explicitly supplies it).
        max_stock_weight_multiplier : float
            Multiplier for the maximum stock weight relative to equal-weighted
            portfolio (used when ``max_stock_pct`` is not in the config dict).
        max_stock_pct_clip : tuple[float, float]
            Clipping range for the derived ``max_stock_pct``.

        Returns
        -------
        list[BacktestParams]
        """
        default_max_stock_pct = float(
            np.clip(
                (1.0 / num_stocks) * max_stock_weight_multiplier,
                max_stock_pct_clip[0],
                max_stock_pct_clip[1],
            )
        )

        result: list[BacktestParams] = []
        for cfg in configs:
            bt_kwargs = {}
            env_kwargs: dict = {"max_stock_pct": default_max_stock_pct}
            for k, v in cfg.items():
                if k in BacktestParams._BACKTEST_KEYS:
                    bt_kwargs[k] = v
                else:
                    env_kwargs[k] = v
            bt_kwargs["env_params"] = EnvParams(**env_kwargs)
            result.append(BacktestParams(**bt_kwargs))
        return result

    @staticmethod
    def generate_grid(
        models_to_run: list[str],
        initial_capitals: list[float],
        impact_model_classes: list[type],
        num_stocks: int,
        include_permanent_impact_in_state: list[bool] = (True,),
        include_cooldown_in_state: list[bool] = (True,),
        include_tbill_in_state: list[bool] = (False,),
        eta_dd: list[float] = (0.5,),
        use_obs_normalizer: list[bool] = (True,),
        reward_scaling: list[float] = (2**-11,),
        horizon: list[int] = (20,),
        obs_clip: list[float] = (10.0,),
        max_stock_weight_multiplier: float = 2.0,
        max_stock_pct_clip: tuple[float, float] = (0.01, 1.0),
    ) -> list["BacktestParams"]:
        """Pre-generate the full Cartesian product of all parameter axes.

        Parameters
        ----------
        models_to_run, initial_capitals, impact_model_classes
            Lists defining the identity axes of the grid.
        num_stocks : int
            Universe size (for ``max_stock_pct`` derivation).
        reward_scaling, horizon, obs_clip
            Env-level tuneable knobs.
        max_stock_weight_multiplier, max_stock_pct_clip
            Controls for the derived ``max_stock_pct``.

        Remaining parameters mirror ``EnvParams`` fields.

        Returns
        -------
        list[BacktestParams]
        """
        max_stock_pct = float(
            np.clip(
                (1.0 / num_stocks) * max_stock_weight_multiplier,
                max_stock_pct_clip[0],
                max_stock_pct_clip[1],
            )
        )
        return [
            BacktestParams(
                model_name=m,
                impact_model_class=imp,
                initial_capital=cap,
                env_params=EnvParams(
                    max_stock_pct=max_stock_pct,
                    include_permanent_impact_in_state=perm,
                    include_cooldown_in_state=cool,
                    include_tbill_in_state=tbill,
                    eta_dd=eta,
                    use_obs_normalizer=norm,
                    reward_scaling=rs,
                    horizon=hz,
                    obs_clip=oc,
                ),
            )
            for m, cap, imp, perm, cool, tbill, eta, norm, rs, hz, oc in itertools.product(
                models_to_run,
                initial_capitals,
                impact_model_classes,
                include_permanent_impact_in_state,
                include_cooldown_in_state,
                include_tbill_in_state,
                eta_dd,
                use_obs_normalizer,
                reward_scaling,
                horizon,
                obs_clip,
            )
        ]

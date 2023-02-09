# DRL models from RLlib
import ray
from ray.rllib.agents.a3c import a2c
from ray.rllib.agents.ddpg import ddpg
from ray.rllib.agents.ddpg import td3
from ray.rllib.agents.ppo import ppo
from ray.rllib.agents.sac import sac

MODELS = {"a2c": a2c, "ddpg": ddpg, "td3": td3, "sac": sac, "ppo": ppo}

from ray.tune.registry import register_env


# MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}


class Rllib_model:
    def __init__(self, trainer):
        self.trainer = trainer

    def __call__(self, state):
        return self.trainer.compute_single_action(state)


class DRLAgent:
    """Implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class
    Methods
    -------
        get_model()
            setup DRL algorithms
        train_model()
            train DRL algorithms in a train dataset
            and output the trained model
        DRL_prediction()
            make a prediction in a test dataset and get results
    """

    def __init__(self, env, init_ray=True):
        self.env = env
        if init_ray:
            ray.init(
                ignore_reinit_error=True
            )  # Other Ray APIs will not work until `ray.init()` is called.

    def get_model(
        self,
        model_name,
        env_config,
        model_config=None,
        framework="torch",
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        model = MODELS[model_name]
        # get algorithm default configration based on algorithm in RLlib
        if model_config is None:
            if model_name == "a2c":
                model_config = model.A2C_DEFAULT_CONFIG.copy()
            elif model_name == "td3":
                model_config = model.TD3_DEFAULT_CONFIG.copy()
            else:
                model_config = model.DEFAULT_CONFIG.copy()

        register_env("finrl_env", self.env)
        model_config["env"] = "finrl_env"
        model_config["env_config"] = env_config
        model_config["log_level"] = "WARN"
        model_config["framework"] = framework

        return model, model_config

    def train_model(
        self,
        model,
        model_name,
        model_config,
        total_episodes=100,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_name == "ppo":
            trainer = model.PPOTrainer(env=self.env, config=model_config)
        elif model_name == "a2c":
            trainer = model.A2CTrainer(env=self.env, config=model_config)
        elif model_name == "ddpg":
            trainer = model.DDPGTrainer(env=self.env, config=model_config)
        elif model_name == "td3":
            trainer = model.TD3Trainer(env=self.env, config=model_config)
        elif model_name == "sac":
            trainer = model.SACTrainer(env=self.env, config=model_config)

        cwd = "./test_" + str(model_name)
        for _ in range(total_episodes):
            trainer.train()
            # save the trained model
            trainer.save(cwd)

        ray.shutdown()

        return trainer

    @staticmethod
    def DRL_prediction(
        model_name,
        env,
        env_config,
        agent_path="./test_ppo/checkpoint_000100/checkpoint-100",
        init_ray=True,
        model_config=None,
        framework="torch",
    ):
        if init_ray:
            ray.init(
                ignore_reinit_error=True
            )  # Other Ray APIs will not work until `ray.init()` is called.

        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_config is None:
            if model_name == "a2c":
                model_config = MODELS[model_name].A2C_DEFAULT_CONFIG.copy()
            elif model_name == "td3":
                model_config = MODELS[model_name].TD3_DEFAULT_CONFIG.copy()
            else:
                model_config = MODELS[model_name].DEFAULT_CONFIG.copy()

        register_env("finrl_env", env)
        model_config["env"] = "finrl_env"
        model_config["env_config"] = env_config
        model_config["log_level"] = "WARN"
        model_config["framework"] = framework

        # ray.init() # Other Ray APIs will not work until `ray.init()` is called.
        if model_name == "ppo":
            trainer = MODELS[model_name].PPOTrainer(env=env, config=model_config)
        elif model_name == "a2c":
            trainer = MODELS[model_name].A2CTrainer(env=env, config=model_config)
        elif model_name == "ddpg":
            trainer = MODELS[model_name].DDPGTrainer(env=env, config=model_config)
        elif model_name == "td3":
            trainer = MODELS[model_name].TD3Trainer(env=env, config=model_config)
        elif model_name == "sac":
            trainer = MODELS[model_name].SACTrainer(env=env, config=model_config)

        try:
            trainer.restore(agent_path)
            print("Restoring from checkpoint path", agent_path)
        except BaseException:
            raise ValueError("Fail to load agent!")

        agent = Rllib_model(trainer)

        return agent

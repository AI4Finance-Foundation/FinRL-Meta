# use three drl libraries: elegantrl, rllib, and SB3
from agents.elegantrl_models import DRLAgent as DRLAgent_erl
from agents.rllib_models import DRLAgent as DRLAgent_rllib
from agents.stablebaselines3_models import DRLAgent as DRLAgent_sb3


def test(drl_lib, env, model_name, **kwargs):
    # load elegantrl needs state dim, action dim and net dim
    net_dimension = kwargs.get("net_dimension", 2**7)
    cwd = kwargs.get("cwd", "./" + str(model_name))

    if drl_lib == "elegantrl":
        episode_total_assets = DRLAgent_erl.DRL_prediction(
            model_name=model_name,
            cwd=cwd,
            net_dimension=net_dimension,
            environment=env,
        )

        return episode_total_assets

    elif drl_lib == "rllib":
        # load agent
        episode_total_assets = DRLAgent_rllib.DRL_prediction(
            model_name=model_name,
            env=env,
            agent_path=cwd,
        )

        return episode_total_assets

    elif drl_lib == "stable_baselines3":
        episode_total_assets = DRLAgent_sb3.DRL_prediction_load_from_file(
            model_name=model_name, environment=env, cwd=cwd
        )

        return episode_total_assets

    else:
        raise ValueError("DRL library input is NOT supported. Please check.")

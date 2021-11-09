from gym.wrappers import TimeLimit
from gym.envs.registration import register
import pandas as pd
file ="./data/split/GBPUSD/weekly/GBPUSD_2017_2.csv"
df = pd.read_csv(file)

register(
    id='TradingGym-v0',
    entry_point='neo_finrl.env_fx_trading.env_fx:tgym',
    kwargs={'df': df,
            'env_config_file':'./neo_finrl/env_fx_trading/config/gdbusd-test-1.json'
        },
    max_episode_steps = 1440, #  (Optional[int]): The maximum number of steps that an episode can consist of
    reward_threshold=20000.0, #  (Optional[int]): The reward threshold before the task is considered solved
    nondeterministic=False, #  (bool): Whether this environment is non-deterministic even after seeding
    # order_enforce = True, # (Optional[int]): Whether to wrap the environment in an orderEnforcing wrapper
)

register(
    id='TradingGym-v1',
    entry_point='neo_finrl.env_fx_trading.env_fx:tgym',
    kwargs={'df': df,
            'env_config_file':'./neo_finrl/env_fx_trading/config/gdbusd-test-1.json'
        },
    max_episode_steps = 1440 * 5,
    reward_threshold=30000.0,
    nondeterministic=False,
)
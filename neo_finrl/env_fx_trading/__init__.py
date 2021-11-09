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
    max_episode_steps = 1440,
    reward_threshold=20000.0,
)

register(
    id='TradingGym-v1',
    entry_point='neo_finrl.env_fx_trading.env_fx:tgym',
    kwargs={'df': df,
            'env_config_file':'./neo_finrl/env_fx_trading/config/gdbusd-test-1.json'
        },
    max_episode_steps = 1440 * 5,
    reward_threshold=30000.0,
)
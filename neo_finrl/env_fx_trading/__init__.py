from gym.envs.registration import register

register(
    id='TradingGym-v0',
    entry_point='neo_finrl.env_fx_trading.env_fx:tgym',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1440},
    reward_threshold=20000.0,
    description = 'Trading environment for FX trading 5 mins bar weekly',
)

register(
    id='TradingGym-v1',
    entry_point='neo_finrl.env_fx_trading.env_fx:tgym',
    tags={'wrapper_config.TimeLimit.max_episode_steps': 1440*5},
    reward_threshold=30000.0,
    description = 'Trading environment for FX trading 1 min bar weekly',
)
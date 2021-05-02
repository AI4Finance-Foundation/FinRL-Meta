from gym.envs.registration import register

register(
    id='tradingEnv-v0',
    entry_point='gym_finrl.envs:StockTradingEnv',
)

# register(
#     id='trading_env_v1',
#     entry_point='gym_finrl.envs:StockTradingEnv',
# )

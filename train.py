from elegantrl.agent import *
from elegantrl.run import *
import torch
import ray
from neo_finrl.data_processor import DataProcessor

def train(start_date, end_date, ticker_list, data_source, time_interval,
          technical_indicator_list, drl_lib, env, agent, if_vix=True,
          **kwargs):
    # fetch data
    DP = DataProcessor(data_source, **kwargs)
    data = DP.download_data(ticker_list, start_date, end_date, time_interval)
    data = DP.clean_data(data)
    data = DP.add_technical_indicator(data, technical_indicator_list)
    if if_vix:
        data = DP.add_vix(data)
    price_array, tech_array, risk_array = DP.df_to_array(data, if_vix)

    # read parameters
    env_config = {'price_array': price_array,
                  'tech_array': tech_array,
                  'risk_array': risk_array,
                  'if_train': True}
    env_instance = env(config=env_config)

    learning_rate = kwargs.get('learning_rate', 0.00025)
    batch_size = kwargs.get('batch_size', 2 ** 9)
    gamma = kwargs.get('gamma', 0.99)
    seed = kwargs.get('seed', 312)
    total_timesteps = kwargs.get('total_timesteps', 1e6)
    net_dimension = kwargs.get('net_dimension', 2 ** 7)
    cwd = kwargs.get('cwd', './' + str(agent))

    # train using different libraries
    if drl_lib == 'elegantrl':

        if agent == 'ppo':
            args = Arguments(if_on_policy=True)
        else:
            raise ValueError('Invalid agent input or the agent input is not \
                             supported yet.')
        # try:
        #     args.agent = AgentPPO()
        #     args.env=env_instance
        #     args.cwd = cwd
        #     args.learning_rate = learning_rate
        #     args.batch_size = batch_size
        #     args.gamma = gamma
        #     args.seed = seed
        #     args.break_step = total_timesteps
        #     args.net_dimension = net_dimension
        # except:
        #     print('Invalid parameters input! Use default value.')
        #     args.agent = AgentPPO()
        #     args.env=env_instance
        #     args.learning_rate = 0.00025
        #     args.batch_size = 128
        #     args.gamma = 0.99
        #     args.seed = 312
        #     args.break_step = 1e6
        #     args.net_dimension = 2**7

        args.cwd = cwd
        args.env = env_instance

        args.agent = AgentPPO()
        args.agent.if_use_cri_target = True
        args.learning_rate = learning_rate  
        args.net_dim = net_dimension
        args.batch_size = batch_size
        args.gamma = gamma
        args.seed = seed
        args.break_step = int(total_timesteps)

        train_and_evaluate(args)
        #args.worker_num = 4
        #args.target_step = args.env.max_step * 2
        #train_and_evaluate_mp(args)

    elif drl_lib == 'rllib':
        # ray.init(num_gpus=1, local_mode = True)
        ray.init()
        assert ray.is_initialized() == True
        if agent == 'ppo':
            from ray.rllib.agents import ppo
            from ray.rllib.agents.ppo.ppo import PPOTrainer
            try:
                total_episodes = kwargs.get('total_episodes')
            except:
                print('total training episodes is not given! Use default value 1000')
                total_episodes = 1000
            config = ppo.DEFAULT_CONFIG.copy()
            config['env'] = env
            config["log_level"] = "WARN"
            config['gamma'] = 0.99
            config['env_config'] = {'price_array': price_array,
                                    'tech_array': tech_array,
                                    'risk_array': risk_array,
                                    'if_train': True}
            trainer = PPOTrainer(env=env, config=config)
            for i in range(total_episodes):
                trainer.train()
            trainer.save(cwd)

        else:
            raise ValueError('Invalid agent input or the agent input is not \
                 supported yet.')

    elif drl_lib == 'stable_baselines3':

        if agent == 'ppo':
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv

            env_train = DummyVecEnv([lambda: env_instance])
            model = PPO("MlpPolicy", env_train, learning_rate=learning_rate,
                        n_steps=2048, batch_size=batch_size, ent_coef=0.0,
                        gamma=gamma, seed=seed)
            model.learn(total_timesteps=total_timesteps, tb_log_name='ppo')
            print('Training finished!')
            model.save(cwd)
            print('Trained model saved in ' + str(cwd))

    else:
        raise ValueError('DRL library input is NOT supported. Please check.')


if __name__ == '__main__':
    from neo_finrl.config import FAANG_TICKER
    from neo_finrl.config import TECHNICAL_INDICATORS_LIST
    from neo_finrl.config import TRAIN_START_DATE
    from neo_finrl.config import TRAIN_END_DATE

    # construct environment
    from neo_finrl.env_stock_trading.env_stock_trading import StockTradingEnv

    env = StockTradingEnv

    # demo for elegantrl
    train(start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE,
          ticker_list=FAANG_TICKER, data_source='yahoofinance',
          time_interval='1D', technical_indicator_list=TECHNICAL_INDICATORS_LIST,
          drl_lib='elegantrl', env=env, agent='ppo', cwd='./test_ppo'
          , total_timesteps=3e5)

    # demo for rllib
    train(start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE,
          ticker_list=FAANG_TICKER, data_source='yahoofinance',
          time_interval='1D', technical_indicator_list=TECHNICAL_INDICATORS_LIST,
          drl_lib='rllib', env=env, agent='ppo', cwd='./test_ppo'
          , total_episodes=100)

    # demo for stable-baselines3
    train(start_date=TRAIN_START_DATE, end_date=TRAIN_END_DATE,
          ticker_list=FAANG_TICKER, data_source='yahoofinance',
          time_interval='1D', technical_indicator_list=TECHNICAL_INDICATORS_LIST,
          drl_lib='stable_baselines3', env=env, agent='ppo', cwd='./test_ppo'
          , total_timesteps=3e5)

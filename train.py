from elegantrl.agent import *
from elegantrl.run import *
import torch 
import ray

def train_erl(data_dic, drl_lib, env, agent, **kwargs):
    if 'price_ary' in data_dic and 'tech_ary' in data_dic and 'turbulence_ary'\
    in data_dic:
        price_ary = data_dic['price_ary']
        tech_ary = data_dic['tech_ary']
        turbulence_ary = data_dic['turbulence_ary']
    elif 'price_ary' in data_dic and 'tech_ary' in data_dic and 'turbulence_ary'\
    not in data_dic:
        price_ary = data_dic['price_ary']
        tech_ary = data_dic['tech_ary']
    else:
        raise ValueError('Invalid input data_dic!')
    
    env_config = {'price_ary':price_ary,
            'tech_ary':tech_ary,
            'turbulence_ary':turbulence_ary,
            'if_train':True}
    env_instance = env(config=env_config)
    
    learning_rate = kwargs.get('learning_rate', 0.00025)
    batch_size = kwargs.get('batch_size', 2**7)
    gamma = kwargs.get('gamma', 0.99)
    seed = kwargs.get('seed', 312)
    total_timesteps = kwargs.get('total_timesteps', 1e6)
    net_dimension = kwargs.get('net_dimension', 2**7)
    cwd = kwargs.get('cwd','./'+str(agent))
    if drl_lib == 'elegantrl':
        if agent == 'ppo':
            args = Arguments(agent=AgentPPO(), env=env_instance, if_on_policy=True)
        else:
            raise ValueError('Invalid agent input or the agent input is not \
                             supported yet.')
        try:
            args.cwd = cwd
            args.learning_rate = learning_rate
            args.batch_size = batch_size
            args.gamma = gamma
            args.seed = seed
            args.break_step = total_timesteps
            args.net_dimension = net_dimension
        except:
            print('Invalid parameters input! Use default value.')
            args.learning_rate = 0.00025
            args.batch_size = 128
            args.gamma = 0.99
            args.seed = 312
            args.break_step = 1e6
            args.net_dimension = 2**7
            
        train_and_evaluate(args)
    elif drl_lib == 'rllib':
        ray.init(ignore_reinit_error=True)
        if agent == 'ppo':
            from ray.rllib.agents import ppo
            from ray.rllib.agents.ppo.ppo import PPOTrainer
 
            config = ppo.DEFAULT_CONFIG.copy()
            config['env'] = env
            config["log_level"] = "WARN"
            config['env_config'] = {'price_ary':price_ary,
                                    'tech_ary':tech_ary,
                                    'turbulence_ary':turbulence_ary,
                                    'if_train':True}
            agent = PPOTrainer(env=env, config=config)
            agent.train()
            agent.save(cwd)
        
        else:
            raise ValueError('Invalid agent input or the agent input is not \
                 supported yet.')
            
    elif drl_lib == 'stable_baselines3':
        if agent == 'ppo':
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            env_train = DummyVecEnv([lambda : env_instance])
            model = PPO("MlpPolicy", env_train, learning_rate=learning_rate, 
                        n_steps=2048, batch_size=batch_size, ent_coef=0.0, 
                        gamma=gamma, seed=seed)
            model.learn(total_timesteps=total_timesteps, tb_log_name = 'ppo')
            print('Training finished!')
            model.save(cwd)
            print('Trained model saved in ' + str(cwd))
            
            
if __name__ == '__main__':    
    #fetch data
    from neo_finrl.data_processors.processor_alpaca import AlpacaEngineer as AE
    API_KEY = ""
    API_SECRET = ""
    APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
    AE = AE(API_KEY,
            API_SECRET,
            APCA_API_BASE_URL)
    stock_list = ['FB',  'AMZN', 'AAPL', 'NFLX', 'GOOG']
    start_date = '2021-01-01'
    end_date = '2021-01-10'
    tech_indicator_list = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30',
            'close_30_sma', 'close_60_sma']
    data = AE.data_fetch(stock_list, start_date, end_date, time_interval = '1Min')
    data = AE.clean_data(data)
    print(data)
    data = AE.add_technical_indicators(data)
    print(data)
    data = AE.add_turbulence(data)
    print(data)
    price_ary, tech_ary, turb_ary = AE.df_to_ary(data, tech_indicator_list)
    data_dic = {'price_ary':price_ary, 'tech_ary':tech_ary, 'turbulence_ary':turb_ary}
    #construct environment
    from neo_finrl.env_stock_trading.env_stock_alpaca import StockTradingEnv
    env = StockTradingEnv
    
    #demo for elegantrl
    train_erl(data_dic, drl_lib='elegantrl', env=env, agent='ppo', cwd='./test_ppo_erl'
              ,total_timesteps=3e4)
    
    '''#demo for rllib
    train_erl(data_dic, drl_lib='rllib', env=env, agent='ppo', cwd='./test_ppo_erl'
              ,total_timesteps=3e4)'''
    
    #demo for stable-baselines3
    train_erl(data_dic, drl_lib='stable_baselines3', env=env, agent='ppo', cwd='./test_ppo_erl'
              ,total_timesteps=3e4)

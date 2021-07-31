from elegantrl.run import *
import ray
import torch 

def train_erl(data_dic, drl_lib, env, agent, **kwargs):
    
    #load data
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
    
    #build environment
    env = env(price_ary = price_ary, tech_ary = tech_ary, turbulence_ary = \
              turbulence_ary, if_train = True)
    
    #set parameters for DRL algorithm
    learning_rate = kwargs.get('learning_rate', 0.00025)
    batch_size = kwargs.get('batch_size', 2**7)
    gamma = kwargs.get('gamma', 0.99)
    seed = kwargs.get('seed', 312)
    total_timesteps = kwargs.get('total_timesteps', 1e6)
    net_dimension = kwargs.get('net_dimension', 2**7)
    cwd = kwargs.get('cwd','./'+str(agent))
    
    #select drl library and train the agent
    if drl_lib == 'elegantrl':
        if agent == 'ppo':
            from elegantrl.agent import AgentPPO
            args = Arguments(agent=AgentPPO, env=env)
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
            train_and_evaluate(args)
        except:
            print('Invalid parameters input! Use default value.')
            args.learning_rate = 0.00025
            args.batch_size = 128
            args.gamma = 0.99
            args.seed = 312
            args.break_step = 1e6
            args.net_dimension = 2**7
            args.env = env
            train_and_evaluate(args)
            
    elif drl_lib == 'rllib':
        ray.init(ignore_reinit_error=True)
        if agent == 'ppo':
            from ray.rllib.agents.ppo import ddppo
            from ray.rllib.agents.ppo.ddppo import DDPPOTrainer
            config = ddppo.DEFAULT_CONFIG.copy()
            try:
                for key,value in kwargs.items():
                    config[key] = value
            except:
                print('Invalid parameters input! Use default value.')
                config = ddppo.DEFAULT_CONFIG.copy()
            
            agent = DDPPOTrainer(env=env, config=config)
            agent.train()
            agent.save(cwd)
        
        else:
            raise ValueError('Invalid agent input or the agent input is not \
                 supported yet.')
            
    elif drl_lib == 'stable_baselines3':
        if agent == 'ppo':
            from stable_baselines3 import PPO
            model = PPO("MlpPolicy", env, learning_rate=learning_rate, 
                        n_steps=2048, batch_size=batch_size, ent_coef=0.0, 
                        gamma=gamma, seed=seed)
            model.learn(total_timesteps=total_timesteps)
            model.save(cwd)
            
if __name__ == '__main__':
    #demo for elegantrl
    
    #fetch data
    from neo_finrl.data_processors.alpaca_engineer import AlpacaEngineer as AE
    API_KEY = ""
    API_SECRET = ""
    APCA_API_BASE_URL = 'https://paper-api.alpaca.markets'
    AE = AE(API_KEY,
            API_SECRET,
            APCA_API_BASE_URL)
    stock_list = ['FB',  'AMZN', 'AAPL', 'NFLX', 'GOOG']
    start_date = '2021-01-01'
    end_date = '2021-01-31'
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
    from neo_finrl.environments.env_stock_trading.env_stock_alpaca import StockTradingEnv
    env = StockTradingEnv
    
    #train
    train_erl(data_dic, drl_lib='elegantrl', env=env, agent='ppo', cwd='./test_ppo_erl')

# 1. Getting Started - Load Python Packages

# 1.1. Import Packages
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import sys
import time
import datetime
sys.path.append("../FinRL-Library")
os.chdir('FinRL-Meta')
from meta import config
from meta.data_processor import DataProcessor
from meta.env_fx_trading.env_fx import tgym

# 2. Download and Preprocess Data
config.USE_TIME_ZONE_SELFDEFINED = 1
config.TIME_ZONE_SELFDEFINED = 'US/Eastern'

dp = DataProcessor(data_source="yahoofinance",
                   start_date = '2017-01-01',
                   end_date = '2018-01-01',
                   time_interval='1D')

dp.run(ticker_list = ['GBPUSD=X'], technical_indicator_list = config.INDICATORS, if_vix=False)
df = dp.dataframe

df['time'] = df['time'] + ' 00:00'
df['time'] = pd.to_datetime(df['time'], format='%Y.%m.%d %H:%M')
symbol="GBPUSD"
df['symbol'] = symbol
df['dt'] = df['time']
df.index = df['dt']
df['minute'] = df['dt'].dt.minute
df['hour'] = df['dt'].dt.hour
df['weekday'] = df['dt'].dt.dayofweek
df['week'] = df['dt'].dt.isocalendar().week
df['month'] = df['dt'].dt.month
df['year'] = df['dt'].dt.year
df['day'] = df['dt'].dt.day

# 3. Train
def train(env, agent, df, if_vix = True,**kwargs):
    learning_rate = kwargs.get('learning_rate', 2 ** -15)
    batch_size = kwargs.get('batch_size', 2 ** 11 )
    gamma = kwargs.get('gamma', 0.99)
    seed = kwargs.get('seed', 312)
    total_timesteps = kwargs.get('total_timesteps', 1e6)
    net_dimension = kwargs.get('net_dimension', 2**9)
    cwd = kwargs.get('cwd','./'+str(agent))

    # env_instance = map(env, [pd.read_csv(f) for f in files])
    if agent == 'ppo':
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

        # env_train = [x.get_sb_env for x in env_instance ]
        vector_env = [lambda:env(df)]
        env_train = SubprocVecEnv(vector_env)
        model = PPO("MlpPolicy", env_train, learning_rate=learning_rate,
                    n_steps=2048, batch_size=batch_size, ent_coef=0.0,
                    gamma=gamma, seed=seed)
        start_time = time.time()
        s = datetime.datetime.now()
        print(f'Training start: {s}')
        model.learn(total_timesteps=total_timesteps, tb_log_name = 'ppo')
        print('Training finished!')
        model_name = "./data/models/GBPUSD-week-" + s.strftime('%Y%m%d%H%M%S')
        model.save(model_name)
        print(f'Trained model saved in {model_name}')
        print(f"trainning time: {(time.time() - start_time)}")

    else:
        raise ValueError('DRL library input is NOT supported. Please check.')

df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"}, inplace=True)

train(env=tgym,agent="ppo",df=df)

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

t = tgym(df)

# if model: del model # remove to demonstrate saving and loading
model_name='./data/models/GBPUSD-week-20220820173439.zip'
model = PPO.load(model_name)

start_time = time.time()
obs = t.reset()
t.current_step=0
done = False

while not done:
    action, _states = model.predict(obs)
    obs, rewards, done, info = t.step(action)
    t.render(mode='graph')

print(f"--- running time: {(time.time() - start_time)}---")

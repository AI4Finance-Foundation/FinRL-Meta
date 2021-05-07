import gym
import torch

from elegantrl.run import *
from neo_finrl.data_fetch.data_fetch_ccxt import ccxt_fetch_data
from neo_finrl.preprocess.preprocess_ccxt import preprocess_btc

from neo_finrl.envs.bitcoin_env import BitcoinEnv
from elegantrl.agent import *

'''data_fetch'''
df = ccxt_fetch_data(start = '20210101 00:00:00', end = '20210107 00:00:00',
                    pair = 'BTC/USDT', period = '1m')

data_ary = preprocess_btc(df)

args = Arguments(agent=None, env=None, gpu_id=0)
args.agent = AgentDQN()

'''choose environment'''
args.env = BitcoinEnv(processed_ary = data_ary, if_train=True)
args.env_eval = BitcoinEnv(processed_ary = data_ary, if_train=False)
args.net_dim = 2 ** 9 # change a default hyper-parameters
args.batch_size = 2 ** 8

train_and_evaluate(args)

args = Arguments(agent=None, env=None, gpu_id=0)
args.agent = AgentDQN()
args.env = BitcoinEnv(processed_ary = data_ary, if_train=False)
args.net_dim = 2 ** 9 # change a default hyper-parameters
args.batch_size = 2 ** 8
args.if_remove = False
args.cwd = './AgentDQN/BitcoinEnv_0'
args.init_before_training()
# Draw the graph
BitcoinEnv(processed_ary = data_ary, if_train=False)\
.draw_cumulative_return(self = args.env, _torch = torch)

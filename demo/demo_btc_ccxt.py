import torch
import sys
from elegantrl2.run import *
from neo_finrl.ccxt.ccxt_engineer import CCXTEngineer
from neo_finrl.ccxt.env_btc_ccxt6 import BitcoinEnv
from elegantrl2.agent import *
import numpy as np
import pandas as pd

CE = CCXTEngineer()
df = CE.data_fetch(start = '20210101 00:00:00', end = '20210630 00:00:00',
                    pair_list = ['BTC/USDT'], period = '1m')
df = CE.add_technical_indicators(df, pair_list= ['BTC/USDT'])
price_ary, tech_ary, date_ary = CE.df_to_ary(df, pair_list=['BTC/USDT'])
assert price_ary.shape[0] == tech_ary.shape[0]
assert tech_ary.shape[0] == date_ary.shape[0]

date_ary = np.asarray(date_ary, dtype=str)
mid1 = np.where(date_ary == '2021-05-01T08:00:00.000000000')[0][0]
mid2 = np.where(date_ary == '2021-06-01T08:00:00.000000000')[0][0]
print(mid1,mid2)

args = Arguments(agent=None, env=None, gpu_id=0)
args.agent = AgentPPO()
args.agent.cri_target = True
args.agent.lambda_entropy = 0.02
args.gamma = 0.997
args.env = BitcoinEnv(time_frequency = 15, price_ary=price_ary, tech_ary=tech_ary, mid1=mid1, mid2=mid2, mode ='train', gamma=args.gamma)
args.env_eval = BitcoinEnv(time_frequency = 15, price_ary=price_ary, tech_ary=tech_ary, mid1=mid1, mid2=mid2, mode ='test', gamma=args.gamma)

args.repeat_times = 2 ** 4
args.learning_rate = 2 ** -12
args.net_dim = 2 ** 7
args.batch_size = args.net_dim * 4

args.eval_gap = 2 ** 7
args.eval_times1 = 2 ** 0
args.eval_times2 = 2 ** 1
args.break_step = int(5e5)
args.if_allow_break = False
args.random_seed = 312
args.gpu_id = 0
args.cwd = './BitcoinEnv_AgentPPO'
args.random_seed += int(args.gpu_id)
args.target_step = args.env.max_step * 1
args.worker_num = 2
train_and_evaluate_mp(args)

args.agent = AgentPPO()
print(args.agent.Cri)
episode_returns, btc_returns = BitcoinEnv(time_frequency = 15, price_ary=price_ary, tech_ary=tech_ary, mid1=mid1, mid2=mid2, mode ='trade', gamma=args.gamma).draw_cumulative_return(args, _torch = torch)

import os
import gym
import time
import torch
import numpy as np
import numpy.random as rd
from copy import deepcopy
from elegantrl.replay import ReplayBuffer, ReplayBufferMP
from elegantrl.env import PreprocessEnv
from elegantrl.run import *

## DEMO using AgentPPO
def run__demo():
    import gym
    import neo_finrl
    gym.logger.set_level(40)  # Block warning: 'WARN: Box bound precision lowered by casting to float32'

    """DEMO 3: Custom Continuous action env: FinanceStock-v1"""
    args = Arguments(if_on_policy=True)
    '''choose an DRL algorithm'''
    from elegantrl.agent import AgentPPO
    args.agent = AgentPPO()

    # from Env import FinanceMultiStockEnv

    # args.env = FinanceMultiStockEnv(if_train=True, train_beg=0, train_len=1024)
    # args.env_eval = FinanceMultiStockEnv(if_train=False, train_beg=0, train_len=1024)  # eva_len = 1699 - train_len
    args.env = gym.make('tradingEnv-v0')
    args.env_eval = gym.make('tradingEnv-v0')
    args.reward_scale = 2 ** 0  # RewardRange: 0 < 1.0 < 1.25 <
    args.break_step = int(5e6)
    args.max_step = args.env.max_step
    args.max_memo = (args.max_step - 1) * 8
    args.batch_size = 2 ** 11
    args.if_allow_break = False
    "TotalStep:  2e5, TargetReward: 1.25, UsedTime:  200s"
    "TotalStep:  4e5, TargetReward: 1.50, UsedTime:  400s"
    "TotalStep: 10e5, TargetReward: 1.62, UsedTime: 1000s"

    '''train and evaluate'''
    train_and_evaluate(args)
    # args.rollout_num = 8
    # train_and_evaluate__multiprocessing(args)  # try multiprocessing in complete version
    exit()


    
if __name__ == '__main__':
    run__demo()

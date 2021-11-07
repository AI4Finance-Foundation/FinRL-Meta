from neo_finrl.env_fx_trading.env_fx import tgym
import os
import pandas as pd
from distutils.util import strtobool
import time
from torch.utils.tensorboard import SummaryWriter
import torch
import numpy as np
import random
import gym

def file_list(num_files):
    files = []
    for i in range(0,num_files): 
        file = f"./data/split/GBPUSD/weekly/GBPUSD_2017_{i}.csv"
        if os.path.isfile(file):
            files.append(file)
        else:
            print(f'not exist: {file}')
    return files    

def make_env(env, file, seed, rank, log_dir, max_episode_steps=None):
    env.seed(seed + rank)
    env.log_dir = log_dir
    env.max_episode_steps = max_episode_steps
    
    return [lambda:env(df=pd.read_csv(file))]

envs = gym.vector.SyncVectorEnv(
    make_env(
        tgym, file, seed=1, rank=rank, log_dir='./data/log', max_episode_steps=None) 
        for rank, file in enumerate(file_list(4)))

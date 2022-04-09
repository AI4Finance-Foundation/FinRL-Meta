import sys
from neo_finrl.agents.elegantrl.run import *
import torch 
import numpy as np

def demo_nas100_GPU_Podracer():  # 1.7+ 2.0+
    args = Arguments(if_on_policy=True)  # hyper-parameters of on-policy is different from off-policy
    args.random_seed = 312

    from neo_finrl.agents.elegantrl.agent import AgentPPO
    args.agent = AgentPPO()
    # args.agent.cri_target = False  # todo beta0
    args.agent.cri_target = True  # todo beta1
    args.agent.lambda_entropy = 0.02

    args.gamma = 0.999

    from neo_finrl.environments.env_nas100_wrds import StockEnvNAS100
    args.env = StockEnvNAS100(if_eval=False, gamma=args.gamma, turbulence_thresh=30)
    args.env_eval = StockEnvNAS100(if_eval=True, gamma=args.gamma, turbulence_thresh=15)

    args.repeat_times = 2 ** 4
    args.learning_rate = 2 ** -14
    args.net_dim = 2 ** 8
    args.batch_size = args.net_dim * 4

    args.eval_gap = 2 ** 8
    args.eval_times1 = 2 ** 0
    args.eval_times2 = 2 ** 1
    args.break_step = int(25e6)
    args.if_allow_break = False


    args.gpu_id = 0
    args.random_seed += int(args.gpu_id)
    args.target_step = args.env.max_step * 2
    args.worker_num = 4
    train_and_evaluate_mp(args)

    args.env = StockEnvNAS100(if_trade=True, gamma=args.gamma, turbulence_thresh=15)
    args.cwd = './StockEnvNAS_AgentPPO_1'
    episode_returns = StockEnvNAS100(if_trade=True, gamma=args.gamma, turbulence_thresh=15).draw_cumulative_return(args, _torch=torch)
    episode_returns = np.asarray(episode_returns,dtype = float)
    print(episode_returns[-1])
    np.save('./episode_returns.npy', episode_returns)
    
demo_nas100_GPU_Podracer()

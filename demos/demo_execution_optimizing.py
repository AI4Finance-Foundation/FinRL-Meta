'''Source: https://github.com/AI4Finance-Foundation/Liquidation-Analysis-using-Multi-Agent-Reinforcement-Learning-ICML-2019/blob/master/Model_training.ipynb'''
'''Paper: Multi-agent reinforcement learning for liquidation strategy analysis accepted by ICML 2019 AI in Finance: Applications and Infrastructure for Multi-Agent Learning. (https://arxiv.org/abs/1906.11046)'''

import numpy as np
from collections import deque
import finrl_meta.env_execution_optimizing.liquidation.env_execution_optimizing as env
from finrl_meta.env_execution_optimizing.liquidation import utils
from finrl_meta.env_execution_optimizing.liquidation.ddpg_agent import Agent

# Get the default financial and AC Model parameters
financial_params, ac_params = utils.get_env_param()

# Create simulation environment
env = env.MarketEnvironment()

# Initialize Feed-forward DNNs for Actor and Critic models. 
agent1 = Agent(state_size=env.observation_space_dimension(), action_size=env.action_space_dimension(),random_seed = 1225)
agent2 = Agent(state_size=env.observation_space_dimension(), action_size=env.action_space_dimension(),random_seed = 108)
# Set the liquidation time
lqt = 60

# Set the number of trades
n_trades = 60

# Set trader's risk aversion
tr1 = 1e-6
tr2 = 1e-6

# Set the number of episodes to run the simulation
episodes = 1300
shortfall_list = []
shortfall_hist1 = np.array([])
shortfall_hist2 = np.array([])
shortfall_deque1 = deque(maxlen=100)
shortfall_deque2 = deque(maxlen=100)
for episode in range(episodes): 
    # Reset the enviroment
    cur_state = env.reset(seed = episode, liquid_time = lqt, num_trades = n_trades, lamb1 = tr1,lamb2 = tr2)

    # set the environment to make transactions
    env.start_transactions()

    for i in range(n_trades + 1):
      
        # Predict the best action for the current state. 
        cur_state1 = np.delete(cur_state,8)
        cur_state2 = np.delete(cur_state,7)
        #print(cur_state[5:])
        action1 = agent1.act(cur_state1, add_noise = True)
        action2 = agent2.act(cur_state2, add_noise = True)
        #print(action1,action2)
        # Action is performed and new state, reward, info are received. 
        new_state, reward1, reward2, done1, done2, info = env.step(action1,action2)
        
        # current state, action, reward, new state are stored in the experience replay
        new_state1 = np.delete(new_state,8)
        new_state2 = np.delete(new_state,7)
        agent1.step(cur_state1, action1, reward1, new_state1, done1)
        agent2.step(cur_state2, action2, reward2, new_state2, done2)
        # roll over new state
        cur_state = new_state

        if info.done1 and info.done2:
            shortfall_hist1 = np.append(shortfall_hist1, info.implementation_shortfall1)
            shortfall_deque1.append(info.implementation_shortfall1)
            
            shortfall_hist2 = np.append(shortfall_hist2, info.implementation_shortfall2)
            shortfall_deque2.append(info.implementation_shortfall2)
            break
        
    if (episode + 1) % 100 == 0: # print average shortfall over last 100 episodes
        print('\rEpisode [{}/{}]\tAverage Shortfall for Agent1: ${:,.2f}'.format(episode + 1, episodes, np.mean(shortfall_deque1)))        
        print('\rEpisode [{}/{}]\tAverage Shortfall for Agent2: ${:,.2f}'.format(episode + 1, episodes, np.mean(shortfall_deque2)))
        shortfall_list.append([np.mean(shortfall_deque1),np.mean(shortfall_deque2)])
print('\nAverage Implementation Shortfall for Agent1: ${:,.2f} \n'.format(np.mean(shortfall_hist1)))
print('\nAverage Implementation Shortfall for Agent2: ${:,.2f} \n'.format(np.mean(shortfall_hist2)))
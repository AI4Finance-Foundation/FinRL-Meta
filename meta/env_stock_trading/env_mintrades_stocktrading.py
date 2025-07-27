from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")

# from stable_baselines3.common.logger import Logger, KVWriter, CSVOutputFormat

class StockTradingEnvMinTrades(gym.Env):
    """
    A stock trading env for OpenAI gym
    The environment is implements the simple StockTradingEnv as in env_stocktrading.py.
    However this env implements a custom reward function based on Sharpe Ratio, Profit factor, 
    and a constraint that atleast 75% of the available trading timestamps must involve a trade. 
    The exact implementation can be found in the _get_reward() function. 
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],

        min_trade_factor = 0.75 # (%) of tradeable timestamps which must involve a trade, else be penalized
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount  # get the initial cash
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.min_trade_factor = min_trade_factor
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.unique_trades = 0
        self.episode = 0
        self.step_count = 0
        # memorize all the total balance change
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]  # the initial total asset is calculated by cash + sum (num_share_stock_i * price_stock_i)
        self.rewards_memory = []
        self.actions_memory = []
        self.state_memory = (
            []
        )  # we need sometimes to preserve the state in the middle of trading process
        self.date_memory = [self._get_date()]
        #         self.logger = Logger('results',[CSVOutputFormat])
        # self.reset()
        self.trade_history = []  # list of {'profit': float}
        self.trade_days = set()  # set of unique datetime.date for trade activity
        self._seed()
    
    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + 2 * self.stock_dim + 1] != True:
                if self.state[index + self.stock_dim + 1] > 0:
                    sell_num_shares = min(abs(action), self.state[index + self.stock_dim + 1])
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    # Track trade profit (realized) in trade_history
                    trade_profit = (self.state[index + 1] - self.state[index + 1]) * sell_num_shares  # simplification; you may want to store the buy price on purchase for real PnL
                    if sell_num_shares > 0:
                        self.trade_history.append({'profit': trade_profit})
                        self.trade_days.add(self._get_date())
                    self.state[0] += sell_amount
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0
            return sell_num_shares

        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    if self.state[index + self.stock_dim + 1] > 0:
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = (
                            self.state[index + 1]
                            * sell_num_shares
                            * (1 - self.sell_cost_pct[index])
                        )
                        trade_profit = (self.state[index + 1] - self.state[index + 1]) * sell_num_shares
                        if sell_num_shares > 0:
                            self.trade_history.append({'profit': trade_profit})
                            self.trade_days.add(self._get_date())
                        self.state[0] += sell_amount
                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += (
                            self.state[index + 1]
                            * sell_num_shares
                            * self.sell_cost_pct[index]
                        )
                        self.trades += 1
                    else:
                        sell_num_shares = 0
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = _do_sell_normal()
        else:
            sell_num_shares = _do_sell_normal()
        return sell_num_shares
    
    def _buy_stock(self, index, action):
        def _do_buy():
            if self.state[index + 2 * self.stock_dim + 1] != True:
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )
                buy_num_shares = min(available_amount, action)
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount
                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
                # Track trade entry (for activity ratio)
                if buy_num_shares > 0:
                    self.trade_days.add(self._get_date())
            else:
                buy_num_shares = 0
            return buy_num_shares

        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
        return buy_num_shares
    
    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()
    
    def _get_reward(self):
        """
        Custom reward combining Sharpe Ratio, Profit Factor,
        and a bonus/penalty based on trade activity.

        Mathematical Representation
        Reward = Sharpe-Ratio + Profit-Factor + Min-Trades-Penalty

        Sharpe Ratio = (sqroot(N) x r_mean / sigma_r) where, 
        N = annualization factor (e.g., 252 for daily returns)
        r_mean = Mean of periodic returns
        sigma_r = Standard deviation of periodic returns

        Profit Factor = sum(profits)/sum(losses)

        Min-Trades-Penalty = {
                              0 if num_trades/total_timestamps >= 0.75
                             -1 if num_trades/total_timestamps < 0.75
                             }

        """
        returns = np.diff(self.asset_memory) / self.asset_memory[:-1]
        sharpe = np.sqrt(252) * np.mean(returns) / np.std(returns) if len(returns) > 1 and np.std(returns) != 0 else 0

        profits = [t['profit'] for t in self.trade_history if t['profit'] > 0]
        losses = [t['profit'] for t in self.trade_history if t['profit'] < 0]
        profit_factor = sum(profits) / abs(sum(losses)) if losses else float('inf') if profits else 0

        trade_days_count = self.unique_trades
        total_days_count = self.step_count
        trade_ratio = trade_days_count / total_days_count if total_days_count > 0 else 0

        activity_penalty = 0 if trade_ratio >= self.min_trade_factor else -1

        total_reward = sharpe + profit_factor + activity_penalty

        """
        if self.step_count % 100 == 0:
            print(f"Timestamp No : {self.step_count}, Unique Trade days : {self.unique_trades}")
            print(f"[Reward Breakdown] Sharpe: {sharpe:.2f}, PF: {profit_factor:.2f}, Trade Ratio: {trade_ratio:.2f}, Penalty: {activity_penalty}, Reward: {total_reward:.2f}")
        """

        return total_reward

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        self.step_count += 1

        if self.terminal:
            if self.make_plots:
                self._make_plot()
            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)]) * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            # Calculate custom reward at episode end
            self.reward = self._get_reward()

            df_total_value = pd.DataFrame(self.asset_memory)
            df_total_value.columns = ["account_value"]
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(
                1
            )
            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )
            
            print("\n=================================")
            print(f"day: {self.day}, episode: {self.episode}")
            print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
            print(f"end_total_asset: {end_total_asset:0.2f}")
            print(f"total_reward: {self.reward:0.2f}")
            print(f"total_cost: {self.cost:0.2f}")
            print(f"total_trades: {self.trades}")
            if df_total_value["daily_return"].std() != 0:
                print(f"Sharpe: {sharpe:0.3f}")
            print("=================================\n")

            self.step_count = 0
            self.unique_trades = 0
            # Usual diagnostic and export code...

            return self.state, self.reward, self.terminal, False, {}
        
        else:
            actions = actions * self.hmax
            actions = actions.astype(int)

            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)

            begin_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            argsort_actions = np.argsort(actions)
            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]
            isTrade = False
            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)
                isTrade = True
            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])
                isTrade = True

            if(isTrade):
                self.unique_trades += 1
            
            self.actions_memory.append(actions)

            # next timestep
            self.day += 1
            self.data = self.df.loc[self.day, :]

            if self.turbulence_threshold is not None:
                if len(self.df.tic.unique()) == 1:
                    self.turbulence = self.data[self.risk_indicator_col]
                elif len(self.df.tic.unique()) > 1:
                    self.turbulence = self.data[self.risk_indicator_col].values[0]

            self.state = self._update_state()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            """
            profit = end_total_asset - begin_total_asset
            self.trade_history.append({'profit':profit})
            """
            self.asset_memory.append(end_total_asset)
            self.date_memory.append(self._get_date())
            self.reward = self._get_reward()
            self.rewards_memory.append(self.reward)
            self.state_memory.append(self.state)

            return self.state, self.reward, self.terminal, False, {}
    
    def reset(self, *, seed=None, options=None):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()
        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.unique_trades = 0
        self.step_count = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        self.trade_history = []
        self.trade_days = set()
        self.state_memory = []
        return self.state, {}

    def render(self, mode="human", close=False):
        return self.state
    
    def _initiate_state(self):
        if self.initial:
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
                )
            else:
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        else:
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]
                    + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
                )
            else:
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
                )
        return state
    
    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
                + sum([self.data[tech].values.tolist() for tech in self.tech_indicator_list], [])
            )
        else:
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([self.data[tech]] for tech in self.tech_indicator_list), [])
            )
        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def save_state_memory(self):
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]
            state_list = self.state_memory
            df_states = pd.DataFrame(state_list)
            df_states.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        return df_states

    def save_asset_memory(self):
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame({"date": date_list, "account_value": asset_list})
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]
            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
        

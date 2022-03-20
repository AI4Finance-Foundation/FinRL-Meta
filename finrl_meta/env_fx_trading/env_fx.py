import datetime
import math
import random

import gym
import numpy as np
from gym import spaces
from gym.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl_meta.env_fx_trading.util.log_render import render_to_file
from finrl_meta.env_fx_trading.util.plot_chart import TradingChart
from finrl_meta.env_fx_trading.util.read_config import EnvConfig


class tgym(gym.Env):
    """forex/future/option trading gym environment
    1. Three action space (0 Buy, 1 Sell, 2 Nothing)
    2. Multiple trading pairs (EURUSD, GBPUSD...) under same time frame
    3. Timeframe from 1 min to daily as long as use candlestick bar (Open, High, Low, Close)
    4. Use StopLose, ProfitTaken to realize rewards. each pair can configure it own SL and PT in configure file
    5. Configure over night cash penalty and each pair's transaction fee and overnight position holding penalty
    6. Split dataset into daily, weekly or monthly..., with fixed time steps, at end of len(df). The business
        logic will force to Close all positions at last Close price (game over).
    7. Must have df column name: [(time_col),(asset_col), Open,Close,High,Low,day] (case sensitive)
    8. Addition indicators can add during the data process. 78 available TA indicator from Finta
    9. Customized observation list handled in json config file. 
    10. ProfitTaken = fraction_action * max_profit_taken + SL. 
    11. SL is pre-fixed
    12. Limit order can be configure, if limit_order == True, the action will preset buy or sell at Low or High of the bar,
        with a limit_order_expiration (n bars). It will be triggered if the price go cross. otherwise, it will be drop off
    13. render mode:
        human -- display each steps realized reward on console
        file -- create a transaction log
        graph -- create transaction in graph (under development)
    14.
    15. Reward, we want to incentivize profit that is sustained over long periods of time. 
        At each step, we will set the reward to the account balance multiplied by 
        some fraction of the number of time steps so far.The purpose of this is to delay 
        rewarding the agent too fast in the early stages and allow it to explore 
        sufficiently before optimizing a single strategy too deeply. 
        It will also reward agents that maintain a higher balance for longer,
        rather than those who rapidly gain money using unsustainable strategies.
    16. Observation_space contains all of the input variables we want our agent 
        to consider before making, or not making a trade. We want our agent to “see” 
        the forex data points (Open price, High, Low, Close, time serial, TA) in the game window, 
        as well a couple other data points like its account balance, current positions, 
        and current profit.The intuition here is that for each time step, we want our agent 
        to consider the price action leading up to the current price, as well as their 
        own portfolio’s status in order to make an informed decision for the next action.
    17. reward is forex trading unit Point, it can be configure for each trading pair
    """
    metadata = {'render.modes': ['graph', 'human', 'file', 'none']}

    def __init__(self, df, env_config_file='./neo_finrl/env_fx_trading/config/gdbusd-test-1.json') -> None:
        assert df.ndim == 2
        super(tgym, self).__init__()
        self.cf = EnvConfig(env_config_file)
        self.observation_list = self.cf.env_parameters("observation_list")

        self.balance_initial = self.cf.env_parameters("balance")
        self.over_night_cash_penalty = self.cf.env_parameters("over_night_cash_penalty")
        self.asset_col = self.cf.env_parameters("asset_col")
        self.time_col = self.cf.env_parameters("time_col")
        self.random_start = self.cf.env_parameters("random_start")
        self.log_filename = self.cf.env_parameters("log_filename") + datetime.datetime.now(
        ).strftime('%Y%m%d%H%M%S') + '.csv'

        self.df = df
        self.df["_time"] = df[self.time_col]
        self.df["_day"] = df["weekday"]
        self.assets = df[self.asset_col].unique()
        self.dt_datetime = df[self.time_col].sort_values().unique()
        self.df = self.df.set_index(self.time_col)
        self.visualization = False

        # --- reset value ---
        self.equity_list = [0] * len(self.assets)
        self.balance = self.balance_initial
        self.total_equity = self.balance + sum(self.equity_list)
        self.ticket_id = 0
        self.transaction_live = []
        self.transaction_history = []
        self.transaction_limit_order = []
        self.current_draw_downs = [0.0] * len(self.assets)
        self.max_draw_downs = [0.0] * len(self.assets)
        self.max_draw_down_pct = sum(self.max_draw_downs) / self.balance * 100
        self.current_step = 0
        self.episode = -1
        self.current_holding = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        self.current_day = 0
        self.done_information = ''
        self.log_header = True
        # --- end reset ---
        self.cached_data = [
            self.get_observation_vector(_dt) for _dt in self.dt_datetime
        ]
        self.cached_time_serial = ((self.df[["_time", "_day"]].sort_values("_time")) \
                                   .drop_duplicates()).values.tolist()

        self.reward_range = (-np.inf, np.inf)
        self.action_space = spaces.Box(low=0,
                                       high=3,
                                       shape=(len(self.assets),))
        # first two 3 = balance,current_holding, max_draw_down_pct
        _space = 3 + len(self.assets) \
                 + len(self.assets) * len(self.observation_list)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(_space,))
        print(
            f'initial done:\n'
            f'observation_list:{self.observation_list}\n '
            f'assets:{self.assets}\n '
            f'time serial: {min(self.dt_datetime)} -> {max(self.dt_datetime)} length: {len(self.dt_datetime)}'
        )
        self._seed()

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _history_df(self, i):
        pass

    def _take_action(self, actions, done):
        # action = math.floor(x),
        # profit_taken = math.ceil((x- math.floor(x)) * profit_taken_max - stop_loss_max )
        # _actions = np.floor(actions).astype(int)
        # _profit_takens = np.ceil((actions - np.floor(actions)) *self.cf.symbol(self.assets[i],"profit_taken_max")).astype(int)
        _action = 2
        _profit_taken = 0
        rewards = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        # need use multiply assets
        for i, x in enumerate(actions):
            self._o = self.get_observation(self.current_step, i, "Open")
            self._h = self.get_observation(self.current_step, i, "High")
            self._l = self.get_observation(self.current_step, i, "Low")
            self._c = self.get_observation(self.current_step, i, "Close")
            self._t = self.get_observation(self.current_step, i, "_time")
            self._day = self.get_observation(self.current_step, i, "_day")
            _action = math.floor(x)
            rewards[i] = self._calculate_reward(i, done)
            if self.cf.symbol(self.assets[i], "limit_order"):
                self._limit_order_process(i, _action, done)
            if _action in (0, 1) and not done \
                    and self.current_holding[i] < self.cf.symbol(self.assets[i], "max_current_holding"):
                # generating PT based on action fraction
                _profit_taken = math.ceil(
                    (x - _action) * self.cf.symbol(self.assets[i], "profit_taken_max")) + self.cf.symbol(self.assets[i],
                                                                                                         "stop_loss_max")
                self.ticket_id += 1
                if self.cf.symbol(self.assets[i], "limit_order"):
                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": 1,
                        "ActionPrice": self._l if _action == 0 else self._h,
                        "SL": self.cf.symbol(self.assets[i], "stop_loss_max"),
                        "PT": _profit_taken,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": 0,
                        "Reward": -self.cf.symbol(self.assets[i], "transaction_fee"),
                        "DateDuration": self._day,
                        "Status": 0,
                        "LimitStep": self.current_step,
                        "ActionStep": -1,
                        "CloseStep": -1,
                    }
                    self.transaction_limit_order.append(transaction)
                else:
                    transaction = {
                        "Ticket": self.ticket_id,
                        "Symbol": self.assets[i],
                        "ActionTime": self._t,
                        "Type": _action,
                        "Lot": 1,
                        "ActionPrice": self._c,
                        "SL": self.cf.symbol(self.assets[i], "stop_loss_max"),
                        "PT": _profit_taken,
                        "MaxDD": 0,
                        "Swap": 0.0,
                        "CloseTime": "",
                        "ClosePrice": 0.0,
                        "Point": 0,
                        "Reward": -self.cf.symbol(self.assets[i], "transaction_fee"),
                        "DateDuration": self._day,
                        "Status": 0,
                        "LimitStep": self.current_step,
                        "ActionStep": self.current_step,
                        "CloseStep": -1,
                    }
                    self.current_holding[i] += 1
                    self.tranaction_open_this_step.append(transaction)
                    self.balance -= self.cf.symbol(self.assets[i], "transaction_fee")
                    self.transaction_live.append(transaction)

        return sum(rewards)

    def _calculate_reward(self, i, done):
        _total_reward = 0
        _max_draw_down = 0
        for tr in self.transaction_live:
            if tr["Symbol"] == self.assets[i]:
                _point = self.cf.symbol(self.assets[i], "point")
                # cash discount overnight
                if self._day > tr["DateDuration"]:
                    tr["DateDuration"] = self._day
                    tr["Reward"] -= self.cf.symbol(self.assets[i], "over_night_penalty")

                if tr["Type"] == 0:  # buy
                    # stop loss trigger
                    _sl_price = tr["ActionPrice"] - tr["SL"] / _point
                    _pt_price = tr["ActionPrice"] + tr["PT"] / _point
                    if done:
                        p = (self._c - tr["ActionPrice"]) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                    elif self._l <= _sl_price:
                        self._manage_tranaction(tr, -tr["SL"], _sl_price)
                        _total_reward += -tr["SL"]
                        self.current_holding[i] -= 1
                    elif self._h >= _pt_price:
                        self._manage_tranaction(tr, tr["PT"], _pt_price)
                        _total_reward += tr["PT"]
                        self.current_holding[i] -= 1
                    else:  # still open
                        self.current_draw_downs[i] = int(
                            (self._l - tr["ActionPrice"]) * _point)
                        _max_draw_down += self.current_draw_downs[i]
                        if (
                                self.current_draw_downs[i] < 0
                                and tr["MaxDD"] > self.current_draw_downs[i]
                        ):
                            tr["MaxDD"] = self.current_draw_downs[i]

                elif tr["Type"] == 1:  # sell
                    # stop loss trigger
                    _sl_price = tr["ActionPrice"] + tr["SL"] / _point
                    _pt_price = tr["ActionPrice"] - tr["PT"] / _point
                    if done:
                        p = (tr["ActionPrice"] - self._c) * _point
                        self._manage_tranaction(tr, p, self._c, status=2)
                        _total_reward += p
                    elif self._h >= _sl_price:
                        self._manage_tranaction(tr, -tr["SL"], _sl_price)
                        _total_reward += -tr["SL"]
                        self.current_holding[i] -= 1
                    elif self._l <= _pt_price:
                        self._manage_tranaction(tr, tr["PT"], _pt_price)
                        _total_reward += tr["PT"]
                        self.current_holding[i] -= 1
                    else:
                        self.current_draw_downs[i] = int(
                            (tr["ActionPrice"] - self._h) * _point)
                        _max_draw_down += self.current_draw_downs[i]
                        if (
                                self.current_draw_downs[i] < 0
                                and tr["MaxDD"] > self.current_draw_downs[i]
                        ):
                            tr["MaxDD"] = self.current_draw_downs[i]

                if _max_draw_down > self.max_draw_downs[i]:
                    self.max_draw_downs[i] = _max_draw_down

        return _total_reward

    def _limit_order_process(self, i, _action, done):
        for tr in self.transaction_limit_order:
            if tr["Symbol"] == self.assets[i]:
                if tr["Type"] != _action or done:
                    self.transaction_limit_order.remove(tr)
                    tr["Status"] = 3
                    tr["CloseStep"] = self.current_step
                    self.transaction_history.append(tr)
                elif (tr["ActionPrice"] >= self._l and _action == 0) or (tr["ActionPrice"] <= self._h and _action == 1):
                    tr["ActionStep"] = self.current_step
                    self.current_holding[i] += 1
                    self.balance -= self.cf.symbol(self.assets[i], "transaction_fee")
                    self.transaction_limit_order.remove(tr)
                    self.transaction_live.append(tr)
                    self.tranaction_open_this_step.append(tr)
                elif tr["LimitStep"] + self.cf.symbol(self.assets[i], "limit_order_expiration") > self.current_step:
                    tr["CloseStep"] = self.current_step
                    tr["Status"] = 4
                    self.transaction_limit_order.remove(tr)
                    self.transaction_history.append(tr)

    def _manage_tranaction(self, tr, _p, close_price, status=1):
        self.transaction_live.remove(tr)
        tr["ClosePrice"] = close_price
        tr["Point"] = int(_p)
        tr["Reward"] = int(tr["Reward"] + _p)
        tr["Status"] = status
        tr["CloseTime"] = self._t
        self.balance += int(tr["Reward"])
        self.total_equity -= int(abs(tr["Reward"]))
        self.tranaction_close_this_step.append(tr)
        self.transaction_history.append(tr)

    def step(self, actions):
        # Execute one time step within the environment
        self.current_step += 1
        done = (self.balance <= 0
                or self.current_step == len(self.dt_datetime) - 1)
        if done:
            self.done_information += f'Episode: {self.episode} Balance: {self.balance} Step: {self.current_step}\n'
            self.visualization = True
        reward = self._take_action(actions, done)
        if self._day > self.current_day:
            self.current_day = self._day
            self.balance -= self.over_night_cash_penalty
        if self.balance != 0:
            self.max_draw_down_pct = abs(
                sum(self.max_draw_downs) / self.balance * 100)

            # no action anymore
        obs = ([self.balance, self.max_draw_down_pct] +
               self.current_holding +
               self.current_draw_downs +
               self.get_observation(self.current_step))
        return np.array(obs).astype(np.float32), reward, done, {
            "Close": self.tranaction_close_this_step
        }

    def get_observation(self, _step, _iter=0, col=None):
        if (col is None):
            return self.cached_data[_step]
        if col == '_day':
            return self.cached_time_serial[_step][1]

        elif col == '_time':
            return self.cached_time_serial[_step][0]
        col_pos = -1
        for i, _symbol in enumerate(self.observation_list):
            if _symbol == col:
                col_pos = i
                break
        assert col_pos >= 0
        return self.cached_data[_step][_iter * len(self.observation_list) +
                                       col_pos]

    def get_observation_vector(self, _dt, cols=None):
        cols = self.observation_list
        v = []
        for a in self.assets:
            subset = self.df.query(
                f'{self.asset_col} == "{a}" & {self.time_col} == "{_dt}"')
            assert not subset.empty
            v += subset.loc[_dt, cols].tolist()
        assert len(v) == len(self.assets) * len(cols)
        return v

    def reset(self):
        # Reset the state of the environment to an initial state
        self.seed()

        if self.random_start:
            self.current_step = random.choice(
                range(int(len(self.dt_datetime) * 0.5)))
        else:
            self.current_step = 0

        self.equity_list = [0] * len(self.assets)
        self.balance = self.balance_initial
        self.total_equity = self.balance + sum(self.equity_list)
        self.ticket_id = 0
        self.transaction_live = []
        self.transaction_history = []
        self.transaction_limit_order = []
        self.current_draw_downs = [0.0] * len(self.assets)
        self.max_draw_downs = [0.0] * len(self.assets)
        self.max_draw_down_pct = sum(self.max_draw_downs) / self.balance * 100
        self.episode = -1
        self.current_holding = [0] * len(self.assets)
        self.tranaction_open_this_step = []
        self.tranaction_close_this_step = []
        self.current_day = 0
        self.done_information = ''
        self.log_header = True
        self.visualization = False

        _space = (
                [self.balance, self.max_draw_down_pct] +
                [0] * len(self.assets) +
                [0] * len(self.assets) +
                self.get_observation(self.current_step))
        return np.array(_space).astype(np.float32)

    def render(self, mode='human', title=None, **kwargs):
        # Render the environment to the screen
        if mode in ('human', 'file'):
            printout = mode == 'human'
            pm = {
                "log_header": self.log_header,
                "log_filename": self.log_filename,
                "printout": printout,
                "balance": self.balance,
                "balance_initial": self.balance_initial,
                "tranaction_close_this_step": self.tranaction_close_this_step,
                "done_information": self.done_information
            }
            render_to_file(**pm)
            if self.log_header: self.log_header = False
        elif mode == 'graph' and self.visualization:
            print('plotting...')
            p = TradingChart(self.df, self.transaction_history)
            p.plot()

    def close(self):
        pass

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

import gym
import numpy as np
from numpy import random as rd


class StockTradingEnv(gym.Env):
    def __init__(
            self,
            config,
            initial_account=1e6,
            gamma=0.99,
            turbulence_thresh=99,
            min_stock_rate=0.1,
            max_stock=1e2,
            initial_capital=1e6,
            buy_cost_pct=1e-3,
            sell_cost_pct=1e-3,
            reward_scaling=2 ** -11,
            initial_stocks=None,
    ):
        price_array = config["price_array"]
        tech_array = config["tech_array"]
        turbulence_array = config["turbulence_array"]
        if_train = config["if_train"]
        self.price_array = price_array.astype(np.float32)
        self.tech_array = tech_array.astype(np.float32)
        self.turbulence_array = turbulence_array

        self.tech_array = self.tech_array * 2 ** -7
        self.turbulence_bool = (turbulence_array > turbulence_thresh).astype(np.float32)
        self.turbulence_array = (
                self.sigmoid_sign(turbulence_array, turbulence_thresh) * 2 ** -5
        ).astype(np.float32)

        stock_dim = self.price_array.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        self.initial_stocks = (
            np.zeros(stock_dim, dtype=np.float32)
            if initial_stocks is None
            else initial_stocks
        )

        # reset()
        self.time = None
        self.cash = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # cash + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_array.shape[1]
        # cash + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0

        self.observation_space = gym.spaces.Box(
            low=-3000, high=3000, shape=(self.state_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )

    def reset(self):
        self.time = 0
        price = self.price_array[self.time]

        if self.if_train:
            self.stocks = (
                    self.initial_stocks + rd.randint(0, 64, size=self.initial_stocks.shape)
            ).astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.cash = (
                    self.initial_capital * rd.uniform(0.95, 1.05)
                    - (self.stocks * price).sum()
            )
        else:
            self.stocks = self.initial_stocks.astype(np.float32)
            self.stocks_cool_down = np.zeros_like(self.stocks)
            self.cash = self.initial_capital

        self.total_asset = self.cash + (self.stocks * price).sum()
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(price)  # state

    def step(self, actions):
        actions = (actions * self.max_stock).astype(int)
        self.time += 1
        price = self.price_array[self.time]
        self.stocks_cool_down += 1

        if self.turbulence_bool[self.time] == 0:
            min_action = int(self.max_stock * self.min_stock_rate)  # stock_cd
            for index in np.where(actions < -min_action)[0]:  # sell_index:
                if price[index] > 0:  # Sell only if current asset is > 0
                    sell_num_shares = min(self.stocks[index], -actions[index])
                    self.stocks[index] -= sell_num_shares
                    self.cash += (
                            price[index] * sell_num_shares * (1 - self.sell_cost_pct)
                    )
                    self.stocks_cool_down[index] = 0
            for index in np.where(actions > min_action)[0]:  # buy_index:
                if (
                        price[index] > 0
                ):  # Buy only if the price is > 0 (no missing data in this particular date)
                    buy_num_shares = min(self.cash // price[index], actions[index])
                    self.stocks[index] += buy_num_shares
                    self.cash -= (
                            price[index] * buy_num_shares * (1 + self.buy_cost_pct)
                    )
                    self.stocks_cool_down[index] = 0

        else:  # sell all when turbulence
            self.cash += (self.stocks * price).sum() * (1 - self.sell_cost_pct)
            self.stocks[:] = 0
            self.stocks_cool_down[:] = 0

        state = self.get_state(price)
        total_asset = self.cash + (self.stocks * price).sum()
        reward = (total_asset - self.total_asset) * self.reward_scaling
        self.total_asset = total_asset

        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.time == self.max_step
        if done:
            reward = self.gamma_reward
            self.episode_return = total_asset / self.initial_total_asset

        return state, reward, done, dict()

    def get_state(self, price):
        cash = np.array(self.cash * (2 ** -12), dtype=np.float32)
        scale = np.array(2 ** -6, dtype=np.float32)
        return np.hstack(
            (
                cash,
                self.turbulence_array[self.time],
                self.turbulence_bool[self.time],
                price * scale,
                self.stocks * scale,
                self.stocks_cool_down,
                self.tech_array[self.time],
            )
        )  # state.astype(np.float32)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh

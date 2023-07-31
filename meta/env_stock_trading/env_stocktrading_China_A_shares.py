import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        stock_dim,
        hmax,
        initial_amount,
        buy_cost_pct,
        sell_cost_pct,
        reward_scaling,
        state_space,
        action_space,
        tech_indicator_list,
        turbulence_threshold=None,
        make_plots=False,
        print_verbosity=2,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        initial_buy=False,  # Use half of initial amount to buy
        hundred_each_trade=True,  # The number of shares per lot must be an integer multiple of 100
    ):
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
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
        self.turbulence_threshold = turbulence_threshold
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration
        # initalize state
        self.initial_buy = initial_buy
        self.hundred_each_trade = hundred_each_trade
        self.state = self._initiate_state()

        # initialize reward
        self.reward = 0
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.episode = 0
        # memorize all the total balance change
        self.portfolio_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self._seed()

    def _sell_stock(self, index, action):
        def _do_sell_normal():
            if self.state[index + 1] > 0:
                # Sell only if the price is > 0 (no missing data in this particular date)
                # perform sell action based on the sign of the action
                if self.state[index + self.stock_dim + 1] > 0:
                    # Sell only if current asset is > 0
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    if self.hundred_each_trade:
                        sell_num_shares = sell_num_shares // 100 * 100

                    sell_amount = self.state[index + 1] * sell_num_shares
                    cost_amount = sell_amount * self.sell_cost_pct
                    self.state[0] += sell_amount - cost_amount
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += cost_amount
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                sell_num_shares = 0

            return sell_num_shares

        # perform sell action based on the sign of the action
        if self.turbulence_threshold is not None:
            if self.turbulence >= self.turbulence_threshold:
                if self.state[index + 1] > 0:
                    # Sell only if the price is > 0 (no missing data in this particular date)
                    # if turbulence goes over threshold, just clear out all positions
                    if self.state[index + self.stock_dim + 1] > 0:
                        # Sell only if current asset is > 0
                        sell_num_shares = self.state[index + self.stock_dim + 1]
                        sell_amount = self.state[index + 1] * sell_num_shares
                        cost_amount = sell_amount * self.sell_cost_pct

                        self.state[0] += sell_amount - cost_amount

                        self.state[index + self.stock_dim + 1] = 0
                        self.cost += cost_amount
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
            if self.state[index + 1] > 0:
                # Buy only if the price is > 0 (no missing data in this particular date)
                available_amount = self.state[0] // self.state[index + 1]

                # update balance
                buy_num_shares = min(available_amount, action)
                if self.hundred_each_trade:
                    buy_num_shares = buy_num_shares // 100 * 100

                if buy_num_shares > 0:
                    buy_amount = self.state[index + 1] * buy_num_shares
                    cost_amount = buy_amount * self.buy_cost_pct

                    self.state[0] -= buy_amount + cost_amount

                    self.state[index + self.stock_dim + 1] += buy_num_shares

                    self.cost += cost_amount
                    self.trades += 1
                else:
                    buy_num_shares = 0
            else:
                buy_num_shares = 0

            return buy_num_shares

        # perform buy action based on the sign of the action
        if self.turbulence_threshold is None:
            buy_num_shares = _do_buy()
        else:
            if self.turbulence < self.turbulence_threshold:
                buy_num_shares = _do_buy()
            else:
                buy_num_shares = 0
                pass

        return buy_num_shares

    def _make_plot(self):
        portfolio_df = self.get_portfolio_df()
        plt.plot(portfolio_df["date"], portfolio_df["total_asset"], color="r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            print(f"Episode: {self.episode}")
            if self.make_plots:
                self._make_plot()

            portfolio_df = self.get_portfolio_df()
            begin_total_asset = portfolio_df["prev_total_asset"].iloc[0]
            end_total_asset = portfolio_df["total_asset"].iloc[-1]
            tot_reward = end_total_asset - begin_total_asset

            portfolio_df["daily_return"] = portfolio_df["total_asset"].pct_change(1)

            sharpe = None
            if portfolio_df["daily_return"].std() != 0:
                sharpe = (
                    (252**0.5)
                    * portfolio_df["daily_return"].mean()
                    / portfolio_df["daily_return"].std()
                )

            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {begin_total_asset:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if sharpe is not None:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    f"results/actions_{self.mode}_{self.model_name}_{self.episode}.csv"
                )
                portfolio_df.to_csv(
                    f"results/portfolio_{self.mode}_{self.model_name}_{self.episode}.csv",
                    index=False,
                )

            # Add outputs to logger interface
            # logger.record(key="environment/portfolio_value", value=end_total_asset)
            # logger.record(key="environment/total_reward", value=tot_reward)
            # logger.record(key="environment/total_reward_pct", value=(tot_reward / (end_total_asset - tot_reward)) * 100)
            # logger.record(key="environment/total_cost", value=self.cost)
            # logger.record(key="environment/total_trades", value=self.trades)

            return self.state, self.reward, self.terminal, {}

        else:
            actions = actions * self.hmax  # actions initially is scaled between 0 to 1
            actions = actions.astype(
                int
            )  # convert into integer because we can't by fraction of shares
            if self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    actions = np.array([-self.hmax] * self.stock_dim)

            # calculate information before trading
            begin_cash = self.state[0]
            begin_market_value = self._get_market_value()
            begin_total_asset = begin_cash + begin_market_value
            begin_cost = self.cost
            begin_trades = self.trades
            begin_stock = self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]

            argsort_actions = np.argsort(actions)

            sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

            for index in sell_index:
                actions[index] = self._sell_stock(index, actions[index]) * (-1)

            for index in buy_index:
                actions[index] = self._buy_stock(index, actions[index])

            if self.turbulence_threshold is not None:
                self.turbulence = self.data["turbulence"].values[0]

            # calculate information after trading
            end_cash = self.state[0]
            end_market_value = self._get_market_value()
            end_total_asset = end_cash + end_market_value
            end_cost = self.cost
            end_trades = self.trades
            end_stock = self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]

            self.actions_memory.append(actions)

            i_list = []
            for i in range(self.stock_dim):
                if begin_stock[i] - end_stock[i] == 0:
                    i_list.append(i)

            self.reward = end_total_asset - begin_total_asset
            for i in i_list:
                self.reward -= (
                    self.state[i + 1] * self.state[self.stock_dim + 1 + i]
                ) * 0.001

            date = self._get_date()

            self.portfolio_memory.append(
                {
                    "date": date,
                    "prev_total_asset": begin_total_asset,
                    "prev_cash": begin_cash,
                    "prev_market_value": begin_market_value,
                    "total_asset": end_total_asset,
                    "cash": end_cash,
                    "market_value": end_market_value,
                    "cost": end_cost - begin_cost,
                    "trades": end_trades - begin_trades,
                    "reward": self.reward,
                }
            )
            self.date_memory.append(date)

            self.reward = self.reward * self.reward_scaling

            # update next state
            self.day += 1
            self.data = self.df.loc[self.day, :]
            self.state = self._update_state()

        return self.state, self.reward, self.terminal, {}

    def reset(self):
        # initiate state
        self.day = 0
        self.data = self.df.loc[self.day, :]

        self.state = self._initiate_state()
        self.turbulence = 0
        self.cost = 0
        self.trades = 0
        self.terminal = False
        # self.iteration=self.iteration
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.portfolio_memory = []

        self.episode += 1

        return self.state

    def render(self, mode="human", close=False):
        return self.state

    def _initiate_state(self):
        if self.initial:
            # For Initial State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + [0] * self.stock_dim
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                )

                if self.initial_buy:
                    state = self.initial_buy_()
            else:
                # for single stock
                state = (
                    [self.initial_amount]
                    + [self.data.close]
                    + [0] * self.stock_dim
                    + sum(
                        [[self.data[tech]] for tech in self.tech_indicator_list],
                        [],
                    )
                )
        else:
            # Using Previous State
            if len(self.df.tic.unique()) > 1:
                # for multiple stock
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        [
                            self.data[tech].values.tolist()
                            for tech in self.tech_indicator_list
                        ],
                        [],
                    )
                )
            else:
                # for single stock
                state = (
                    [self.previous_state[0]]
                    + [self.data.close]
                    + self.previous_state[
                        (self.stock_dim + 1) : (self.stock_dim * 2 + 1)
                    ]
                    + sum(
                        [[self.data[tech]] for tech in self.tech_indicator_list],
                        [],
                    )
                )
        return state

    def _update_state(self):
        if len(self.df.tic.unique()) > 1:
            # for multiple stock
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    [
                        self.data[tech].values.tolist()
                        for tech in self.tech_indicator_list
                    ],
                    [],
                )
            )

        else:
            # for single stock
            state = (
                [self.state[0]]
                + [self.data.close]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    [[self.data[tech]] for tech in self.tech_indicator_list],
                    [],
                )
            )

        return state

    def _get_date(self):
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    def get_portfolio_df(self):
        portfolio_df = pd.DataFrame(self.portfolio_memory)
        portfolio_df["date"] = pd.to_datetime(portfolio_df["date"])
        portfolio_df.sort_values("date", inplace=True)
        return portfolio_df[
            [
                "date",
                "prev_total_asset",
                "prev_cash",
                "prev_market_value",
                "total_asset",
                "cash",
                "market_value",
                "cost",
                "trades",
                "reward",
            ]
        ]

    def _get_total_asset(self):
        """
        get current total asset value
        """
        return self.state[0] + self._get_market_value()

    def _get_market_value(self):
        """
        get current market value
        """
        return sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        )

    def save_asset_memory(self):
        portfolio_df = self.get_portfolio_df()
        df_account_value = portfolio_df[["date", "total_asset"]].rename(
            columns={"total_asset": "account_value"}
        )
        return df_account_value

    def save_action_memory(self):
        if len(self.df.tic.unique()) > 1:
            # date and close price length must match actions length
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list)
            df_date.columns = ["date"]

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
            # df_actions = pd.DataFrame({'date':date_list,'actions':action_list})
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

    def initial_buy_(self):
        """Initialize the state, already bought some"""
        prices = self.data.close.values.tolist()
        # only use half of the initial amount
        market_values_each_tic = 0.5 * self.initial_amount // len(prices)
        buy_nums_each_tic = [int(market_values_each_tic // p) for p in prices]
        if self.hundred_each_trade:
            buy_nums_each_tic = buy_nums_each_tic // 100 * 100

        buy_amount = sum(np.array(prices) * np.array(buy_nums_each_tic))

        state = (
            [self.initial_amount - buy_amount]
            + prices
            + buy_nums_each_tic
            + sum(
                [self.data[tech].values.tolist() for tech in self.tech_indicator_list],
                [],
            )
        )

        return state

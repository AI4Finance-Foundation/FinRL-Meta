"""From FinRL https://github.com/AI4Finance-LLC/FinRL/tree/master/finrl/env"""
import gym
import matplotlib
import numpy as np
import pandas as pd
from gym import spaces
from gym.utils import seeding

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path


class PortfolioAllocationEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step


    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df,
        initial_amount,
        order_df=True,
        normalize_df="by_previous_time",
        reward_scaling=1,
        comission_fee_model="trf",
        comission_fee_pct=0,
        features=["close", "high", "low"],
        valuation_feature="close",
        time_column="date",
        time_format="%Y-%m-%d",
        tic_column="tic",
        time_window=1,
        cwd="./",
        new_gym_api=False
    ):
        # super(StockEnv, self).__init__()
        # money = 10 , scope = 1
        self.time_window = time_window
        self.time_index = time_window - 1
        self.time_column = time_column
        self.time_format = time_format
        self.tic_column = tic_column
        self.df = df
        self.initial_amount = initial_amount
        self.reward_scaling = reward_scaling
        self.comission_fee_pct = comission_fee_pct
        self.comission_fee_model = comission_fee_model
        self.features = features
        self.valuation_feature = valuation_feature
        self.cwd = Path(cwd)
        self.new_gym_api = new_gym_api

        # results file
        self.results_file = self.cwd / "results" / "rl"
        self.results_file.mkdir(parents=True, exist_ok=True)

        # price variation
        self.df_price_variation = None

        # preprocess data
        self.preprocess_data(order_df, normalize_df)

        # dims and spaces
        self.tic_list = self.df[self.tic_column].unique()
        self.stock_dim = len(self.tic_list)
        self.action_space = 1 + self.stock_dim

        # sort datetimes
        self.sorted_times = sorted(set(self.df[time_column]))

        # define action space
        self.action_space = spaces.Box(low=0, high=1, shape=(self.action_space,))

        # define observation state
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(
                len(self.features),
                self.time_window,
                self.stock_dim
            ),
        )

        # load data from a pandas dataframe
        date_time = self.sorted_times[self.time_index]

        self.terminal = False
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount

        # memorize portfolio value each step
        self.asset_memory = {
            "initial" : [self.initial_amount],
            "final" : [self.initial_amount]
        }
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        # memorize actions performed
        self.actions_memory = [[1 / (1 + self.stock_dim)] * (1 + self.stock_dim)]
        # memorize portfolio weights at the ending of time step
        self.final_weights = [[1 / (1 + self.stock_dim)] * (1 + self.stock_dim)]
        # memorize datetimes
        self.date_memory = [date_time]

    def step(self, actions):
        self.terminal = self.time_index >= len(self.sorted_times) - 1

        if self.terminal:
            df = pd.DataFrame(
                {"date": self.date_memory, "daily_return": self.portfolio_return_memory}
            )
            df.set_index("date", inplace=True)
            plt.plot(np.exp((df.daily_return).cumsum()) * self.initial_amount, "r")
            plt.savefig(self.results_file / "cumulative_reward.png")
            plt.close()

            plt.plot(self.portfolio_return_memory, "r")
            plt.savefig(self.results_file / "rewards.png")
            plt.close()

            print("=================================")
            print("Initial portfolio value:{}".format(self.asset_memory['final'][0]))
            print("Final portfolio value: {}".format(self.portfolio_value))
            print("Final accumulative portfolio value: {}".format(self.portfolio_value / self.asset_memory['final'][0]))

            df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            df_daily_return.columns = ["daily_return"]
            df_daily_return["daily_return"] = np.exp(df_daily_return["daily_return"])
            if df_daily_return["daily_return"].std() != 0:
                sharpe = (
                    df_daily_return["daily_return"].mean()
                    / df_daily_return["daily_return"].std()
                )
                print("Sharpe ratio: ", sharpe)
            print("=================================")

            if self.new_gym_api:
                return self.state, self.reward, self.terminal, False, self.info
            return self.state, self.reward, self.terminal, self.info

        else:
            # transform action to numpy array (if it's a list)
            actions = np.array(actions, dtype=np.float32)

            # if necessary, normalize weights
            if np.sum(actions) == 1 and np.min(actions) >= 0:
                weights = actions
            else:
                weights = self._softmax_normalization(actions)
                
            # save initial portfolio weights for this time step
            self.actions_memory.append(weights)

            # get last step final weights and portfolio_value
            last_weights = self.final_weights[-1]

            # load next state
            self.time_index += 1
            self.state, self.info = self.get_state_and_info_from_time_index(self.time_index)

            # if using weights vector modifier, we need to modify weights vector
            if self.comission_fee_model == "wvm":
                delta_weights = weights - last_weights
                delta_assets = delta_weights[1:] # disconsider 
                # calculate fees considering weights modification
                fees = np.sum(np.abs(delta_assets * self.portfolio_value))
                if fees > weights[0] * self.portfolio_value:
                    weights = last_weights
                    # maybe add negative reward
                else:
                    portfolio = weights * self.portfolio_value
                    portfolio[0] -= fees
                    self.portfolio_value = np.sum(portfolio) # new portfolio value
                    weights = portfolio / self.portfolio_value # new weights
            elif self.comission_fee_model == "trf":
                last_mu = 1
                mu = 1 - 2 * self.comission_fee_pct + self.comission_fee_pct ** 2
                while abs(mu - last_mu) > 1e-10:
                    last_mu = mu
                    mu = (1 - self.comission_fee_pct * weights[0] - 
                          (2 * self.comission_fee_pct - self.comission_fee_pct ** 2) *
                          np.sum(np.maximum(last_weights[1:] - mu * weights[1:], 0))) / (1 - self.comission_fee_pct * weights[0])
                self.portfolio_value = mu * self.portfolio_value

            # save initial portfolio value of this time step
            self.asset_memory["initial"].append(self.portfolio_value)

            # time passes and time variation changes the portfolio distribution
            portfolio = self.portfolio_value * (weights * self.price_variation)

            # calculate new portfolio value and weights
            self.portfolio_value = np.sum(portfolio)
            weights = portfolio / self.portfolio_value

            # save final portfolio value and weights of this time step
            self.asset_memory["final"].append(self.portfolio_value)
            self.final_weights.append(weights)

            # save date memory
            self.date_memory.append(self.info["end_time"])

            # define portfolio return
            portfolio_return = np.log(self.asset_memory["final"][-1] / self.asset_memory["final"][-2])

            # save portfolio return memory
            self.portfolio_return_memory.append(portfolio_return)

            # Define portfolio return
            self.reward = portfolio_return
            self.reward = self.reward * self.reward_scaling

        if self.new_gym_api:
            return self.state, self.reward, self.terminal, False, self.info
        return self.state, self.reward, self.terminal, self.info

    def reset(self):
        # time_index must start a little bit in the future to implement lookback
        self.time_index = self.time_window - 1
        self.asset_memory = {
            "initial" : [self.initial_amount],
            "final" : [self.initial_amount]
        }
        self.state, self.info = self.get_state_and_info_from_time_index(self.time_index)
        self.portfolio_value = self.initial_amount
        self.terminal = False
        
        self.portfolio_return_memory = [0]
        self.actions_memory = [[1 / (1 + self.stock_dim)] * (1 + self.stock_dim)]
        self.final_weights = [[1 / (1 + self.stock_dim)] * (1 + self.stock_dim)]
        self.date_memory = [self.info["end_time"]]

        if self.new_gym_api:
            return self.state, self.info
        return self.state

    def get_state_and_info_from_time_index(self, time_index):
        end_time = self.sorted_times[time_index]
        start_time = self.sorted_times[time_index - (self.time_window - 1)]

        # define data to be used in this time step
        self.data = self.df[
            (self.df[self.time_column] >= start_time) &
            (self.df[self.time_column] <= end_time)
        ][[self.time_column, self.tic_column] + self.features]

        # define price variation of this time_step
        self.price_variation = self.df_price_variation[
                self.df_price_variation[self.time_column] == end_time
            ][self.valuation_feature].to_numpy()
        self.price_variation = np.insert(self.price_variation, 0, 1)
        
        # define state to be returned
        state = None
        for tic in self.tic_list:
            tic_data = self.data[self.data[self.tic_column] == tic]
            tic_data = tic_data[self.features].to_numpy().T
            tic_data = tic_data[..., np.newaxis]
            state = tic_data if state is None else np.append(state, tic_data, axis=2)
        info = {
            "tics": self.tic_list,
            "start_time": start_time,
            "end_time": end_time,
            "data": self.data,
            "price_variation": self.price_variation
        }
        return state, info

    def render(self, mode="human"):
        return self.state

    def _softmax_normalization(self, actions):
        numerator = np.exp(actions)
        denominator = np.sum(np.exp(actions))
        softmax_output = numerator / denominator
        return softmax_output

    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory

        df_account_value = pd.DataFrame(
            {"date": date_list, "daily_return": portfolio_return}
        )
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ["date"]

        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic.values
        df_actions.index = df_date.date

        return df_actions
    
    def enumerate_portfolio(self):
        print("Index: 0. Tic: Cash")
        for index, tic in enumerate(self.tic_list):
            print("Index: {}. Tic: {}".format(index + 1, tic))

    def preprocess_data(self, order, normalize):
        # order time dataframe by tic and time
        if order:
            self.df = self.df.sort_values(by=[self.tic_column, self.time_column])
        # defining price variation after ordering dataframe
        self.df_price_variation = self._temporal_variation_df()
        # apply normalization
        if normalize:
            self._normalize_dataframe(normalize)
        # transform str to datetime
        self.df[self.time_column] = pd.to_datetime(self.df[self.time_column])
        self.df_price_variation[self.time_column] = pd.to_datetime(self.df_price_variation[self.time_column])
        

    def _normalize_dataframe(self, normalize):
        if type(normalize) == str: 
            if normalize == "by_fist_time_window_value":
                print("Normalizing {} by first time window value...".format(self.features))
                self.df = self._temporal_variation_df(self.time_window - 1)
            elif normalize == "by_previous_time":
                print("Normalizing {} by previous time...".format(self.features))
                self.df = self._temporal_variation_df()
            elif normalize.startswith("by_"):
                normalizer_column = normalize[3:]
                print("Normalizing {} by {}".format(self.features, normalizer_column))
                for column in self.features:
                    self.df[column] = self.df[column] / self.df[normalizer_column]
        elif callable(normalize):
            print("Applying custom normalization function...")
            self.df = normalize(self.df)
        else:
            print("No normalization was performed.")


    def _temporal_variation_df(self, periods=1):
        df_temporal_variation = self.df.copy()
        prev_columns = []
        for column in self.features:
            prev_column = "prev_{}".format(column)
            prev_columns.append(prev_column)
            df_temporal_variation[prev_column] = df_temporal_variation.groupby(self.tic_column)[column].shift(periods=periods)
            df_temporal_variation[column] = df_temporal_variation[column] / df_temporal_variation[prev_column]
        df_temporal_variation = df_temporal_variation.drop(columns=prev_columns).fillna(1).reset_index(drop=True)
        return df_temporal_variation

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

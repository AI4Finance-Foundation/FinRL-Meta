# 1. Import Packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import time
import numpy as np
import sys

sys.path.insert(0, "/FinRL-Meta")
import os

os.chdir("FinRL-Meta")

from meta.env_crypto_trading.env_multiple_crypto import CryptoEnv
from train import train
from test import test
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# 2. Set Parameters
TICKER_LIST = [
    "BTCUSDT",
    "ETHUSDT",
    "ADAUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "SOLUSDT",
    "DOTUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "UNIUSDT",
]

time_interval = "1d"

TRAIN_START_DATE = "2020-10-01"
TRAIN_END_DATE = "2021-11-08"

TEST_START_DATE = "2021-11-08"
TEST_END_DATE = "2022-01-22"

INDICATORS = [
    "macd",
    "rsi",
    "cci",
    "dx",
]  # self-defined technical indicator list is NOT supported yet

net_dimension = 2**9

ERL_PARAMS = {
    "learning_rate": 2**-15,
    "batch_size": 2**11,
    "gamma": 0.99,
    "seed": 312,
    "net_dimension": 2**9,
    "target_step": 5000,
    "eval_gap": 30,
    "eval_times": 1,
}

# 3. Create Multiple Cryptocurrencies Trading Env
initial_capital = 1e6
env = CryptoEnv

# 4. Training
start_time = time.time()

train(
    start_date=TRAIN_START_DATE,
    end_date=TRAIN_END_DATE,
    ticker_list=TICKER_LIST,
    data_source="binance",
    time_interval=time_interval,
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    current_working_dir="./test_ppo",
    erl_params=ERL_PARAMS,
    break_step=5e4,
    if_vix=False,
)

duration_train = round((time.time() - start_time), 2)

# 5. Testing
start_time = time.time()

account_value_erl = test(
    start_date=TEST_START_DATE,
    end_date=TEST_END_DATE,
    ticker_list=TICKER_LIST,
    data_source="binance",
    time_interval=time_interval,
    technical_indicator_list=INDICATORS,
    drl_lib="elegantrl",
    env=env,
    model_name="ppo",
    current_working_dir="./test_ppo",
    net_dimension=net_dimension,
    if_vix=False,
)

duration_test = round((time.time() - start_time), 2)

# 6. Plotting
get_ipython().run_line_magic("matplotlib", "inline")

# calculate agent returns
account_value_erl = np.array(account_value_erl)
agent_returns = account_value_erl / account_value_erl[0]

# calculate buy-and-hold btc returns
price_array = np.load("./price_array.npy")
btc_prices = price_array[:, 0]
buy_hold_btc_returns = btc_prices / btc_prices[0]

# calculate equal weight portfolio returns
price_array = np.load("./price_array.npy")
initial_prices = price_array[0, :]
equal_weight = np.array([1e5 / initial_prices[i] for i in range(len(TICKER_LIST))])
equal_weight_values = []

for i in range(0, price_array.shape[0]):
    equal_weight_values.append(np.sum(equal_weight * price_array[i]))

equal_weight_values = np.array(equal_weight_values)
equal_returns = equal_weight_values / equal_weight_values[0]

# plot
plt.figure(dpi=200)
plt.grid()
plt.grid(which="minor", axis="y")
plt.title("Cryptocurrency Trading ", fontsize=12)
plt.plot(agent_returns, label="Trained RL Agent", color="red")
plt.plot(buy_hold_btc_returns, label="Buy-and-Hold BTC", color="blue")
plt.plot(equal_returns, label="Equal Weight Portfolio", color="green")
plt.ylabel("Return", fontsize=10)
plt.xlabel("Times (%s)" % time_interval, fontsize=10)
plt.xticks(size=10)
plt.yticks(size=10)

"""ax = plt.gca()
ax.xaxis.set_major_locator(ticker.MultipleLocator(210))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(21))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.005))
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=2))
ax.xaxis.set_major_formatter(ticker.FixedFormatter([]))"""

plt.legend(fontsize=8)

print("TRAIN_START_DATE: ", TRAIN_START_DATE, "   TRAIN_END_DATE: ", TRAIN_END_DATE)
print("TEST_START_DATE: ", TEST_START_DATE, "   TEST_END_DATE: ", TEST_END_DATE)
print("Time_Interval: ", time_interval)
print("ERL_PARAMS: ", ERL_PARAMS)
print("Episode_Total_Return: ", account_value_erl[-1] / initial_capital)

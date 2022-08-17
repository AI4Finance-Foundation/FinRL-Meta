# 1. Getting Started - Load Python Packages
# 1.1. Import Packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")
import sys

sys.path.append("../FinRL-Library")
os.chdir("FinRL-Meta")
from meta import config
from meta import config_tickers
from meta.data_processor import DataProcessor
from meta.env_portfolio_allocation.env_portfolio_yahoofinance import (
    StockPortfolioEnv,
)
from agents.stablebaselines3_models import DRLAgent
from plot import (
    backtest_stats,
    backtest_plot,
    get_daily_return,
    get_baseline,
    convert_daily_return_to_pyfolio_ts,
)

# 1.2. Create Folders
import main

main.check_and_make_directories(
    [
        config.DATA_SAVE_DIR,
        config.TRAINED_MODEL_DIR,
        config.TENSORBOARD_LOG_DIR,
        config.RESULTS_DIR,
    ]
)

# 2. Download and Preprocess Data
print(f"DOW_30_TICKER: {config_tickers.DOW_30_TICKER}")

dp = DataProcessor(
    data_source="yahoofinance",
    start_date="2009-01-01",
    end_date="2019-01-01",
    time_interval="1D",
)

dp.run(
    ticker_list=config_tickers.DOW_30_TICKER,
    technical_indicator_list=config.INDICATORS,
    if_vix=False,
)
df = dp.dataframe

df.head()

print("Shape of DataFrame: ", df.shape)

# Add covariance matrix as states
df.rename(columns={"time": "date"}, inplace=True)
df = df.sort_values(["date", "tic"], ignore_index=True)
df.index = df.date.factorize()[0]
df.drop(columns=["index"], inplace=True)

cov_list = []
return_list = []

# look back is one year
lookback = 252
for i in range(lookback, len(df.index.unique())):
    data_lookback = df.loc[i - lookback : i, :]
    price_lookback = data_lookback.pivot_table(
        index="date", columns="tic", values="close"
    ).dropna(axis=1)
    return_lookback = price_lookback.pct_change().dropna()
    return_list.append(return_lookback)

    covs = return_lookback.cov().values
    cov_list.append(covs)

df_cov = pd.DataFrame(
    {
        "date": df.date.unique()[lookback:],
        "cov_list": cov_list,
        "return_list": return_list,
    }
)
df = df.merge(df_cov, on="date")
df = df.sort_values(["date", "tic"]).reset_index(drop=True)
print("Shape of DataFrame: ", df.shape)

df.head()

# 4. Design Environment

# Training data split: 2009-01-01 to 2018-01-01
train = dp.data_split(df, "2009-01-01", "2018-01-01")

train.head()

# Environment for Portfolio Allocation
stock_dimension = len(train.tic.unique())
state_space = stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "transaction_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": config.INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
}

e_train_gym = StockPortfolioEnv(df=train, **env_kwargs)

env_train, _ = e_train_gym.get_sb_env()
# print(type(env_train))

# 5. Implement DRL Algorithms

# initialize
agent = DRLAgent(env=env_train)

# Model 1: A2C
agent = DRLAgent(env=env_train)
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.005, "learning_rate": 0.0002}
model_a2c = agent.get_model(model_name="a2c", model_kwargs=A2C_PARAMS)
trained_a2c = agent.train_model(
    model=model_a2c, tb_log_name="a2c", total_timesteps=50000
)
trained_a2c.save("/FinRL-Meta/trained_models/trained_a2c.zip")

# Model 2: PPO
agent = DRLAgent(env=env_train)
PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 0.0001,
    "batch_size": 128,
}
model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)
trained_ppo = agent.train_model(
    model=model_ppo, tb_log_name="ppo", total_timesteps=80000
)
trained_ppo.save("/FinRL-Meta/trained_models/trained_ppo.zip")

# Model 3: DDPG
agent = DRLAgent(env=env_train)
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS)
trained_ddpg = agent.train_model(
    model=model_ddpg, tb_log_name="ddpg", total_timesteps=50000
)
trained_ddpg.save("/FinRL-Meta/trained_models/trained_ddpg.zip")

# Model 4: SAC
agent = DRLAgent(env=env_train)
SAC_PARAMS = {
    "batch_size": 128,
    "buffer_size": 100000,
    "learning_rate": 0.0003,
    "learning_starts": 100,
    "ent_coef": "auto_0.1",
}
model_sac = agent.get_model("sac", model_kwargs=SAC_PARAMS)
trained_sac = agent.train_model(
    model=model_sac, tb_log_name="sac", total_timesteps=50000
)
trained_sac.save("/FinRL-Meta/trained_models/trained_sac.zip")

# Model 5: TD3
agent = DRLAgent(env=env_train)
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
model_td3 = agent.get_model("td3", model_kwargs=TD3_PARAMS)
trained_td3 = agent.train_model(
    model=model_td3, tb_log_name="td3", total_timesteps=30000
)
trained_td3.save("/FinRL-Meta/trained_models/trained_td3.zip")

# Trading
trade = dp.data_split(df, "2018-01-01", "2019-01-01")
e_trade_gym = StockPortfolioEnv(df=trade, **env_kwargs)

print("Shape of Trade DataFrame: ", trade.shape)

df_daily_return, df_actions = DRLAgent.DRL_prediction(
    model=trained_a2c, environment=e_trade_gym
)

df_daily_return.head()

df_daily_return.to_csv("/FinRL-Meta/results/df_daily_return.csv")

df_actions.head()

df_actions.to_csv("/FinRL-Meta/results/df_actions.csv")

# 6. Backtest Our Strategy

# 6.1. BackTestStats
from pyfolio import timeseries

DRL_strat = convert_daily_return_to_pyfolio_ts(df_daily_return)
perf_func = timeseries.perf_stats
perf_stats_all = perf_func(
    returns=DRL_strat,
    factor_returns=DRL_strat,
    positions=None,
    transactions=None,
    turnover_denom="AGB",
)

print("==============DRL Strategy Stats===========")
perf_stats_all

# baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
    ticker="^DJI",
    start=df_daily_return.loc[0, "date"],
    end=df_daily_return.loc[len(df_daily_return) - 1, "date"],
)

stats = backtest_stats(baseline_df, value_col_name="close")

# 6.2. BackTestPlot
import pyfolio

baseline_df = get_baseline(
    ticker="^DJI", start=df_daily_return.loc[0, "date"], end="2021-11-01"
)

baseline_returns = get_daily_return(baseline_df, value_col_name="close")

with pyfolio.plotting.plotting_context(font_scale=1.1):
    pyfolio.create_full_tear_sheet(
        returns=DRL_strat, benchmark_rets=baseline_returns, set_context=False
    )

# Min-Variance Portfolio Allocation
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models

unique_tic = trade.tic.unique()
unique_trade_date = trade.date.unique()

df.head()

# Calculate_portfolio_minimum_variance
portfolio = pd.DataFrame(index=range(1), columns=unique_trade_date)
initial_capital = 1000000
portfolio.loc[0, unique_trade_date[0]] = initial_capital

for i in range(len(unique_trade_date) - 1):
    df_temp = df[df.date == unique_trade_date[i]].reset_index(drop=True)
    df_temp_next = df[df.date == unique_trade_date[i + 1]].reset_index(drop=True)
    # Sigma = risk_models.sample_cov(df_temp.return_list[0])

    # calculate covariance matrix
    Sigma = df_temp.return_list[0].cov()

    # portfolio allocation
    ef_min_var = EfficientFrontier(None, Sigma, weight_bounds=(0, 0.1))

    # minimum variance
    raw_weights_min_var = ef_min_var.min_volatility()

    # get weights
    cleaned_weights_min_var = ef_min_var.clean_weights()

    # current capital
    cap = portfolio.iloc[0, i]

    # current cash invested for each stock
    current_cash = [element * cap for element in list(cleaned_weights_min_var.values())]

    # current held shares
    current_shares = list(np.array(current_cash) / np.array(df_temp.close))

    # next time period price
    next_price = np.array(df_temp_next.close)

    ##next_price * current share to calculate next total account value
    portfolio.iloc[0, i + 1] = np.dot(current_shares, next_price)

portfolio = portfolio.T
portfolio.columns = ["account_value"]

portfolio.head()

a2c_cumpod = (df_daily_return.daily_return + 1).cumprod() - 1

min_var_cumpod = (portfolio.account_value.pct_change() + 1).cumprod() - 1

dji_cumpod = (baseline_returns + 1).cumprod() - 1

# Plotly: DRL, Min-Variance, DJIA
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go

time_ind = pd.Series(df_daily_return.date)

trace0_portfolio = go.Scatter(
    x=time_ind, y=a2c_cumpod, mode="lines", name="A2C (Portfolio Allocation)"
)

trace1_portfolio = go.Scatter(x=time_ind, y=dji_cumpod, mode="lines", name="DJIA")
trace2_portfolio = go.Scatter(
    x=time_ind, y=min_var_cumpod, mode="lines", name="Min-Variance"
)
# trace3_portfolio = go.Scatter(x = time_ind, y = ddpg_cumpod, mode = 'lines', name = 'DDPG')
# trace4_portfolio = go.Scatter(x = time_ind, y = addpg_cumpod, mode = 'lines', name = 'Adaptive-DDPG')
# trace5_portfolio = go.Scatter(x = time_ind, y = min_cumpod, mode = 'lines', name = 'Min-Variance')
# trace4 = go.Scatter(x = time_ind, y = addpg_cumpod, mode = 'lines', name = 'Adaptive-DDPG')
# trace2 = go.Scatter(x = time_ind, y = portfolio_cost_minv, mode = 'lines', name = 'Min-Variance')
# trace3 = go.Scatter(x = time_ind, y = spx_value, mode = 'lines', name = 'SPX')

fig = go.Figure()
fig.add_trace(trace0_portfolio)
fig.add_trace(trace1_portfolio)
fig.add_trace(trace2_portfolio)

fig.update_layout(
    legend=dict(
        x=0,
        y=1,
        traceorder="normal",
        font=dict(family="sans-serif", size=15, color="black"),
        bgcolor="White",
        bordercolor="white",
        borderwidth=2,
    ),
)

# fig.update_layout(legend_orientation="h")

fig.update_layout(
    title={
        #'text': "Cumulative Return using FinRL",
        "y": 0.85,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    }
)

# with Transaction cost
# fig.update_layout(title =  'Quarterly Trade Date')

fig.update_layout(
    #    margin=dict(l=20, r=20, t=20, b=20),
    paper_bgcolor="rgba(1,1,0,0)",
    plot_bgcolor="rgba(1, 1, 0, 0)",
    # xaxis_title="Date",
    yaxis_title="Cumulative Return",
    xaxis={
        "type": "date",
        "tick0": time_ind[0],
        "tickmode": "linear",
        "dtick": 86400000.0 * 80,
    },
)
fig.update_xaxes(
    showline=True,
    linecolor="black",
    showgrid=True,
    gridwidth=1,
    gridcolor="LightSteelBlue",
    mirror=True,
)
fig.update_yaxes(
    showline=True,
    linecolor="black",
    showgrid=True,
    gridwidth=1,
    gridcolor="LightSteelBlue",
    mirror=True,
)
fig.update_yaxes(zeroline=True, zerolinewidth=1, zerolinecolor="LightSteelBlue")

fig.show()

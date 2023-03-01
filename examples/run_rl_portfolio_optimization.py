import os

import pandas as pd
from stable_baselines3 import A2C

from agents.stablebaselines3_models import DRLAgent
from meta import config
from meta import config_tickers
from meta.data_processor import DataProcessor
from meta.env_portfolio_allocation.env_portfolio_yahoofinance import StockPortfolioEnv


def data_split(df, start, end, target_date_col="time"):
    """
    split the dataset into training or testing using time
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    dates = pd.to_datetime(df[target_date_col])
    data = df[(dates >= start) & (dates < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def main(
    start_date=config.TRAIN_START_DATE,
    end_date=config.TRADE_END_DATE,
    ticker_list=config_tickers.DOW_30_TICKER,
    time_interval="1D",
    data_source="yahoofinance",
    technical_indicator_list=config.INDICATORS,
    if_vix=True,
    hmax=100,
    initial_amount=1000000,
    transaction_cost_pct=0.001,
    reward_scaling=1e-4,
    use_cached_model=False,
):
    # download data
    dp = DataProcessor(
        data_source=data_source,
        start_date=start_date,
        end_date=end_date,
        time_interval=time_interval,
    )

    price_array, tech_array, turbulence_array = dp.run(
        ticker_list,
        technical_indicator_list,
        if_vix=if_vix,
        cache=True,
        select_stockstats_talib=0,
    )

    # add covariance matrix as states
    df = dp.dataframe
    df = df.sort_values(["time", "tic"], ignore_index=True)
    df.index = df.time.factorize()[0]

    df["pct_change"] = df.groupby("tic").close.pct_change()

    cov_list = []
    # look back is one year
    lookback = 252
    for i in range(lookback, len(df.index.unique())):
        data_lookback = df.loc[i - lookback : i, :]
        price_lookback = data_lookback.pivot_table(
            index="time", columns="tic", values="close"
        )
        return_lookback = price_lookback.pct_change().dropna()
        covs = return_lookback.cov().values
        cov_list.append(covs)
    # df["mean_pct_change_lookback"] = df.rolling(lookback)["pct_change"].mean()
    # df["ewm_returns"] = df["pct_change"].ewm(span=50).mean()
    df_cov = pd.DataFrame({"time": df.time.unique()[lookback:], "cov_list": cov_list})
    df = df.merge(df_cov, on="time")
    df = df.sort_values(["time", "tic"]).reset_index(drop=True)

    train_df = data_split(df, start=config.TRAIN_START_DATE, end=config.TRAIN_END_DATE)
    stock_dimension = len(train_df.tic.unique())
    state_space = stock_dimension
    print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

    env_kwargs = {
        "hmax": hmax,
        "initial_amount": initial_amount,
        "transaction_cost_pct": transaction_cost_pct,
        "state_space": state_space,
        "stock_dim": stock_dimension,
        "tech_indicator_list": config.INDICATORS,
        "action_space": stock_dimension,
        "reward_scaling": reward_scaling,
    }
    # use cached model if you are iterating on evaluation code
    if use_cached_model and os.path.exists("saved_models/a2c_model.pt"):
        trained_a2c = A2C.load("saved_models/a2c_model.pt")
    else:
        e_train_gym = StockPortfolioEnv(df=train_df, **env_kwargs)
        agent = DRLAgent(env=e_train_gym)

        A2C_PARAMS = {"n_steps": 10, "ent_coef": 0.005, "learning_rate": 0.0004}
        model_a2c = agent.get_model(model_name="a2c", model_kwargs=A2C_PARAMS)
        trained_a2c = agent.train_model(
            model=model_a2c, tb_log_name="a2c", total_timesteps=40000
        )
        # save trained_a2c model
        trained_a2c.save("saved_models/a2c_model.pt")

    # evaluate on test data
    test_df = data_split(df, start=config.TEST_START_DATE, end=config.TEST_END_DATE)
    e_test_gym = StockPortfolioEnv(df=test_df, **env_kwargs)

    df_daily_return, df_actions = DRLAgent.DRL_prediction(
        model=trained_a2c,
        environment=e_test_gym,
    )


if __name__ == "__main__":
    """
    python FinRL-Meta/examples/run_rl_portfolio_optimization.py
    """
    main()

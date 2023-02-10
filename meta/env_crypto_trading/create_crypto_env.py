# 1. Import Packages
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import sys

sys.path.insert(0, "FinRL-Meta")

import os

os.chdir("FinRL-Meta")

from meta.data_processor import DataProcessor
from meta.env_crypto_trading.env_multiple_crypto import CryptoEnv


# Creating the Training Environment
def create_train_env(
    data_source,
    train_start_date,
    train_end_date,
    time_interval,
    ticker_list,
    technical_indicator_list,
    if_vix,
    **kwargs
):
    dp = DataProcessor(
        data_source, train_start_date, train_end_date, time_interval, **kwargs
    )

    price_array, tech_array, turbulence_array = dp.run(
        ticker_list, technical_indicator_list, if_vix
    )

    data_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
    }

    env_train = CryptoEnv(
        config=data_config,
        lookback=kwargs.get("lookback", 1),
        initial_capital=kwargs.get("initial_capital", 1e6),
        buy_cost_pct=kwargs.get("buy_cost_pct", 1e-3),
        sell_cost_pct=kwargs.get("sell_cost_pct", 1e-3),
        gamma=kwargs.get("gamma", 0.99),
    )

    return env_train


# Creating the Testing Environment
def create_test_env(
    data_source,
    test_start_date,
    test_end_date,
    time_interval,
    ticker_list,
    technical_indicator_list,
    if_vix,
    **kwargs
):
    dp = DataProcessor(
        data_source, test_start_date, test_end_date, time_interval, **kwargs
    )

    price_array, tech_array, turbulence_array = dp.run(
        ticker_list, technical_indicator_list, if_vix
    )

    np.save("./price_array.npy", price_array)

    data_config = {
        "price_array": price_array,
        "tech_array": tech_array,
        "turbulence_array": turbulence_array,
    }

    env_test = CryptoEnv(
        config=data_config,
        lookback=kwargs.get("lookback", 1),
        initial_capital=kwargs.get("initial_capital", 1e6),
        buy_cost_pct=kwargs.get("buy_cost_pct", 1e-3),
        sell_cost_pct=kwargs.get("sell_cost_pct", 1e-3),
        gamma=kwargs.get("gamma", 0.99),
    )

    return env_test

import pytest
from finrl_meta.data_processor import DataProcessor
from typing import List
import pandas as pd


@pytest.fixture(scope="session")
def start_date():
    return "2021-01-01"


@pytest.fixture(scope="session")
def end_date():
    return "2021-10-31"


@pytest.fixture(scope="session")
def time_interval():
    return "1d"


@pytest.fixture(scope="session")
def ticker_list_str():
    return "AAPL"


@pytest.fixture(scope="session")
def ticker_list():
    return ["AAPL"]

def test_yahoo_download(
    time_interval: str,
    start_date: str,
    end_date: str,
    ticker_list: List[str],
) -> None:
    """
    Tests the Yahoo Downloader and the returned data shape
    Tries to retrieve Apple's stock data, 1day interval, from Yahoo Finance
    A list with 1 element is passed to the ticker: ['AAPL']
    """
    assert isinstance(start_date, str)
    assert isinstance(end_date, str)
    data_source = "yahoofinance"
    dp = DataProcessor(data_source, start_date, end_date, time_interval)
    dp.download_data(ticker_list)
    assert isinstance(dp.dataframe, pd.DataFrame)
    assert dp.dataframe.shape == (210, 9)


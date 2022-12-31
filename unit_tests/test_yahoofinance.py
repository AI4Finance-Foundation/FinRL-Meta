import copy
from typing import List

import pandas as pd
import pytest

from meta.config import INDICATORS
from meta.config_tickers import DOW_30_TICKER
from meta.config_tickers import SINGLE_TICKER
from meta.data_processor import DataProcessor


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
def tech_indicator_list():
    return INDICATORS


pytestmark = pytest.mark.parametrize(
    "ticker_input, expected_df_size", [(SINGLE_TICKER, 210), (DOW_30_TICKER, 6300)]
)


class TestYahoo:
    def test_yahoofinance_download(
        self,
        time_interval: str,
        start_date: str,
        end_date: str,
        ticker_input: List[str],
        expected_df_size: int,
    ) -> None:
        """
        Tests the Yahoo Downloader and the returned data shape
        """
        assert isinstance(start_date, str)
        assert isinstance(end_date, str)
        data_source = "yahoofinance"
        dp = DataProcessor(data_source, start_date, end_date, time_interval)
        dp.download_data(ticker_input)
        assert isinstance(dp.dataframe, pd.DataFrame)
        assert (
            dp.dataframe.shape
            == (
                expected_df_size,
                9,
            )
            or dp.dataframe.shape == (expected_df_size - 1, 9)
            or dp.dataframe.shape == (expected_df_size - 30, 9)
        )

    def test_yahoofinance_clean_data(
        self,
        time_interval: str,
        start_date: str,
        end_date: str,
        ticker_input: List[str],
        expected_df_size: List[str],
    ) -> None:
        """
        Tests the Yahoo Downloader and the clean_data() function
        """
        data_source = "yahoofinance"
        dp = DataProcessor(data_source, start_date, end_date, time_interval)
        dp.download_data(ticker_input)
        dp.clean_data()

    @pytest.mark.parametrize("talib", [0, 1])
    def test_yahoofinance_add_tech_indicators(
        self,
        time_interval: str,
        start_date: str,
        end_date: str,
        ticker_input: List[str],
        tech_indicator_list: List[str],
        expected_df_size: int,
        talib: int,
    ) -> None:
        """
        Tests the Yahoo Downloader and the returned data shape
        """
        data_source = "yahoofinance"
        dp = DataProcessor(data_source, start_date, end_date, time_interval)
        dp.download_data(ticker_input)
        dp.add_technical_indicator(tech_indicator_list, select_stockstats_talib=talib)

    @pytest.mark.parametrize("if_vix", [True, False])
    def test_yahoofinance_run(
        self,
        time_interval: str,
        start_date: str,
        end_date: str,
        ticker_input: List[str],
        tech_indicator_list: List[str],
        expected_df_size: int,
        if_vix: bool,
    ) -> None:
        data_source = "yahoofinance"
        dp = DataProcessor(data_source, start_date, end_date, time_interval)
        price_array, tech_array, turbulence_array = dp.run(
            ticker_input, tech_indicator_list, if_vix
        )

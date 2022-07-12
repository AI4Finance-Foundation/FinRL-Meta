from typing import List

import pandas as pd
import pytest

from meta.config import INDICATORS
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
    "ticker_input_binance, expected_df_size_binance",
    [(["BTCUSDT"], 302), (["ETHUSDT"], 302)],
)


class TestBinance:
    def test_binance_download(
        self,
        time_interval: str,
        start_date: str,
        end_date: str,
        ticker_input_binance: List[str],
        expected_df_size_binance: int,
        tech_indicator_list: List[str],
    ) -> None:
        """
        Tests the Binance Downloader and the returned data shape
        """
        assert isinstance(start_date, str)
        assert isinstance(end_date, str)
        data_source = "binance"
        dp = DataProcessor(data_source, start_date, end_date, time_interval)
        dp.download_data(ticker_input_binance)
        assert isinstance(dp.dataframe, pd.DataFrame)
        assert (
            dp.dataframe.shape
            == (
                expected_df_size_binance,
                8,
            )
            or dp.dataframe.shape == (expected_df_size_binance - 1, 8)
            or dp.dataframe.shape == (expected_df_size_binance + 1, 8)
        )

    def test_binance_clean_data(
        self,
        time_interval: str,
        start_date: str,
        end_date: str,
        ticker_input_binance: List[str],
        expected_df_size_binance: int,
        tech_indicator_list: List[str],
    ) -> None:
        data_source = "binance"
        dp = DataProcessor(data_source, start_date, end_date, time_interval)
        dp.download_data(ticker_input_binance)

    @pytest.mark.parametrize("talib", [0, 1])
    def test_binance_add_tech_indicators(
        self,
        time_interval: str,
        start_date: str,
        end_date: str,
        ticker_input_binance: List[str],
        expected_df_size_binance: int,
        tech_indicator_list: List[str],
        talib: int,
    ) -> None:
        data_source = "binance"
        dp = DataProcessor(data_source, start_date, end_date, time_interval)
        dp.download_data(ticker_input_binance)
        dp.add_technical_indicator(tech_indicator_list, select_stockstats_talib=talib)

    @pytest.mark.parametrize("if_vix", [True, False])
    def test_binance_run(
        self,
        time_interval: str,
        start_date: str,
        end_date: str,
        ticker_input_binance: List[str],
        expected_df_size_binance: int,
        tech_indicator_list: List[str],
        if_vix: bool,
    ) -> None:
        data_source = "binance"
        dp = DataProcessor(data_source, start_date, end_date, time_interval)
        price_array, tech_array, turbulence_array = dp.run(
            ticker_input_binance, tech_indicator_list, if_vix
        )

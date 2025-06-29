from typing import List

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from meta.config import INDICATORS
from meta.config_tickers import DOW_30_TICKER
from meta.config_tickers import SINGLE_TICKER
from meta.data_processor import DataProcessor
from meta.data_processors._base import DataSource
from meta.data_processors.yahoofinance import Yahoofinance


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


@pytest.fixture(scope="session")
def yahoofinance_processor():
    """Fixture for YahooFinance processor instance."""
    return Yahoofinance(
        data_source="yahoofinance",
        start_date="2020-01-01",
        end_date="2020-01-03",
        time_interval="1d"
    )


@pytest.fixture(scope="session")
def sample_raw_data():
    """Fixture for sample raw data for testing _adjust_prices method."""
    return pd.DataFrame(
        {
            "date": ["2020-01-01", "2020-01-02"],
            "open": [100.0, 102.0],
            "high": [105.0, 106.0],
            "low": [98.0, 100.0],
            "close": [104.0, 105.0],
            "adjusted_close": [100.0, 102.9],  # Adjusted close price - crucial for the method
            "volume": [10000, 11000],
            "tic": ["AAPL", "AAPL"],
        }
    )


@pytest.mark.parametrize(
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
        data_source = DataSource.yahoofinance
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
        data_source = DataSource.yahoofinance
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
        data_source = DataSource.yahoofinance
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
        data_source = DataSource.yahoofinance
        dp = DataProcessor(data_source, start_date, end_date, time_interval)
        price_array, tech_array, turbulence_array = dp.run(
            ticker_input, tech_indicator_list, if_vix
        )


class TestYahooAdjustPrices:
    """Separate test class for _adjust_prices functionality without parametrization."""
    
    def test_adjust_prices_calculates_correctly(
        self, 
        yahoofinance_processor: Yahoofinance,
        sample_raw_data: pd.DataFrame
    ) -> None:
        """Test that prices are adjusted correctly based on adjusted_close/close ratio."""
        # Ensure required columns exist
        assert "adjusted_close" in sample_raw_data.columns
        assert "close" in sample_raw_data.columns

        adjusted_df = yahoofinance_processor._adjust_prices(sample_raw_data.copy())

        # Calculate expected values
        adj_ratio_1 = sample_raw_data.loc[0, "adjusted_close"] / sample_raw_data.loc[0, "close"]
        adj_ratio_2 = sample_raw_data.loc[1, "adjusted_close"] / sample_raw_data.loc[1, "close"]

        expected_data = pd.DataFrame(
            {
                "date": ["2020-01-01", "2020-01-02"],
                "open": [
                    sample_raw_data.loc[0, "open"] * adj_ratio_1,
                    sample_raw_data.loc[1, "open"] * adj_ratio_2,
                ],
                "high": [
                    sample_raw_data.loc[0, "high"] * adj_ratio_1,
                    sample_raw_data.loc[1, "high"] * adj_ratio_2,
                ],
                "low": [
                    sample_raw_data.loc[0, "low"] * adj_ratio_1,
                    sample_raw_data.loc[1, "low"] * adj_ratio_2,
                ],
                "close": [
                    sample_raw_data.loc[0, "close"] * adj_ratio_1,  # close becomes adjusted
                    sample_raw_data.loc[1, "close"] * adj_ratio_2,  # close becomes adjusted
                ],
                "adjusted_close": [
                    sample_raw_data.loc[0, "adjusted_close"],
                    sample_raw_data.loc[1, "adjusted_close"],
                ],
                "volume": [10000, 11000],
                "tic": ["AAPL", "AAPL"],
            }
        )

        # Select only the columns present in the expected output for comparison
        # and ensure the same column order and index
        adjusted_df_compare = adjusted_df[expected_data.columns].reset_index(drop=True)
        expected_data = expected_data.reset_index(drop=True)

        # Use pandas testing utility for robust DataFrame comparison
        assert_frame_equal(adjusted_df_compare, expected_data, check_dtype=False)

    def test_adjust_prices_drops_columns(
        self,
        yahoofinance_processor: Yahoofinance,
        sample_raw_data: pd.DataFrame
    ) -> None:
        """Test that the temporary 'adj' column is dropped after adjustment."""
        # Ensure required columns exist
        assert "adjusted_close" in sample_raw_data.columns
        assert "close" in sample_raw_data.columns

        adjusted_df = yahoofinance_processor._adjust_prices(sample_raw_data.copy())

        # Ensure the temporary 'adj' column is dropped
        assert "adj" not in adjusted_df.columns
        
        # Ensure other essential columns remain
        assert "open" in adjusted_df.columns
        assert "close" in adjusted_df.columns  # Note: This is the *new* adjusted close
        assert "tic" in adjusted_df.columns
        assert "date" in adjusted_df.columns
        assert "volume" in adjusted_df.columns
        assert "adjusted_close" in adjusted_df.columns  # This column is kept in FinRL-Meta

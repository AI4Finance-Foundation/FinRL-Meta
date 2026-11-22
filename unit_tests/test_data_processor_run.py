import numpy as np
import pandas as pd
import pytest

from meta.data_processor import DataProcessor
from meta.data_processors._base import DataSource


def _synthetic_ohlcv(tic="EURUSD", n=60):
    """Offline OHLCV frame in the shape clean_data() expects."""
    dates = pd.date_range("2021-01-01", periods=n, freq="D")
    close = 1.10 + np.sin(np.arange(n) / 6.0) * 0.01
    return pd.DataFrame(
        {
            "tic": tic,
            "time": dates,
            "open": close,
            "high": close + 0.001,
            "low": close - 0.001,
            "close": close,
            "adjusted_close": close,
            "volume": np.full(n, 1000.0),
        }
    )


def _offline_processor(downloads):
    """DataProcessor whose download step is recorded instead of fetched."""
    dp = DataProcessor(
        data_source=DataSource.fxmacrodata,
        start_date="2021-01-01",
        end_date="2021-03-01",
        time_interval="1d",
    )

    def fake_download(ticker_list=None, **kwargs):
        downloads.append(tuple(ticker_list))
        dp.processor.dataframe = _synthetic_ohlcv()
        dp.dataframe = dp.processor.dataframe

    dp.download_data = fake_download
    return dp


@pytest.mark.parametrize("cache", [True, False])
def test_run_returns_arrays(tmp_path, monkeypatch, cache):
    """run() must return the three arrays for every documented argument value."""
    monkeypatch.chdir(tmp_path)
    dp = _offline_processor([])

    price_array, tech_array, risk_array = dp.run(
        ["EURUSD"], ["macd"], if_vix=False, cache=cache
    )

    assert price_array.shape == (60, 1)
    assert tech_array.shape == (60, 1)
    assert risk_array is None


def test_run_cache_reuses_dataset(tmp_path, monkeypatch):
    """A second run() with the same parameters must load the cached dataset."""
    monkeypatch.chdir(tmp_path)

    first_downloads = []
    _offline_processor(first_downloads).run(
        ["EURUSD"], ["macd"], if_vix=False, cache=True
    )
    assert first_downloads == [("EURUSD",)]

    second_downloads = []
    _offline_processor(second_downloads).run(
        ["EURUSD"], ["macd"], if_vix=False, cache=True
    )
    assert second_downloads == []

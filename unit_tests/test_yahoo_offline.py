import pandas as pd
import pytest

from meta.data_processors import yahoofinance
from meta.data_processors._base import DataSource


def test_daily_interval_alias():
    processor = yahoofinance.Yahoofinance(DataSource.yahoofinance, "2024-01-01", "2024-01-03", "1D")
    assert processor.time_interval == "1d"


@pytest.mark.parametrize("supports_proxy", [False, True])
@pytest.mark.parametrize("proxy", [None, "http://proxy"])
def test_download_proxy_compatibility(monkeypatch, tmp_path, supports_proxy, proxy):
    calls = []
    frame = pd.DataFrame({"Open": [1.0], "High": [1.0], "Low": [1.0], "Close": [1.0], "Adj Close": [1.0], "Volume": [1]}, index=pd.DatetimeIndex(["2024-01-02"], name="Date"))
    if supports_proxy:
        def download(ticker, start, end, interval, auto_adjust, proxy=None):
            calls.append((interval, proxy))
            return frame.copy()
    else:
        def download(ticker, start, end, interval, auto_adjust):
            calls.append((interval, None))
            return frame.copy()
    monkeypatch.setattr(yahoofinance.yf, "download", download)
    monkeypatch.setattr(yahoofinance, "calc_time_zone", lambda *args: "America/New_York")
    processor = yahoofinance.Yahoofinance(DataSource.yahoofinance, "2024-01-01", "2024-01-03", "1D")
    processor.download_data(["AAPL"], save_path=str(tmp_path / "data.csv"), proxy=proxy)
    assert calls == [("1d", proxy if supports_proxy else None)]

import numpy as np
import pandas as pd
import pytest

from meta.data_processor import DataProcessor
from meta.data_processors._base import DataSource
from meta.data_processors import yahoofinance


class StubProcessor:
    downloads = 0

    def __init__(self, *args, **kwargs):
        self.dataframe = pd.DataFrame()

    def download_data(self, ticker_list):
        type(self).downloads += 1
        self.dataframe = pd.DataFrame({"close": [1.0, 2.0]})

    def clean_data(self):
        pass

    def add_technical_indicator(self, *args):
        pass

    def df_to_array(self, *args):
        values = self.dataframe["close"].to_numpy().reshape(-1, 1)
        return values, values.copy(), values.copy()


@pytest.mark.parametrize("cache", [False, True])
def test_run_offline(monkeypatch, tmp_path, cache):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(yahoofinance, "Yahoofinance", StubProcessor)
    StubProcessor.downloads = 0
    def make_processor():
        return DataProcessor(DataSource.yahoofinance, "2024-01-01", "2024-01-03", "1d")

    processor = make_processor()
    arrays = processor.run(["AAPL"], [], False, cache=cache)
    assert all(isinstance(array, np.ndarray) for array in arrays)
    assert arrays[0].ravel().tolist() == [1.0, 2.0]
    if cache:
        second = make_processor()
        second_arrays = second.run(["AAPL"], [], False, cache=True)
        assert StubProcessor.downloads == 1
        pd.testing.assert_frame_equal(second.dataframe, processor.dataframe)
        assert second.dataframe is second.processor.dataframe
        np.testing.assert_array_equal(second_arrays[0], arrays[0])


def test_constructor_error_preserves_cause(monkeypatch):
    def fail(*args, **kwargs):
        raise TypeError("underlying failure")
    monkeypatch.setattr(yahoofinance, "Yahoofinance", fail)
    with pytest.raises(ValueError, match="underlying failure") as error:
        DataProcessor(DataSource.yahoofinance, "2024-01-01", "2024-01-03", "1d")
    assert isinstance(error.value.__cause__, TypeError)

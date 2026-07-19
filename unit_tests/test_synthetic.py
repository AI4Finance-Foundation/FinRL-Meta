import numpy as np
import pandas as pd
import pytest

from meta.data_processors.synthetic import generate_synthetic_data
from meta.data_processors.synthetic import SUPPORTED_PATTERNS

EXPECTED_COLUMNS = ["tic", "time", "open", "high", "low", "close", "adjusted_close", "volume"]


class TestSyntheticDataGenerator:
    @pytest.mark.parametrize("pattern", SUPPORTED_PATTERNS)
    def test_generate_returns_expected_schema(self, pattern: str) -> None:
        df = generate_synthetic_data(
            tic_list=["SYN1"],
            start_date="2021-01-01",
            end_date="2021-02-01",
            pattern=pattern,
            seed=1,
        )
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == EXPECTED_COLUMNS
        assert not df.empty
        assert (df["close"] > 0).all()

    def test_multiple_tickers_are_independent(self) -> None:
        df = generate_synthetic_data(
            tic_list=["SYN1", "SYN2"],
            start_date="2021-01-01",
            end_date="2021-02-01",
            pattern="random_walk",
            seed=2,
        )
        assert set(df["tic"].unique()) == {"SYN1", "SYN2"}
        series_1 = df[df.tic == "SYN1"]["close"].to_numpy()
        series_2 = df[df.tic == "SYN2"]["close"].to_numpy()
        assert not np.allclose(series_1, series_2)

    def test_seed_reproducibility(self) -> None:
        kwargs = dict(
            tic_list=["SYN1"],
            start_date="2021-01-01",
            end_date="2021-02-01",
            pattern="random_walk",
            seed=42,
        )
        df1 = generate_synthetic_data(**kwargs)
        df2 = generate_synthetic_data(**kwargs)
        pd.testing.assert_frame_equal(df1, df2)

    def test_trend_pattern_increases_on_average(self) -> None:
        df = generate_synthetic_data(
            tic_list=["SYN1"],
            start_date="2021-01-01",
            end_date="2021-06-01",
            pattern="trend",
            trend=1.0,
            noise_std=0.01,
            seed=3,
        )
        close = df["close"].to_numpy()
        assert close[-1] > close[0]

    def test_unknown_pattern_raises(self) -> None:
        with pytest.raises(ValueError):
            generate_synthetic_data(
                tic_list=["SYN1"],
                start_date="2021-01-01",
                end_date="2021-01-10",
                pattern="not_a_real_pattern",
            )

    def test_unsupported_time_interval_raises(self) -> None:
        with pytest.raises(NotImplementedError):
            generate_synthetic_data(
                tic_list=["SYN1"],
                start_date="2021-01-01",
                end_date="2021-01-10",
                time_interval="1h",
            )

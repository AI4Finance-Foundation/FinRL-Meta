"""Synthetic OHLCV data generator for FinRL-Meta.

Generates idealized synthetic price series (sine wave, trend, random walk)
in FinRL-Meta's standard OHLCV schema. Useful for quickly testing/debugging
trading environments and agents without needing to download real market
data, and for pre-training agents on simple patterns before exposing them
to real, noisier data.

See discussion: https://github.com/AI4Finance-Foundation/FinRL-Meta/issues/70
"""

from typing import List
from typing import Optional

import numpy as np
import pandas as pd

STANDARD_COLUMNS = [
    "tic",
    "time",
    "open",
    "high",
    "low",
    "close",
    "adjusted_close",
    "volume",
]

SUPPORTED_PATTERNS = ["sine", "trend", "random_walk"]


class SyntheticDataGenerator:
    """Generates synthetic OHLCV price series in FinRL-Meta's standard schema:
    ['tic', 'time', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']
    """

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.default_rng(seed)

    def _base_series(
        self,
        n_steps: int,
        pattern: str,
        start_price: float,
        trend: float,
        amplitude: float,
        period: int,
        noise_std: float,
    ) -> np.ndarray:
        t = np.arange(n_steps)
        if pattern == "sine":
            base = start_price + amplitude * np.sin(2 * np.pi * t / period)
        elif pattern == "trend":
            base = start_price + trend * t
        elif pattern == "random_walk":
            steps = self.rng.normal(loc=trend, scale=amplitude, size=n_steps)
            base = start_price + np.cumsum(steps)
        else:
            raise ValueError(
                f"Unknown pattern '{pattern}'. Supported patterns: {SUPPORTED_PATTERNS}"
            )
        noise = self.rng.normal(loc=0.0, scale=noise_std, size=n_steps)
        series = base + noise
        # prices must stay strictly positive
        series = np.clip(series, a_min=0.01, a_max=None)
        return series

    def generate(
        self,
        tic_list: List[str],
        start_date: str,
        end_date: str,
        time_interval: str = "1d",
        pattern: str = "random_walk",
        start_price: float = 100.0,
        trend: float = 0.05,
        amplitude: float = 1.0,
        period: int = 20,
        noise_std: float = 0.5,
    ) -> pd.DataFrame:
        """Generate a synthetic OHLCV DataFrame matching FinRL-Meta's standard schema.

        :param tic_list: list of ticker symbols to generate independent series for
        :param start_date: 'YYYY-MM-DD'
        :param end_date: 'YYYY-MM-DD'
        :param time_interval: currently only '1d' (daily) is supported
        :param pattern: one of 'sine', 'trend', 'random_walk'
        :param start_price: starting close price for each ticker
        :param trend: drift per step (used by 'trend' and 'random_walk')
        :param amplitude: sine wave amplitude, or random_walk step std dev
        :param period: sine wave period, in steps
        :param noise_std: std dev of additive Gaussian noise on top of the base pattern
        :return: DataFrame with columns [tic, time, open, high, low, close, adjusted_close, volume]
        """
        if time_interval != "1d":
            raise NotImplementedError(
                "SyntheticDataGenerator currently only supports time_interval='1d'."
            )

        dates = pd.bdate_range(start=start_date, end=end_date)  # business days
        n_steps = len(dates)
        if n_steps == 0:
            raise ValueError("start_date/end_date produced zero trading days.")

        frames = []
        for tic in tic_list:
            close = self._base_series(
                n_steps, pattern, start_price, trend, amplitude, period, noise_std
            )
            daily_range = np.abs(
                self.rng.normal(loc=0.0, scale=noise_std / 2, size=n_steps)
            )
            high = close + daily_range
            low = np.clip(close - daily_range, a_min=0.01, a_max=None)
            open_ = np.concatenate([[close[0]], close[:-1]])
            volume = self.rng.integers(low=1_000, high=100_000, size=n_steps)

            frames.append(
                pd.DataFrame(
                    {
                        "tic": tic,
                        "time": dates.strftime("%Y-%m-%d"),
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "adjusted_close": close,
                        "volume": volume,
                    }
                )
            )

        df = pd.concat(frames, ignore_index=True)
        df.sort_values(by=["time", "tic"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df[STANDARD_COLUMNS]


def generate_synthetic_data(
    tic_list: List[str],
    start_date: str,
    end_date: str,
    time_interval: str = "1d",
    pattern: str = "random_walk",
    seed: Optional[int] = None,
    **kwargs,
) -> pd.DataFrame:
    """Convenience function wrapping SyntheticDataGenerator.generate()."""
    return SyntheticDataGenerator(seed=seed).generate(
        tic_list=tic_list,
        start_date=start_date,
        end_date=end_date,
        time_interval=time_interval,
        pattern=pattern,
        **kwargs,
    )


if __name__ == "__main__":
    df = generate_synthetic_data(
        tic_list=["SYN1", "SYN2"],
        start_date="2020-01-01",
        end_date="2020-03-01",
        pattern="sine",
        seed=42,
    )
    print(df.head())

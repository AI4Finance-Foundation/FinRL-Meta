import json
import os
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd

from meta.data_processors._base import _Base

FXMACRODATA_BASE_URL = "https://fxmacrodata.com/api/v1"


class Fxmacrodata(_Base):
    """FXMacroData release-calendar processor for macro event features."""

    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        currency: str = "usd",
        limit: int = 100,
        min_tier: int = None,
        api_key: str = None,
        base_url: str = FXMACRODATA_BASE_URL,
        **kwargs,
    ):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        self.currency = currency.lower()
        self.limit = max(1, int(limit))
        self.min_tier = min_tier
        self.api_key = api_key or os.environ.get("FXMACRODATA_API_KEY")
        self.base_url = base_url.rstrip("/")

    def download_data(self, ticker_list=None):
        params = {"limit": self.limit}
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.base_url}/calendar/{self.currency}?{urlencode(params)}"
        with urlopen(url, timeout=30) as response:  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))

        events = payload.get("data", [])
        if self.min_tier is not None:
            events = [
                event
                for event in events
                if int(event.get("market_tier") or 99) <= int(self.min_tier)
            ]

        self.dataframe = pd.DataFrame(events[: self.limit])
        if self.dataframe.empty:
            return

        if "date" in self.dataframe.columns:
            self.dataframe["time"] = pd.to_datetime(
                self.dataframe["date"], errors="coerce"
            )
            self.dataframe = self.dataframe[
                (self.dataframe["time"] >= self.start_date)
                & (self.dataframe["time"] <= self.end_date)
            ]
        if "announcement_datetime" in self.dataframe.columns:
            self.dataframe["announcement_time"] = pd.to_datetime(
                self.dataframe["announcement_datetime"],
                unit="s",
                utc=True,
                errors="coerce",
            )
        self.dataframe["tic"] = self.currency.upper()
        self.dataframe = self.dataframe.sort_values("time").reset_index(drop=True)

    def clean_data(self):
        """Calendar rows are already normalized by the FXMacroData API."""
        return

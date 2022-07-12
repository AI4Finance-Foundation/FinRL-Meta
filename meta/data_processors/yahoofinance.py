"""Reference: https://github.com/AI4Finance-LLC/FinRL"""
from typing import List

import numpy as np
import pandas as pd
import pytz
import yfinance as yf

try:
    import exchange_calendars as tc
except:
    print(
        "Cannot import exchange_calendars.",
        "If you are using python>=3.7, please install it.",
    )
    import trading_calendars as tc

    print("Use trading_calendars instead for yahoofinance processor..")

from meta.config import (
    BINANCE_BASE_URL,
    TIME_ZONE_BERLIN,
    TIME_ZONE_JAKARTA,
    TIME_ZONE_PARIS,
    TIME_ZONE_SELFDEFINED,
    TIME_ZONE_SHANGHAI,
    TIME_ZONE_USEASTERN,
    USE_TIME_ZONE_SELFDEFINED,
)
from meta.data_processors._base import _Base, calc_time_zone


class Yahoofinance(_Base):
    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs
    ):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)

    def download_data(self, ticker_list: List[str]):
        self.time_zone = calc_time_zone(
            ticker_list, TIME_ZONE_SELFDEFINED, USE_TIME_ZONE_SELFDEFINED
        )
        self.dataframe = pd.DataFrame()
        for tic in ticker_list:
            temp_df = yf.download(
                tic,
                start=self.start_date,
                end=self.end_date,
                interval=self.time_interval,
            )
            temp_df["tic"] = tic
            self.dataframe = pd.concat([self.dataframe, temp_df], axis=0, join="outer")
        self.dataframe.reset_index(inplace=True)
        try:
            self.dataframe.columns = [
                "date",
                "open",
                "high",
                "low",
                "close",
                "adjusted_close",
                "volume",
                "tic",
            ]
        except NotImplementedError:
            print("the features are not supported currently")
        self.dataframe["day"] = self.dataframe["date"].dt.dayofweek
        self.dataframe["date"] = self.dataframe.date.apply(
            lambda x: x.strftime("%Y-%m-%d")
        )
        self.dataframe.dropna(inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)
        print("Shape of DataFrame: ", self.dataframe.shape)
        self.dataframe.sort_values(by=["date", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

    def clean_data(self):
        df = self.dataframe.copy()
        df = df.rename(columns={"date": "time"})
        time_interval = self.time_interval
        tic_list = np.unique(df.tic.values)
        trading_days = self.get_trading_days(start=self.start_date, end=self.end_date)
        if time_interval == "1D":
            times = trading_days
        elif time_interval == "1Min":
            times = []
            for day in trading_days:
                current_time = pd.Timestamp(day + " 09:30:00").tz_localize(
                    self.time_zone
                )
                for _ in range(390):
                    times.append(current_time)
                    current_time += pd.Timedelta(minutes=1)
        else:
            raise ValueError(
                "Data clean at given time interval is not supported for YahooFinance data."
            )
        new_df = pd.DataFrame()
        for tic in tic_list:
            print(("Clean data for ") + tic)
            tmp_df = pd.DataFrame(
                columns=[
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjusted_close",
                    "volume",
                ],
                index=times,
            )
            # get data for current ticker
            tic_df = df[df.tic == tic]
            # fill empty DataFrame using orginal data
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["time"]] = tic_df.iloc[i][
                    [
                        "open",
                        "high",
                        "low",
                        "close",
                        "adjusted_close",
                        "volume",
                    ]
                ]

            # if close on start date is NaN, fill data with first valid close
            # and set volume to 0.
            if str(tmp_df.iloc[0]["close"]) == "nan":
                print("NaN data on start date, fill using first valid data.")
                for i in range(tmp_df.shape[0]):
                    if str(tmp_df.iloc[i]["close"]) != "nan":
                        first_valid_close = tmp_df.iloc[i]["close"]
                        first_valid_adjclose = tmp_df.iloc[i]["adjusted_close"]

                tmp_df.iloc[0] = [
                    first_valid_close,
                    first_valid_close,
                    first_valid_close,
                    first_valid_close,
                    first_valid_adjclose,
                    0.0,
                ]

            # fill NaN data with previous close and set volume to 0.
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    previous_adjusted_close = tmp_df.iloc[i - 1]["adjusted_close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_adjusted_close,
                        0.0,
                    ]

            # merge single ticker data to new DataFrame
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = new_df.append(tmp_df)

            print(("Data clean for ") + tic + (" is finished."))

        # reset index and rename columns
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "time"})
        print("Data clean all finished!")
        self.dataframe = new_df

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(
            pd.Timestamp(start, tz=pytz.UTC), pd.Timestamp(end, tz=pytz.UTC)
        )
        return [str(day)[:10] for day in df]

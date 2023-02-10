import copy
import os
import time
import warnings

warnings.filterwarnings("ignore")
from typing import List

import pandas as pd
from tqdm import tqdm

import stockstats
import talib
from meta.data_processors._base import _Base

import akshare as ak  # pip install akshare


class Akshare(_Base):
    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ):
        start_date = self.transfer_date(start_date)
        end_date = self.transfer_date(end_date)

        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)

        if "adj" in kwargs.keys():
            self.adj = kwargs["adj"]
            print(f"Using {self.adj} method.")
        else:
            self.adj = ""

        if "period" in kwargs.keys():
            self.period = kwargs["period"]
        else:
            self.period = "daily"

    def get_data(self, id) -> pd.DataFrame:
        return ak.stock_zh_a_hist(
            symbol=id,
            period=self.time_interval,
            start_date=self.start_date,
            end_date=self.end_date,
            adjust=self.adj,
        )

    def download_data(
        self, ticker_list: List[str], save_path: str = "./data/dataset.csv"
    ):
        """
        `pd.DataFrame`
            7 columns: A tick symbol, time, open, high, low, close and volume
            for the specified stock ticker
        """
        assert self.time_interval in [
            "daily",
            "weekly",
            "monthly",
        ], "Not supported currently"

        self.ticker_list = ticker_list

        self.dataframe = pd.DataFrame()
        for i in tqdm(ticker_list, total=len(ticker_list)):
            nonstandard_id = self.transfer_standard_ticker_to_nonstandard(i)
            df_temp = self.get_data(nonstandard_id)
            df_temp["tic"] = i
            # df_temp = self.get_data(i)
            self.dataframe = pd.concat([self.dataframe, df_temp])
            # self.dataframe = self.dataframe.append(df_temp)
            # print("{} ok".format(i))
            time.sleep(0.25)

        self.dataframe.columns = [
            "time",
            "open",
            "close",
            "high",
            "low",
            "volume",
            "amount",
            "amplitude",
            "pct_chg",
            "change",
            "turnover",
            "tic",
        ]

        self.dataframe.sort_values(by=["time", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        self.dataframe = self.dataframe[
            ["tic", "time", "open", "high", "low", "close", "volume"]
        ]
        # self.dataframe.loc[:, 'tic'] = pd.DataFrame((self.dataframe['tic'].tolist()))
        self.dataframe["time"] = pd.to_datetime(
            self.dataframe["time"], format="%Y-%m-%d"
        )
        self.dataframe["day"] = self.dataframe["time"].dt.dayofweek
        self.dataframe["time"] = self.dataframe.time.apply(
            lambda x: x.strftime("%Y-%m-%d")
        )

        self.dataframe.dropna(inplace=True)
        self.dataframe.sort_values(by=["time", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        self.save_data(save_path)

        print(
            f"Download complete! Dataset saved to {save_path}. \nShape of DataFrame: {self.dataframe.shape}"
        )

    def data_split(self, df, start, end, target_date_col="time"):
        """
        split the dataset into training or testing using time
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    def transfer_standard_ticker_to_nonstandard(self, ticker: str) -> str:
        # "600000.XSHG" -> "600000"
        # "000612.XSHE" -> "000612"
        # "600000.SH" -> "600000"
        # "000612.SZ" -> "000612"
        if "." in ticker:
            n, alpha = ticker.split(".")
            # assert alpha in ["XSHG", "XSHE"], "Wrong alpha"
        return n

    def transfer_date(self, time: str) -> str:
        if "-" in time:
            time = "".join(time.split("-"))
        elif "." in time:
            time = "".join(time.split("."))
        elif "/" in time:
            time = "".join(time.split("/"))
        return time

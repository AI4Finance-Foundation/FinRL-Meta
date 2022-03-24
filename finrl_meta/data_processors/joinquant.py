import copy
import datetime
import os
from typing import List

import jqdatasdk as jq
import numpy as np

from finrl_meta.data_processors._base import _Base

class Joinquant(_Base):
    def __init__(self, data_source: str, start_date: str, end_date: str, time_interval: str, **kwargs):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        if 'username' in kwargs.keys() and 'password' in kwargs.keys():
            jq.auth(kwargs['username'], kwargs['password'])

    def download_data(self, ticker_list: List[str]):
        # joinquant supports: '1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M'。'1w' denotes one week，‘1M' denotes one month。
        count = len(self.get_trading_days(self.start_date, self.end_date))
        df = jq.get_bars(
            security=ticker_list,
            count=count,
            unit=self.time_interval,
            fields=["date", "open", "high", "low", "close", "volume"],
            end_dt=self.end_date,
        )
        df = df.reset_index().rename(columns={'level_0': 'tic'})
        self.dataframe = df


    def preprocess(df, stock_list):
        n = len(stock_list)
        N = df.shape[0]
        assert N % n == 0
        d = int(N / n)
        stock1_ary = df.iloc[0:d, 1:].values
        temp_ary = stock1_ary
        for j in range(1, n):
            stocki_ary = df.iloc[j * d:(j + 1) * d, 1:].values
            temp_ary = np.hstack((temp_ary, stocki_ary))
        return temp_ary

    # start_day: str
    # end_day: str
    # output: list of str_of_trade_day, e.g., ['2021-09-01', '2021-09-02']
    def get_trading_days(self, start_day: str, end_day: str) -> List[str]:
        dates = jq.get_trade_days(start_day, end_day)
        str_dates = []
        for d in dates:
            tmp = datetime.date.strftime(d, "%Y-%m-%d")
            str_dates.append(tmp)
        # str_dates = [date2str(dt) for dt in dates]
        return str_dates



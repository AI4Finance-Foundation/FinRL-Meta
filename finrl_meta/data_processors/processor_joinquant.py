import jqdatasdk as jq
import pandas as pd
import numpy as np
import copy
import os
from typing import List
from finrl_meta.data_processors.func import calc_all_filenames, date2str, remove_all_files
from finrl_meta.data_processors.func import add_hyphen_for_date
from finrl_meta.data_processors.func import remove_hyphen_for_date
# from basic_processor import BasicProcessor
from finrl_meta.data_processors.basic_processor import BasicProcessor


class JoinquantProcessor(BasicProcessor):
    def __init__(self, data_source: str, **kwargs):
        BasicProcessor.__init__(self, data_source, **kwargs)
        if 'username' in kwargs.keys() and 'password' in kwargs.keys():
            jq.auth(kwargs['username'], kwargs['password'])

    def download_data(self, ticker_list: List[str], start_date: str, end_date: str, time_interval: str
                      ) -> pd.DataFrame:
        unit = None
        # joinquant supports: '1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M'。'1w' denotes one week，‘1M' denotes one month。
        if time_interval == '1D':
            unit = '1d'
        elif time_interval == '1Min':
            unit = '1m'
        else:
            raise ValueError('not supported currently')
        count = len(self.calc_trade_days_by_joinquant(start_date, end_date))
        df = jq.get_bars(
            security=ticker_list,
            count=count,
            unit=unit,
            fields=["date", "open", "high", "low", "close", "volume"],
            end_dt=end_date,
        )

        return df

    def data_fetch(self,stock_list, num, unit, end_dt):
        df = jq.get_bars(security=stock_list, count=num, unit=unit, 
                         fields=['date','open','high','low','close','volume'],
                         end_dt=end_dt)
        return df
    def preprocess(df, stock_list):
        n = len(stock_list)
        N = df.shape[0]
        assert N%n == 0
        d = int(N/n)
        stock1_ary = df.iloc[0:d,1:].values
        temp_ary = stock1_ary
        for j in range(1, n):
            stocki_ary = df.iloc[j*d:(j+1)*d,1:].values
            temp_ary = np.hstack((temp_ary,stocki_ary))
        return temp_ary

    # start_day: str
    # end_day: str
    # output: list of str_of_trade_day, e.g., ['2021-09-01', '2021-09-02']
    def calc_trade_days_by_joinquant(self, start_day: str, end_day: str) -> List[str]:
        dates = jq.get_trade_days(start_day, end_day)
        str_dates = [date2str(dt) for dt in dates]
        return str_dates

    # start_day: str
    # end_day: str
    # output: list of dataframes, e.g., [df1, df2]
    def read_data_from_csv(self, path_of_data, start_day, end_day):
        datasets = []
        selected_days = self.calc_trade_days_by_joinquant(start_day, end_day)
        filenames = calc_all_filenames(path_of_data)
        for filename in filenames:
            dataset_orig = pd.read_csv(filename)
            dataset = copy.deepcopy(dataset_orig)
            days = dataset.iloc[:, 0].values.tolist()
            indices_of_rows_to_drop = [d for d in days if d not in selected_days]
            dataset.drop(index=indices_of_rows_to_drop, inplace=True)
            datasets.append(dataset)
        return datasets

    # start_day: str
    # end_day: str
    # read_data_from_local: if it is true, read_data_from_csv, and fetch data from joinquant otherwise.
    # output: list of dataframes, e.g., [df1, df2]
    def download_data_for_stocks(
            self, stocknames, start_day, end_day, read_data_from_local, path_of_data
    ):
        assert read_data_from_local in [0, 1]
        if read_data_from_local == 1:
            remove = 0
        else:
            remove = 1
        remove_all_files(remove, path_of_data)
        dfs = []
        if read_data_from_local == 1:
            dfs = self.read_data_from_csv(path_of_data, start_day, end_day)
        else:
            if os.path.exists(path_of_data) is False:
                os.makedirs(path_of_data)
            for stockname in stocknames:
                df = jq.get_price(
                    stockname,
                    start_date=start_day,
                    end_date=end_day,
                    frequency="daily",
                    fields=["open", "close", "high", "low", "volume"],
                )
                dfs.append(df)
                df.to_csv(path_of_data + "/" + stockname + ".csv", float_format="%.4f")
        return dfs



import os
import pickle
from typing import List

import numpy as np
import pandas as pd
import yaml

class DataProcessor:
    def __init__(
        self,
        data_source: str,
        start_date: str = None,
        end_date: str = None,
        time_interval: str = None,
        **kwargs,
    ):
        self.data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.dataframe = pd.DataFrame()

        if self.data_source == "akshare":
            from meta.data_processors.akshare import Akshare

            processor_dict = {self.data_source: Akshare}
        elif self.data_source == "alpaca":
            from meta.data_processors.alpaca import Alpaca

            processor_dict = {self.data_source: Alpaca}
        elif self.data_source == "alphavantage":
            from meta.data_processors.alphavantage import Alphavantage

            processor_dict = {self.data_source: Alphavantage}
        elif self.data_source == "baostock":
            from meta.data_processors.baostock import Baostock

            processor_dict = {self.data_source: Baostock}
        elif self.data_source == "binance":
            from meta.data_processors.binance import Binance

            processor_dict = {self.data_source: Binance}
        elif self.data_source == "ccxt":
            from meta.data_processors.ccxt import Ccxt

            processor_dict = {self.data_source: Ccxt}
        elif self.data_source == "iexcloud":
            from meta.data_processors.iexcloud import Iexcloud

            processor_dict = {self.data_source: Iexcloud}
        elif self.data_source == "joinquant":
            from meta.data_processors.joinquant import Joinquant

            processor_dict = {self.data_source: Joinquant}
        elif self.data_source == "quandl":
            from meta.data_processors.quandl import Quandl

            processor_dict = {self.data_source: Quandl}
        elif self.data_source == "quantconnect":
            from meta.data_processors.quantconnect import Quantconnect

            processor_dict = {self.data_source: Quantconnect}
        elif self.data_source == "ricequant":
            from meta.data_processors.ricequant import Ricequant

            processor_dict = {self.data_source: Ricequant}
        elif self.data_source == "tushare":
            from meta.data_processors.tushare import Tushare

            processor_dict = {self.data_source: Tushare}
            with open("./configs/tushare.yaml", "r", encoding="utf-8") as ymlfile:
                self.cfg = yaml.safe_load(ymlfile)
        elif self.data_source == "wrds":
            from meta.data_processors.wrds import Wrds

            processor_dict = {self.data_source: Wrds}
        elif self.data_source == "yahoofinance":
            from meta.data_processors.yahoofinance import Yahoofinance

            processor_dict = {self.data_source: Yahoofinance}
        else:
            print(f"{self.data_source} is NOT supported yet.")

        try:
            self.processor = processor_dict.get(self.data_source)(
                data_source, start_date, end_date, time_interval, **kwargs
            )
            print(f"{self.data_source} successfully connected")
        except:
            raise ValueError(
                f"Please input correct account info for {self.data_source}!"
            )

    def download_share_list(self, save_path=None, filter_item=None, filter_value=None):
        self.processor.download_share_list(self.cfg, save_path, filter_item, filter_value)
        self.sharelist_df = self.processor.sharelist_df
        self.sharelist = self.sharelist_df['stock_code_full'].unique()

    def load_share_list(self, csv_path, filter_item=None, filter_value=None):
        self.sharelist_df = pd.read_csv(csv_path)
        if filter_item is not None:
            assert(len(filter_item) == len(filter_value))
            for col, val in zip(filter_item, filter_value):
                self.sharelist_df = self.sharelist_df[self.sharelist_df[col] == val]
        self.sharelist = self.sharelist_df['stock_code_full'].unique()


    def download_market_daily(self, share_list, start_date, end_date, adj, save_folder=None):
        self.processor.download_market_daily(self.cfg, share_list, start_date, end_date, adj, save_folder)
        self.dataframe = self.processor.dataframe

    def download_data(self, ticker_list):
        self.processor.download_data(ticker_list=ticker_list)
        self.dataframe = self.processor.dataframe

    def clean_data(self):
        self.processor.dataframe = self.dataframe
        self.processor.clean_data()
        self.dataframe = self.processor.dataframe

    def add_technical_indicator(
        self, tech_indicator_list: List[str], select_stockstats_talib: int = 0
    ):
        self.tech_indicator_list = tech_indicator_list
        self.processor.add_technical_indicator(
            tech_indicator_list, select_stockstats_talib
        )
        self.dataframe = self.processor.dataframe

    def add_turbulence(self):
        self.processor.add_turbulence()
        self.dataframe = self.processor.dataframe

    def add_vix(self):
        self.processor.add_vix()
        self.dataframe = self.processor.dataframe

    def df_to_array(self, if_vix: bool) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            self.tech_indicator_list, if_vix
        )
        # fill nan with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return price_array, tech_array, turbulence_array

    def data_split(self, df, start, end, target_date_col="time"):
        """
        split the dataset into training or testing using date
        :param data: (df) pandas dataframe, start, end
        :return: (df) pandas dataframe
        """
        data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
        data = data.sort_values([target_date_col, "tic"], ignore_index=True)
        data.index = data[target_date_col].factorize()[0]
        return data

    def fillna(self):
        self.processor.dataframe = self.dataframe
        self.processor.fillna()
        self.dataframe = self.processor.dataframe

    def run(
        self,
        ticker_list: str,
        technical_indicator_list: List[str],
        if_vix: bool,
        cache: bool = False,
        select_stockstats_talib: int = 0,
    ):
        if self.time_interval == "1s" and self.data_source != "binance":
            raise ValueError(
                "Currently 1s interval data is only supported with 'binance' as data source"
            )

        cache_filename = (
            "_".join(
                ticker_list
                + [
                    self.data_source,
                    self.start_date,
                    self.end_date,
                    self.time_interval,
                ]
            )
            + ".pickle"
        )
        cache_dir = "./cache"
        cache_path = os.path.join(cache_dir, cache_filename)

        if cache and os.path.isfile(cache_path):
            print(f"Using cached file {cache_path}")
            self.tech_indicator_list = technical_indicator_list
            with open(cache_path, "rb") as handle:
                self.processor.dataframe = pickle.load(handle)
        else:
            self.download_data(ticker_list)
            self.clean_data()
            if cache:
                if not os.path.exists(cache_dir):
                    os.mkdir(cache_dir)
                with open(cache_path, "wb") as handle:
                    pickle.dump(
                        self.dataframe,
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

        self.add_technical_indicator(technical_indicator_list, select_stockstats_talib)
        if if_vix:
            self.add_vix()
        price_array, tech_array, turbulence_array = self.df_to_array(if_vix)
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return price_array, tech_array, turbulence_array


def test_joinquant():
    # TRADE_START_DATE = "2019-09-01"
    TRADE_START_DATE = "2020-09-01"
    TRADE_END_DATE = "2021-09-11"

    # supported time interval: '1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M'
    TIME_INTERVAL = "1d"
    TECHNICAL_INDICATOR = [
        "macd",
        "boll_ub",
        "boll_lb",
        "rsi_30",
        "dx_30",
        "close_30_sma",
        "close_60_sma",
    ]

    kwargs = {"username": "xxx", "password": "xxx"}
    p = DataProcessor(
        data_source="joinquant",
        start_date=TRADE_START_DATE,
        end_date=TRADE_END_DATE,
        time_interval=TIME_INTERVAL,
        **kwargs,
    )

    ticker_list = ["000612.XSHE", "601808.XSHG"]

    p.download_data(ticker_list=ticker_list)

    p.clean_data()
    p.add_turbulence()
    p.add_technical_indicator(TECHNICAL_INDICATOR)
    p.add_vix()

    price_array, tech_array, turbulence_array = p.run(
        ticker_list, TECHNICAL_INDICATOR, if_vix=False, cache=True
    )
    pass


# if __name__ == "__main__":
#     # test_joinquant()
#     # test_binance()
#     # test_yahoofinance()
#     test_baostock()
#     # test_quandl()

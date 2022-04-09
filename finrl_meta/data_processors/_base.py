from typing import List

import numpy as np
import pandas as pd
import stockstats
import talib
import copy

import os
import urllib
import zipfile
from datetime import *
from pathlib import Path
from typing import List

from finrl_meta.config import (
TIME_ZONE_SHANGHAI,
TIME_ZONE_USEASTERN,
TIME_ZONE_PARIS,
TIME_ZONE_BERLIN,
TIME_ZONE_JAKARTA,
TIME_ZONE_SELFDEFINED,
USE_TIME_ZONE_SELFDEFINED,
BINANCE_BASE_URL,
)

from finrl_meta.config_tickers import (
HSI_50_TICKER,
SSE_50_TICKER,
CSI_300_TICKER,
DOW_30_TICKER,
NAS_100_TICKER,
SP_500_TICKER,
LQ45_TICKER,
CAC_40_TICKER,
DAX_30_TICKER,
TECDAX_TICKER,
MDAX_50_TICKER,
SDAX_50_TICKER,
)

class _Base:
    def __init__(self, data_source: str, start_date: str, end_date: str, time_interval: str, **kwargs):
        self.data_source: str = data_source
        self.start_date: str = start_date
        self.end_date: str = end_date
        self.time_interval: str = time_interval  # standard time_interval
        # transferred_time_interval will be supported in the future.
        # self.nonstandard_time_interval: str = self.calc_nonstandard_time_interval()  # transferred time_interval of this processor
        self.time_zone: str = ""
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.dictnumpy: dict = {}  # e.g., self.dictnumpy["open"] = np.array([1, 2, 3]), self.dictnumpy["close"] = np.array([1, 2, 3])

    def download_data(self, ticker_list: List[str]):
        pass

    def clean_data(self):
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={'date': 'time'}, inplace=True)
        if "datetime" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={'datetime': 'time'}, inplace=True)
        if self.data_source == "ccxt":
            self.dataframe.rename(columns={'index': 'time'}, inplace=True)

        if self.data_source == 'ricequant':
            ''' RiceQuant data is already cleaned, we only need to transform data format here.
                No need for filling NaN data'''
            self.dataframe.rename(columns={'order_book_id': 'tic'}, inplace=True)
            # raw df uses multi-index (tic,time), reset it to single index (time)
            self.dataframe.reset_index(level=[0, 1], inplace=True)
            # check if there is NaN values
            assert not self.dataframe.isnull().values.any()
        elif self.data_source == 'baostock':
            self.dataframe.rename(columns={'code': 'tic'}, inplace=True)

        self.dataframe.dropna(inplace=True)
        # adjusted_close: adjusted close price
        if 'adjusted_close' not in self.dataframe.columns.values.tolist():
            self.dataframe['adjusted_close'] = self.dataframe['close']
        self.dataframe.sort_values(by=['time', 'tic'], inplace=True)
        self.dataframe = self.dataframe[['tic', 'time', 'open', 'high', 'low', 'close', 'adjusted_close', 'volume']]

    def get_trading_days(self, start: str, end: str) -> List[str]:
        if self.data_source in ["binance", "ccxt", "quantconnect", "ricequant", "tushare"]:
            print(f"Calculate get_trading_days not supported for {self.data_source} yet.")
            return None

    # select_stockstats_talib: 0 (stockstats, default), or 1 (use talib). Users can choose the method.
    def add_technical_indicator(self, tech_indicator_list: List[str], select_stockstats_talib: int = 0):
        """
        calculate technical indicators
        use stockstats/talib package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={'date': 'time'}, inplace=True)

        if self.data_source == "ccxt":
            self.dataframe.rename(columns={'index': 'time'}, inplace=True)

        self.dataframe.reset_index(drop=False, inplace=True)
        if "level_1" in self.dataframe.columns:
            self.dataframe.drop(columns=["level_1"], inplace=True)
        if "level_0" in self.dataframe.columns and "tic" not in self.dataframe.columns:
            self.dataframe.rename(columns={"level_0": "tic"}, inplace=True)
        assert select_stockstats_talib in {0, 1}
        print("tech_indicator_list: ", tech_indicator_list)
        if select_stockstats_talib == 0:  # use stockstats
            stock = stockstats.StockDataFrame.retype(self.dataframe)
            unique_ticker = stock.tic.unique()
            for indicator in tech_indicator_list:
                print("indicator: ", indicator)
                indicator_df = pd.DataFrame()
                for i in range(len(unique_ticker)):
                    try:
                        temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                        temp_indicator = pd.DataFrame(temp_indicator)
                        temp_indicator["tic"] = unique_ticker[i]
                        temp_indicator["time"] = self.dataframe[self.dataframe.tic == unique_ticker[i]][
                            "time"
                        ].to_list()
                        indicator_df = indicator_df.append(
                            temp_indicator, ignore_index=True
                        )
                    except Exception as e:
                        print(e)
                if not indicator_df.empty:
                    self.dataframe = self.dataframe.merge(
                        indicator_df[["tic", "time", indicator]], on=["tic", "time"], how="left"
                    )
        else:  # use talib
            final_df = pd.DataFrame()
            for i in self.dataframe.tic.unique():
                tic_df = self.dataframe[self.dataframe.tic == i]
                tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = talib.MACD(tic_df['close'], fastperiod=12,
                                                                                  slowperiod=26, signalperiod=9)
                tic_df['rsi'] = talib.RSI(tic_df['close'], timeperiod=14)
                tic_df['cci'] = talib.CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
                tic_df['dx'] = talib.DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
                final_df = final_df.append(tic_df)
            self.dataframe = final_df

        self.dataframe.sort_values(by=["time", "tic"], inplace=True)
        time_to_drop = self.dataframe[self.dataframe.isna().any(axis=1)].time.unique()
        self.dataframe = self.dataframe[~self.dataframe.time.isin(time_to_drop)]
        print("Succesfully add technical indicators")

    def add_turbulence(self):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        # df = data.copy()
        # turbulence_index = self.calculate_turbulence(df)
        # df = df.merge(turbulence_index, on="time")
        # df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        # return df
        if self.data_source in ["binance", "ccxt", "iexcloud", "joinquant", "quantconnect"]:
            print(f"Turbulence not supported for {self.data_source} yet. Return original DataFrame.")
        if self.data_source in ["alpaca", "ricequant", "tushare", "wrds", "yahoofinance"]:
            turbulence_index = self.calculate_turbulence()
            self.dataframe = self.dataframe.merge(turbulence_index, on="time")
            self.dataframe.sort_values(["time", "tic"], inplace=True).reset_index(drop=True, inplace=True)

    def calculate_turbulence(self, time_period: int = 252) -> pd.DataFrame:
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df_price_pivot = self.dataframe.pivot(index="time", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = self.dataframe['time'].unique()
        # start after a year
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
                ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                                  hist_price.isna().sum().min():
                                  ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = (current_price[list(filtered_hist_price)] - np.mean(
                filtered_hist_price, axis=0
            ))
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                # avoid large outlier because of the calculation just begins: else turbulence_temp = 0
                turbulence_temp = temp[0][0] if count > 2 else 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"time": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_vix(self):
        """
        add vix from processors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        if self.data_source in ['binance', 'ccxt', 'iexcloud', 'joinquant', 'quantconnect', 'ricequant', 'tushare']:
            print(f'VIX is not applicable for {self.data_source}. Return original DataFrame')
            return

        # if self.data_source == 'yahoofinance':
        #     df = data.copy()
        #     df_vix = self.download_data(
        #         start_date=df.time.min(),
        #         end_date=df.time.max(),
        #         ticker_list=["^VIX"],
        #         time_interval=self.time_interval,
        #     )
        #     df_vix = self.clean_data(df_vix)
        #     vix = df_vix[["time", "adjusted_close"]]
        #     vix.columns = ["time", "vix"]
        #
        #     df = df.merge(vix, on="time")
        #     df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        # elif self.data_source == 'alpaca':
        #     vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        #     cleaned_vix = self.clean_data(vix_df)
        #     vix = cleaned_vix[["time", "close"]]
        #     vix = vix.rename(columns={"close": "VIXY"})
        #
        #     df = data.copy()
        #     df = df.merge(vix, on="time")
        #     df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        # elif self.data_source == 'wrds':
        #     vix_df = self.download_data(['vix'], self.start, self.end_date, self.time_interval)
        #     cleaned_vix = self.clean_data(vix_df)
        #     vix = cleaned_vix[['date', 'close']]
        #
        #     df = data.copy()
        #     df = df.merge(vix, on="date")
        #     df = df.sort_values(["date", "tic"]).reset_index(drop=True)

        if self.data_source == 'yahoofinance':
            ticker = "^VIX"
        elif self.data_source == 'alpaca':
            ticker = "VIXY"
        elif self.data_source == 'wrds':
            ticker = "vix"
        else:
            return
        df = self.dataframe.copy()
        self.dataframe = [ticker]
        self.download_data(self.start, self.end, self.time_interval)
        self.clean_data()
        # vix = cleaned_vix[["time", "close"]]
        # vix = vix.rename(columns={"close": "VIXY"})
        cleaned_vix = self.dataframe.rename(columns={ticker: "vix"})

        df = df.merge(cleaned_vix, on="time")
        df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        self.dataframe = df

    def df_to_array(self, tech_indicator_list: List[str], if_vix: bool):
        unique_ticker = self.dataframe.tic.unique()
        price_array = np.column_stack([self.dataframe[self.dataframe.tic == tic].close for tic in unique_ticker])
        common_tech_indicator_list = [i for i in tech_indicator_list if i in self.dataframe.columns.values.tolist()]
        tech_array = np.hstack([self.dataframe.loc[(self.dataframe.tic == tic), common_tech_indicator_list] for tic in unique_ticker])
        if if_vix:
            risk_array = np.column_stack([self.dataframe[self.dataframe.tic == tic].vix for tic in unique_ticker])
        else:
            risk_array = np.column_stack(
                [self.dataframe[self.dataframe.tic == tic].turbulence for tic in unique_ticker]) if "turbulence" in self.dataframe.columns else None
        print("Successfully transformed into array")
        return price_array, tech_array, risk_array

    # standard_time_interval  s: second, m: minute, h: hour, d: day, w: week, M: month, q: quarter, y: year
    # output time_interval of the processor
    def calc_nonstandard_time_interval(self) -> str:
        if self.data_source == "alpaca":
            pass
        elif self.data_source == "baostock":
            # nonstandard_time_interval: 默认为d，日k线；d=日k线、w=周、m=月、5=5分钟、15=15分钟、30=30分钟、60=60分钟k线数据，不区分大小写；指数没有分钟线数据；周线每周最后一个交易日才可以获取，月线每月最后一个交易日才可以获取。
            pass
            time_intervals = ["5m", "15m", "30m", "60m", "1d", "1w", "1M"]
            assert self.time_interval in time_intervals, "This time interval is not supported. Supported time intervals: " + ",".join(time_intervals)
            if "d" in self.time_interval or "w" in self.time_interval or "M" in self.time_interval:
                return self.time_interval[-1:].lower()
            elif "m" in self.time_interval:
                return self.time_interval[:-1]
        elif self.data_source == "binance":
            # nonstandard_time_interval: 1m,3m,5m,15m,30m,1h,2h,4h,6h,8h,12h,1d,3d,1w,1M
            time_intervals = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"]
            assert self.time_interval in time_intervals, "This time interval is not supported. Supported time intervals: " + ",".join(time_intervals)
            return self.time_interval
        elif self.data_source == "ccxt":
            pass
        elif self.data_source == "iexcloud":
            time_intervals = ["1d"]
            assert self.time_interval in time_intervals, "This time interval is not supported. Supported time intervals: " + ",".join(time_intervals)
            return self.time_interval.upper()
        elif self.data_source == "joinquant":
            # '1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M'
            time_intervals = ["1m", "5m", "15m", "30m", "60m", "120m", "1d", "1w", "1M"]
            assert self.time_interval in time_intervals, "This time interval is not supported. Supported time intervals: " + ",".join(time_intervals)
            return self.time_interval
        elif self.data_source == "quantconnect":
            pass
        elif self.data_source == "ricequant":
            #  nonstandard_time_interval: 'd' - 天，'w' - 周，'m' - 月， 'q' - 季，'y' - 年
            time_intervals = ["d", "w", "M", "q", "y"]
            assert self.time_interval[-1] in time_intervals, "This time interval is not supported. Supported time intervals: " + ",".join(time_intervals)
            if "M" in self.time_interval:
                return self.time_interval.lower()
            else:
                return self.time_interval
        elif self.data_source == "tushare":
            # 分钟频度包括1分、5、15、30、60分数据. Not support currently. 
            # time_intervals = ["1m", "5m", "15m", "30m", "60m", "1d"]
            time_intervals = ["1d"]
            assert self.time_interval in time_intervals, "This time interval is not supported. Supported time intervals: " + ",".join(time_intervals)
            return self.time_interval
        elif self.data_source == "wrds":
            pass
        elif self.data_source == "yahoofinance":
            # nonstandard_time_interval: ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d","1wk", "1mo", "3mo"]
            time_intervals = ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h", "1d", "5d", "1w", "1M", "3M"]
            assert self.time_interval in time_intervals, "This time interval is not supported. Supported time intervals: " + ",".join(time_intervals)
            if "w" in self.time_interval:
                return self.time_interval + "k"
            elif "M" in self.time_interval:
                return self.time_interval[: -1] + "mo"
            else:
                return self.time_interval
        else:
            raise ValueError(f"Not support transfer_standard_time_interval for {self.data_source}")


    # "600000.XSHG" -> "sh.600000"
    # "000612.XSHE" -> "sz.000612"
    def transfer_standard_ticker_to_nonstandard(self, ticker: str) -> str:
        return ticker






def calc_time_zone(ticker_list: List[str], time_zone_selfdefined: str, use_time_zone_selfdefined: int) -> str:
    if use_time_zone_selfdefined == 1:
        time_zone = time_zone_selfdefined
    elif ticker_list in [HSI_50_TICKER, SSE_50_TICKER, CSI_300_TICKER]:
        time_zone = TIME_ZONE_SHANGHAI
    elif ticker_list in [DOW_30_TICKER, NAS_100_TICKER, SP_500_TICKER]:
        time_zone = TIME_ZONE_USEASTERN
    elif ticker_list == CAC_40_TICKER:
        time_zone = TIME_ZONE_PARIS
    elif ticker_list in [DAX_30_TICKER, TECDAX_TICKER, MDAX_50_TICKER, SDAX_50_TICKER]:
        time_zone = TIME_ZONE_BERLIN
    elif ticker_list == LQ45_TICKER:
        time_zone = TIME_ZONE_JAKARTA
    else:
        raise ValueError("Time zone is wrong.")
    return time_zone


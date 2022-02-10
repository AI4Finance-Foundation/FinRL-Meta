from typing import List

import numpy as np
import pandas as pd
import stockstats
import talib

class BasicProcessor:
    def __init__(self, data_source: str, start_date, end_date, time_interval, **kwargs):

        assert data_source in {
            "alpaca",
            "ccxt",
            "binance",
            "iexcloud",
            "joinquant",
            "quantconnect",
            "ricequant",
            "wrds",
            "yahoofinance",
            "tusharepro",
        }, "Data source input is NOT supported yet."
        self.data_source: str = data_source
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.time_zone: str = ""
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.dictnumpy: dict = {}

    def download_data(self, ticker_list: List[str]):
        pass

    def clean_data(self):
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={'date': 'time'}, inplace=True)
        if "datetime" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={'datetime': 'time'}, inplace=True)
        if self.data_source == "ccxt":
            self.dataframe.rename(columns={'index': 'time'}, inplace=True)
        elif self.data_source == 'ricequant':
            ''' RiceQuant data is already cleaned, we only need to transform data format here.
                No need for filling NaN data'''
            self.dataframe.rename(columns={'order_book_id': 'tic'}, inplace=True)
            # raw df uses multi-index (tic,time), reset it to single index (time)
            self.dataframe.reset_index(level=[0, 1], inplace=True)
            # check if there is NaN values
            assert not self.dataframe.isnull().values.any()
        self.dataframe.dropna(inplace=True)
        # adj_close: adjusted close price
        if 'adj_close' not in self.dataframe.columns.values.tolist():
            self.dataframe['adj_close'] = self.dataframe['close']
        self.dataframe.sort_values(by=['time', 'tic'], inplace=True)
        self.dataframe = self.dataframe[['tic', 'time', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]

    def get_trading_days(self, start: str, end: str) -> List[str]:
        if self.data_source in ["binance", "ccxt", "quantconnect", "ricequant", "tusharepro"]:
            print(f"Calculate get_trading_days not supported for {self.data_source} yet.")
            return None

    # use_stockstats_or_talib: 0 (stockstats, default), or 1 (use talib). Users can choose the method.
    def add_technical_indicator(self, tech_indicator_list: List[str], use_stockstats_or_talib: int = 0):
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
        assert use_stockstats_or_talib in {0, 1}
        if use_stockstats_or_talib == 0:  # use stockstats
            stock = stockstats.StockDataFrame.retype(self.dataframe)
            unique_ticker = stock.tic.unique()
            for indicator in tech_indicator_list:
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
        if self.data_source in ["alpaca", "ricequant", "tusharepro", "wrds", "yahoofinance"]:
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
        if self.data_source in ['binance', 'ccxt', 'iexcloud', 'joinquant', 'quantconnect', 'ricequant', 'tusharepro']:
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
        #     vix = df_vix[["time", "adj_close"]]
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

    def df_to_array(self, tech_indicator_list: list, if_vix: bool):
        unique_ticker = self.dataframe.tic.unique()
        price_array = np.column_stack([self.dataframe[self.dataframe.tic == tic].close for tic in unique_ticker])
        tech_array = np.hstack([self.dataframe.loc[(self.dataframe.tic == tic), tech_indicator_list] for tic in unique_ticker])
        if if_vix:
            risk_array = np.column_stack([self.dataframe[self.dataframe.tic == tic].vix for tic in unique_ticker])
        else:
            risk_array = np.column_stack(
                [self.dataframe[self.dataframe.tic == tic].turbulence for tic in unique_ticker]) if "turbulence" in self.dataframe.columns else None
        print("Successfully transformed into array")
        return price_array, tech_array, risk_array

    # standard_time_interval  s: second, m: minute, h: hour, d: day, w: week, M: month
    # output time_interval of the processor
    def transfer_standard_time_interval(self) -> str:
        if self.time_interval == "alpaca":
            pass
        elif self.time_interval == "binance":
            pass
        elif self.time_interval == "ccxt":
            pass
        elif self.time_interval == "iexcloud":
            pass
        elif self.time_interval == "joinquant":
            assert self.time_interval in {'1m', '5m', '15m', '30m', '60m', '120m', '1d', '1w', '1M'}, "This time interval {self.time_interval} is not supported for {self.data_source}"
        elif self.time_interval == "quantconnect":
            pass
        elif self.time_interval == "ricequant":
            pass
        elif self.time_interval == "tusharepro":
            pass
        elif self.time_interval == "wrds":
            pass
        elif self.time_interval == "yahoofinance":
            pass
        else:
            raise ValueError("Not support this time interval: {self.time_interval} in {self.data_source}")



import numpy as np
import pandas as pd
from typing import List
import stockstats
from talib.abstract import CCI, DX, MACD, RSI

TIME_INTERVAL = '1D'


class BasicProcessor:
    def __init__(self, data_source: str, **kwargs):

        assert data_source in ["alpaca", "ccxt", "binance", "iexcloud", "joinquant", "quantconnect", "ricequant", "wrds", "yahoofinance", "tusharepro", ], "Data source input is NOT supported yet."
        self.data_source: str = data_source
        self.time_interval: str = TIME_INTERVAL
        self.time_zone: str = ""
        self.dataframe: pd.DataFrame = pd.DataFrame()
        self.dictnumpy: dict = {}

    def download_data(self, ticker_list: List[str], start_date: str, end_date: str, time_interval: str):
        pass

    def clean_data(self):
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe = self.dataframe.rename(columns={'date': 'time'})
        if "datetime" in self.dataframe.columns.values.tolist():
            self.dataframe = self.dataframe.rename(columns={'datetime': 'time'})
        if self.data_source == "ccxt":
            self.dataframe = self.dataframe.rename(columns={'index': 'time'})
        elif self.data_source == 'ricequant':
            ''' RiceQuant data is already cleaned, we only need to transform data format here.
                No need for filling NaN data'''
            self.dataframe = self.dataframe.rename(columns={'order_book_id': 'tic'})
            # raw df uses multi-index (tic,time), reset it to single index (time)
            self.dataframe = self.dataframe.reset_index(level=[0, 1])
            # check if there is NaN values
            assert not self.dataframe.isnull().values.any()
        self.dataframe = self.dataframe.dropna()
        # adj_close: adjusted close price
        if 'adj_close' not in self.dataframe.columns.values.tolist():
            self.dataframe['adj_close'] = self.dataframe['close']
        self.dataframe = self.dataframe.sort_values(by=['time', 'tic'])
        self.dataframe = self.dataframe[['tic', 'time', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]

    def get_trading_days(self, start: str, end: str) -> List[str]:
        if self.data_source in ["binance", "ccxt", "quantconnect", "ricequant", "tusharepro"]:
            print("Calculate get_trading_days not supported for {} yet.".format(self.data_source))
            return None
        pass

    # use_stockstats_or_talib: 0 (stockstats, default), or 1 (use talib). Users can choose the method.
    def add_technical_indicator(self, tech_indicator_list: List[str], use_stockstats_or_talib: int=0):
        """
        calculate technical indicators
        use stockstats/talib package to add technical inidactors
        """
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe = self.dataframe.rename(columns={'date': 'time'})

        if self.data_source == "ccxt":
            self.dataframe = self.dataframe.rename(columns={'index': 'time'})

        self.dataframe = self.dataframe.reset_index(drop=False)
        if "level_1" in self.dataframe.columns:
            self.dataframe = self.dataframe.drop(columns=["level_1"])
        if "level_0" in self.dataframe.columns and "tic" not in self.dataframe.columns:
            self.dataframe = self.dataframe.rename(columns={"level_0": "tic"})
        assert use_stockstats_or_talib in [0, 1]
        if use_stockstats_or_talib == 0:  # use stockstats
            stock = stockstats.StockDataFrame.retype(self.dataframe.copy())
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
                tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = MACD(tic_df['close'], fastperiod=12,
                                                                                  slowperiod=26, signalperiod=9)
                tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
                tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
                tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
                final_df = final_df.append(tic_df)
            self.dataframe = final_df

        self.dataframe = self.dataframe.sort_values(by=["time", "tic"])
        time_to_drop = self.dataframe[self.dataframe.isna().any(axis=1)].time.unique()
        self.dataframe = self.dataframe[~self.dataframe.time.isin(time_to_drop)]
        print("Succesfully add technical indicators")

    def add_turbulence(self):
        """
        add turbulence index from a precalcualted dataframe
        """
        # df = data.copy()
        # turbulence_index = self.calculate_turbulence(df)
        # df = df.merge(turbulence_index, on="time")
        # df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        # return df
        if self.data_source in ["binance", "ccxt", "iexcloud", "joinquant", "quantconnect"]:
            print("Turbulence not supported for {} yet. Return original DataFrame.".format(self.data_source))
        if self.data_source in ["alpaca", "ricequant", "tusharepro", "wrds", "yahoofinance"]:
            turbulence_index = self.calculate_turbulence(self.dataframe)
            self.dataframe = self.dataframe.merge(turbulence_index, on="time")
            self.dataframe = self.dataframe.sort_values(["time", "tic"]).reset_index(drop=True)

    def calculate_turbulence(self, time_period: int = 252) \
            -> pd.DataFrame:
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = self.dataframe.copy()
        df_price_pivot = df.pivot(index="time", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df['time'].unique()
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
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
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
            print('VIX is not applicable for {}. Return original DataFrame'.format(self.data_source))
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
        self.dataframe = [ticker]
        self.download_data(self.start, self.end, self.time_interval)
        self.clean_data()
        # vix = cleaned_vix[["time", "close"]]
        # vix = vix.rename(columns={"close": "VIXY"})
        cleaned_vix = self.dataframe.rename(columns={ticker: "vix"})

        self.dataframe = self.dataframe.merge(cleaned_vix, on="time")
        self.dataframe = self.dataframe.sort_values(["time", "tic"]).reset_index(drop=True)

    def df_to_array(self, tech_indicator_list: list, if_vix: bool):
        unique_ticker = self.dataframe.tic.unique()
        price_array = np.column_stack([self.dataframe[self.dataframe.tic==tic].close for tic in unique_ticker])
        tech_array = np.hstack([self.dataframe.loc[(self.dataframe.tic==tic), tech_indicator_list] for tic in unique_ticker])
        if if_vix:
            risk_array = np.column_stack([self.dataframe[self.dataframe.tic==tic].vix for tic in unique_ticker])
        else:
            risk_array = np.column_stack([self.dataframe[self.dataframe.tic==tic].turbulence for tic in unique_ticker]) if "turbulence" in self.dataframe.columns else None
        print("Successfully transformed into array")
        return price_array, tech_array, risk_array
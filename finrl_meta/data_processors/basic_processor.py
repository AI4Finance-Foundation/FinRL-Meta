import numpy as np
import pandas as pd
from typing import List
import stockstats
from talib.abstract import CCI, DX, MACD, RSI

TIME_INTERVAL = '1D'


class BasicProcessor:
    def __init__(self, data_source: str, **kwargs):

        assert data_source in ["alpaca", "ccxt", "binance", "iexcloud", "joinquant", "quantconnect", "ricequant", "wrds", "yahoofinance", "tusharepro", ], "Data source input is NOT supported yet."
        self.data_source = data_source
        self.time_interval = TIME_INTERVAL
        self.time_zone = ''

    def download_data(self, ticker_list: List[str], start_date: str, end_date: str, time_interval: str) \
            -> pd.DataFrame:
        pass

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if "date" in df.columns.values.tolist():
            df = df.rename(columns={'date': 'time'})
        if "datetime" in df.columns.values.tolist():
            df = df.rename(columns={'datetime': 'time'})
        if self.data_source == "ccxt":
            df = df.rename(columns={'index': 'time'})
        elif self.data_source == 'ricequant':
            ''' RiceQuant data is already cleaned, we only need to transform data format here.
                No need for filling NaN data'''
            df = df.rename(columns={'order_book_id': 'tic'})
            # raw df uses multi-index (tic,time), reset it to single index (time)
            df = df.reset_index(level=[0, 1])
            # check if there is NaN values
            assert not df.isnull().values.any()
        df2 = df.dropna()
        # adj_close: adjusted close price
        if 'adj_close' not in df2.columns.values.tolist():
            df2['adj_close'] = df2['close']
        df2 = df2.sort_values(by=['time', 'tic'])
        final_df = df2[['tic', 'time', 'open', 'high', 'low', 'close', 'adj_close', 'volume']]
        return final_df

    def get_trading_days(self, start: str, end: str) -> List[str]:
        pass

    # use_stockstats: True (stockstats), or False (use talib). Users can choose the method.
    def add_technical_indicator(self, data: pd.DataFrame, tech_indicator_list: List[str], use_stockstats: bool=True) \
            -> pd.DataFrame:
        """
        calculate technical indicators
        use stockstats/talib package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        # if "date" in df.columns.values.tolist():
        #     df = df.rename(columns={'date': 'time'})
        #
        # if self.data_source == "ccxt":
        #     df = df.rename(columns={'index': 'time'})

        # df = df.reset_index(drop=False)
        # df = df.drop(columns=["level_1"])
        # df = df.rename(columns={"level_0": "tic", "date": "time"})
        if use_stockstats:  # use stockstats
            stock = stockstats.StockDataFrame.retype(df.copy())
            unique_ticker = stock.tic.unique()
            for indicator in tech_indicator_list:
                indicator_df = pd.DataFrame()
                for i in range(len(unique_ticker)):
                    try:
                        temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                        temp_indicator = pd.DataFrame(temp_indicator)
                        temp_indicator["tic"] = unique_ticker[i]
                        temp_indicator["time"] = df[df.tic == unique_ticker[i]][
                            "time"
                        ].to_list()
                        indicator_df = indicator_df.append(
                            temp_indicator, ignore_index=True
                        )
                    except Exception as e:
                        print(e)
                df = df.merge(
                    indicator_df[["tic", "time", indicator]], on=["tic", "time"], how="left"
                )
        else:  # use talib
            final_df = pd.DataFrame()
            for i in df.tic.unique():
                tic_df = df[df.tic == i]
                tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = MACD(tic_df['close'], fastperiod=12,
                                                                                  slowperiod=26, signalperiod=9)
                tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
                tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
                tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
                final_df = final_df.append(tic_df)
            df = final_df

        df = df.sort_values(by=["time", "tic"])
        df = df.dropna()
        print("Succesfully add technical indicators")
        return df

    def add_turbulence(self, data: pd.DataFrame) \
            -> pd.DataFrame:
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="time")
        df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data: pd.DataFrame, time_period: int = 252) \
            -> pd.DataFrame:
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
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

    def add_vix(self, data: pd.DataFrame) \
            -> pd.DataFrame:
        """
        add vix from processors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        if self.data_source in ['binance', 'ccxt', 'iexcloud', 'joinquant', 'quantconnect', 'ricequant']:
            print('VIX is not applicable for {}. Return original DataFrame'.format(self.data_source))
            return data

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
        vix_df = self.download_data([ticker], self.start, self.end, self.time_interval)
        cleaned_vix = self.clean_data(vix_df)
        # vix = cleaned_vix[["time", "close"]]
        # vix = vix.rename(columns={"close": "VIXY"})
        cleaned_vix = cleaned_vix.rename(columns={ticker: "vix"})

        df = data.copy()
        df = df.merge(cleaned_vix, on="time")
        df = df.sort_values(["time", "tic"]).reset_index(drop=True)
        return df

    # tech_array: technical indicator
    # price_array, close price
    def df_to_array(self, df: pd.DataFrame, tech_indicator_list: List[str], if_vix: bool) \
            -> List[np.array]:
        """transform final df to numpy arrays"""
        unique_ticker = df.tic.unique()
        print(unique_ticker)
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["adj_close"]].values
                # price_ary = df[df.tic==tic]['close'].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["vix"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["adj_close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        assert price_array.shape[0] == tech_array.shape[0]
        assert tech_array.shape[0] == turbulence_array.shape[0]
        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array

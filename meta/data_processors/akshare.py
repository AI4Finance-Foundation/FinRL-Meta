import copy
import time
import warnings
warnings.filterwarnings("ignore")
from typing import List

import pandas as pd
from tqdm import tqdm

import stockstats
import talib
from meta.data_processors._base import _Base

import akshare as ak # pip install akshare

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
            self.adj = ''
            
        if "period" in kwargs.keys():
            self.period = kwargs["period"]
        else:
            self.period = 'daily'
        
    def get_data(self, id) -> pd.DataFrame:

        return ak.stock_zh_a_hist(
            symbol = id, 
            period = self.time_interval,
            start_date = self.start_date,
            end_date = self.end_date,
            adjust = self.adj,
            )

    def download_data(self, ticker_list: List[str]):
        """
        `pd.DataFrame`
            7 columns: A tick symbol, date, open, high, low, close and volume
            for the specified stock ticker
        """
        assert self.time_interval in ["daily","weekly",'monthly'], "Not supported currently"

        self.ticker_list = ticker_list

        self.dataframe = pd.DataFrame()
        for i in tqdm(ticker_list, total=len(ticker_list)):
            nonstandard_id = self.transfer_standard_ticker_to_nonstandard(i)
            df_temp = self.get_data(nonstandard_id)
            df_temp["tic"] = i
            # df_temp = self.get_data(i)
            self.dataframe = pd.concat([self.dataframe,df_temp])
            # self.dataframe = self.dataframe.append(df_temp)
            # print("{} ok".format(i))
            time.sleep(0.25)

        self.dataframe.columns = [
            "date",
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

        self.dataframe.sort_values(by=["date", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        self.dataframe = self.dataframe[
            ["tic", "date", "open", "high", "low", "close", "volume"]
        ]
        # self.dataframe.loc[:, 'tic'] = pd.DataFrame((self.dataframe['tic'].tolist()))
        self.dataframe["date"] = pd.to_datetime(self.dataframe["date"], format="%Y-%m-%d")
        self.dataframe["day"] = self.dataframe["date"].dt.dayofweek
        self.dataframe["date"] = self.dataframe.date.apply(
            lambda x: x.strftime("%Y-%m-%d")
        )

        self.dataframe.dropna(inplace=True)
        self.dataframe.sort_values(by=["date", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        print("Shape of DataFrame: ", self.dataframe.shape)

    def clean_data(self):
        dfc = copy.deepcopy(self.dataframe)

        dfcode = pd.DataFrame(columns=["tic"])
        dfdate = pd.DataFrame(columns=["date"])

        dfcode.tic = dfc.tic.unique()

        if "time" in dfc.columns.values.tolist():
            dfc = dfc.rename(columns={"time": "date"})

        dfdate.date = dfc.date.unique()
        dfdate.sort_values(by="date", ascending=False, ignore_index=True, inplace=True)

        # the old pandas may not support pd.merge(how="cross")
        try:
            df1 = pd.merge(dfcode, dfdate, how="cross")
        except:
            print("Please wait for a few seconds...")
            df1 = pd.DataFrame(columns=["tic", "date"])
            for i in range(dfcode.shape[0]):
                for j in range(dfdate.shape[0]):
                    df1 = df1.append(
                        pd.DataFrame(
                            data={
                                "tic": dfcode.iat[i, 0],
                                "date": dfdate.iat[j, 0],
                            },
                            index=[(i + 1) * (j + 1) - 1],
                        )
                    )

        df2 = pd.merge(df1, dfc, how="left", on=["tic", "date"])

        # back fill missing data then front fill
        df3 = pd.DataFrame(columns=df2.columns)
        for i in self.ticker_list:
            df4 = df2[df2.tic == i].fillna(method="bfill").fillna(method="ffill")
            df3 = pd.concat([df3, df4], ignore_index=True)

        df3 = df3.fillna(0)

        # reshape dataframe
        df3 = df3.sort_values(by=["date", "tic"]).reset_index(drop=True)

        print("Shape of DataFrame: ", df3.shape)

        self.dataframe = df3

    def add_technical_indicator(
        self, tech_indicator_list: List[str], select_stockstats_talib: int = 0,drop_na_timestpe: int = 0,
    ):
        """
        calculate technical indicators
        use stockstats/talib package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        if "date" in self.dataframe.columns.values.tolist():
            self.dataframe.rename(columns={"date": "time"}, inplace=True)

        if self.data_source == "ccxt":
            self.dataframe.rename(columns={"index": "time"}, inplace=True)

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
                        temp_indicator["time"] = self.dataframe[
                            self.dataframe.tic == unique_ticker[i]
                        ]["time"].to_list()
                        indicator_df = pd.concat(
                            [indicator_df, temp_indicator],
                            axis=0,
                            join="outer",
                            ignore_index=True,
                        )
                    except Exception as e:
                        print(e)
                if not indicator_df.empty:
                    self.dataframe = self.dataframe.merge(
                        indicator_df[["tic", "time", indicator]],
                        on=["tic", "time"],
                        how="left",
                    )
        else:  # use talib
            final_df = pd.DataFrame()
            for i in self.dataframe.tic.unique():
                tic_df = self.dataframe[self.dataframe.tic == i]
                (
                    tic_df.loc["macd"],
                    tic_df.loc["macd_signal"],
                    tic_df.loc["macd_hist"],
                ) = talib.MACD(
                    tic_df["close"],
                    fastperiod=12,
                    slowperiod=26,
                    signalperiod=9,
                )
                tic_df.loc["rsi"] = talib.RSI(tic_df["close"], timeperiod=14)
                tic_df.loc["cci"] = talib.CCI(
                    tic_df["high"],
                    tic_df["low"],
                    tic_df["close"],
                    timeperiod=14,
                )
                tic_df.loc["dx"] = talib.DX(
                    tic_df["high"],
                    tic_df["low"],
                    tic_df["close"],
                    timeperiod=14,
                )
                final_df = pd.concat([final_df, tic_df], axis=0, join="outer")
            self.dataframe = final_df

        self.dataframe.sort_values(by=["time", "tic"], inplace=True)
        if drop_na_timestpe:
            time_to_drop = self.dataframe[self.dataframe.isna().any(axis=1)].time.unique()
            self.dataframe = self.dataframe[~self.dataframe.time.isin(time_to_drop)]
        self.dataframe.rename(columns={"time": "date"}, inplace=True)
        print("Succesfully add technical indicators")

    # def get_trading_days(self, start: str, end: str) -> List[str]:
    #     print('not supported currently!')
    #     return ['not supported currently!']

    # def add_turbulence(self, data: pd.DataFrame) \
    #         -> pd.DataFrame:
    #     print('not supported currently!')
    #     return pd.DataFrame(['not supported currently!'])

    # def calculate_turbulence(self, data: pd.DataFrame, time_period: int = 252) \
    #         -> pd.DataFrame:
    #     print('not supported currently!')
    #     return pd.DataFrame(['not supported currently!'])

    # def add_vix(self, data: pd.DataFrame) \
    #         -> pd.DataFrame:
    #     print('not supported currently!')
    #     return pd.DataFrame(['not supported currently!'])

    # def df_to_array(self, df: pd.DataFrame, tech_indicator_list: List[str], if_vix: bool) \
    #         -> List[np.array]:
    #     print('not supported currently!')
    #     return pd.DataFrame(['not supported currently!'])

    def data_split(self, df, start, end, target_date_col="date"):
        """
        split the dataset into training or testing using date
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
    
    def transfer_date(self, date: str) -> str:
        if "-" in date:
            date = "".join(date.split("-"))
        elif "." in date:
            date = "".join(date.split("."))
        elif "/" in date:
            date = "".join(date.split("/"))
        return date

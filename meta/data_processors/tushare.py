import copy
import os
import time
import warnings

warnings.filterwarnings("ignore")
from typing import List

import pandas as pd
from tqdm import tqdm
from matplotlib import pyplot as plt

import stockstats
import talib
from meta.data_processors._base import _Base

import tushare as ts


class Tushare(_Base):
    """
    key-value in kwargs
    ----------
        token : str
            get from https://waditu.com/ after registration
        adj: str
            Whether to use adjusted closing price. Default is None.
            If you want to use forward adjusted closing price or 前复权. pleses use 'qfq'
            If you want to use backward adjusted closing price or 后复权. pleses use 'hfq'
    """

    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        assert "token" in kwargs.keys(), "Please input token!"
        self.token = kwargs["token"]
        if "adj" in kwargs.keys():
            self.adj = kwargs["adj"]
            print(f"Using {self.adj} method.")
        else:
            self.adj = None

    def get_data(self, id) -> pd.DataFrame:
        # df1 = ts.pro_bar(ts_code=id, start_date=self.start_date,end_date='20180101')
        # dfb=pd.concat([df, df1], ignore_index=True)
        # print(dfb.shape)
        return ts.pro_bar(
            ts_code=id,
            start_date=self.start_date,
            end_date=self.end_date,
            adj=self.adj,
        )

    def download_data(
        self, ticker_list: List[str], save_path: str = "./data/dataset.csv"
    ):
        """
        `pd.DataFrame`
            7 columns: A tick symbol, time, open, high, low, close and volume
            for the specified stock ticker
        """
        assert self.time_interval == "1d", "Not supported currently"

        self.ticker_list = ticker_list
        ts.set_token(self.token)

        self.dataframe = pd.DataFrame()
        for i in tqdm(ticker_list, total=len(ticker_list)):
            # nonstandard_id = self.transfer_standard_ticker_to_nonstandard(i)
            # df_temp = self.get_data(nonstandard_id)
            df_temp = self.get_data(i)
            self.dataframe = pd.concat(
                [self.dataframe, df_temp], ignore_index=True
            )  # 20240905 updated self.dataframe = self.dataframe.append(df_temp)
            # print("{} ok".format(i))
            time.sleep(0.25)

        self.dataframe.columns = [
            "tic",
            "time",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "volume",
            "amount",
        ]
        self.dataframe.sort_values(by=["time", "tic"], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)

        self.dataframe = self.dataframe[
            ["tic", "time", "open", "high", "low", "close", "volume"]
        ]
        # self.dataframe.loc[:, 'tic'] = pd.DataFrame((self.dataframe['tic'].tolist()))
        self.dataframe["time"] = pd.to_datetime(self.dataframe["time"], format="%Y%m%d")
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
        # "600000.XSHG" -> "600000.SH"
        # "000612.XSHE" -> "000612.SZ"
        n, alpha = ticker.split(".")
        assert alpha in ["XSHG", "XSHE"], "Wrong alpha"
        if alpha == "XSHG":
            nonstandard_ticker = n + ".SH"
        elif alpha == "XSHE":
            nonstandard_ticker = n + ".SZ"
        return nonstandard_ticker

    def save_data(self, path):
        if ".csv" in path:
            path = path.split("/")
            filename = path[-1]
            path = "/".join(path[:-1] + [""])
        else:
            if path[-1] == "/":
                filename = "dataset.csv"
            else:
                filename = "/dataset.csv"

        os.makedirs(path, exist_ok=True)
        self.dataframe.to_csv(path + filename, index=False)

    def load_data(self, path):
        assert ".csv" in path  # only support csv format now
        self.dataframe = pd.read_csv(path)
        columns = self.dataframe.columns
        assert (
            "tic" in columns and "time" in columns and "close" in columns
        )  # input file must have "tic","time" and "close" columns


class ReturnPlotter:
    """
    An easy-to-use plotting tool to plot cumulative returns over time.
    Baseline supports equal weighting(default) and any stocks you want to use for comparison.
    """

    def __init__(self, df_account_value, df_trade, start_date, end_date):
        self.start = start_date
        self.end = end_date
        self.trade = df_trade
        self.df_account_value = df_account_value

    def get_baseline(self, ticket):
        df = ts.get_hist_data(ticket, start=self.start, end=self.end)
        df.loc[:, "dt"] = df.index
        df.index = range(len(df))
        df.sort_values(axis=0, by="dt", ascending=True, inplace=True)
        df["time"] = pd.to_datetime(df["dt"], format="%Y-%m-%d")
        return df

    def plot(self, baseline_ticket=None):
        """
        Plot cumulative returns over time.
        use baseline_ticket to specify stock you want to use for comparison
        (default: equal weighted returns)
        """
        baseline_label = "Equal-weight portfolio"
        tic2label = {"399300": "CSI 300 Index", "000016": "SSE 50 Index"}
        if baseline_ticket:
            # 使用指定ticket作为baseline
            baseline_df = self.get_baseline(baseline_ticket)
            baseline_date_list = baseline_df.time.dt.strftime("%Y-%m-%d").tolist()
            df_date_list = self.df_account_value.time.tolist()
            df_account_value = self.df_account_value[
                self.df_account_value.time.isin(baseline_date_list)
            ]
            baseline_df = baseline_df[baseline_df.time.isin(df_date_list)]
            baseline = baseline_df.close.tolist()
            baseline_label = tic2label.get(baseline_ticket, baseline_ticket)
            ours = df_account_value.account_value.tolist()
        else:
            # 均等权重
            all_date = self.trade.time.unique().tolist()
            baseline = []
            for day in all_date:
                day_close = self.trade[self.trade["time"] == day].close.tolist()
                avg_close = sum(day_close) / len(day_close)
                baseline.append(avg_close)
            ours = self.df_account_value.account_value.tolist()

        ours = self.pct(ours)
        baseline = self.pct(baseline)

        days_per_tick = (
            60  # you should scale this variable accroding to the total trading days
        )
        time = list(range(len(ours)))
        datetimes = self.df_account_value.time.tolist()
        ticks = [tick for t, tick in zip(time, datetimes) if t % days_per_tick == 0]
        plt.title("Cumulative Returns")
        plt.plot(time, ours, label="DDPG Agent", color="green")
        plt.plot(time, baseline, label=baseline_label, color="grey")
        plt.xticks([i * days_per_tick for i in range(len(ticks))], ticks, fontsize=7)

        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")

        plt.legend()
        plt.show()
        plt.savefig(f"plot_{baseline_ticket}.png")

    def plot_all(self):
        baseline_label = "Equal-weight portfolio"
        tic2label = {"399300": "CSI 300 Index", "000016": "SSE 50 Index"}

        # time lists
        # algorithm time list
        df_date_list = self.df_account_value.time.tolist()

        # 399300 time list
        csi300_df = self.get_baseline("399300")
        csi300_date_list = csi300_df.time.dt.strftime("%Y-%m-%d").tolist()

        # 000016 time list
        sh50_df = self.get_baseline("000016")
        sh50_date_list = sh50_df.time.dt.strftime("%Y-%m-%d").tolist()

        # find intersection
        all_date = sorted(
            list(set(df_date_list) & set(csi300_date_list) & set(sh50_date_list))
        )

        # filter data
        csi300_df = csi300_df[csi300_df.time.isin(all_date)]
        baseline_300 = csi300_df.close.tolist()
        baseline_label_300 = tic2label["399300"]

        sh50_df = sh50_df[sh50_df.time.isin(all_date)]
        baseline_50 = sh50_df.close.tolist()
        baseline_label_50 = tic2label["000016"]

        # 均等权重
        baseline_equal_weight = []
        for day in all_date:
            day_close = self.trade[self.trade["time"] == day].close.tolist()
            avg_close = sum(day_close) / len(day_close)
            baseline_equal_weight.append(avg_close)

        df_account_value = self.df_account_value[
            self.df_account_value.time.isin(all_date)
        ]
        ours = df_account_value.account_value.tolist()

        ours = self.pct(ours)
        baseline_300 = self.pct(baseline_300)
        baseline_50 = self.pct(baseline_50)
        baseline_equal_weight = self.pct(baseline_equal_weight)

        days_per_tick = (
            60  # you should scale this variable accroding to the total trading days
        )
        time = list(range(len(ours)))
        datetimes = self.df_account_value.time.tolist()
        ticks = [tick for t, tick in zip(time, datetimes) if t % days_per_tick == 0]
        plt.title("Cumulative Returns")
        plt.plot(time, ours, label="DDPG Agent", color="darkorange")
        plt.plot(
            time,
            baseline_equal_weight,
            label=baseline_label,
            color="cornflowerblue",
        )  # equal weight
        plt.plot(
            time, baseline_300, label=baseline_label_300, color="lightgreen"
        )  # 399300
        plt.plot(time, baseline_50, label=baseline_label_50, color="silver")  # 000016
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")

        plt.xticks([i * days_per_tick for i in range(len(ticks))], ticks, fontsize=7)
        plt.legend()
        plt.show()
        plt.savefig("./plot_all.png")

    def pct(self, l):
        """Get percentage"""
        base = l[0]
        return [x / base for x in l]

    def get_return(self, df, value_col_name="account_value"):
        df = copy.deepcopy(df)
        df["daily_return"] = df[value_col_name].pct_change(1)
        df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d")
        df.set_index("time", inplace=True, drop=True)
        df.index = df.index.tz_localize("UTC")
        return pd.Series(df["daily_return"], index=df.index)

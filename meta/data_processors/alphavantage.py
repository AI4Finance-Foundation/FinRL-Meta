import datetime
import json
from typing import List

import pandas as pd
import requests

from meta.config import BINANCE_BASE_URL
from meta.config import TIME_ZONE_BERLIN
from meta.config import TIME_ZONE_JAKARTA
from meta.config import TIME_ZONE_PARIS
from meta.config import TIME_ZONE_SELFDEFINED
from meta.config import TIME_ZONE_SHANGHAI
from meta.config import TIME_ZONE_USEASTERN
from meta.config import USE_TIME_ZONE_SELFDEFINED
from meta.data_processors._base import _Base
from meta.data_processors._base import calc_time_zone


def transfer_date(d):
    date = str(d.year)
    date += "-"
    if len(str(d.month)) == 1:
        date += "0" + str(d.month)
    else:
        date += d.month
    date += "-"
    if len(str(d.day)) == 1:
        date += "0" + str(d.day)
    else:
        date += str(d.day)
    return date


class Alphavantage(_Base):
    def __init__(
        self,
        data_source: str,
        start_date: str,
        end_date: str,
        time_interval: str,
        **kwargs,
    ):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)

        assert time_interval == "1d", "please set the time_interval 1d"

    # ["1d"]
    def download_data(
        self, ticker_list: List[str], save_path: str = "./data/dataset.csv"
    ):
        # self.time_zone = calc_time_zone(
        #     ticker_list, TIME_ZONE_SELFDEFINED, USE_TIME_ZONE_SELFDEFINED
        # )
        self.dataframe = pd.DataFrame()
        for ticker in ticker_list:
            url = (
                "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol="
                + ticker
                + "&apikey=demo"
            )
            r = requests.get(url)
            data = r.json()
            data2 = json.dumps(data["Time Series (Daily)"])
            # gnData = json.dumps(data["Data"]["gn"])
            df2 = pd.read_json(data2)
            # gnDf = pd.read_json(gnData)

            df3 = pd.DataFrame(df2.values.T, columns=df2.index, index=df2.columns)
            df3.rename(
                columns={
                    "1. open": "open",
                    "2. high": "high",
                    "3. low": "low",
                    "4. close": "close",
                    "5. volume": "volume",
                },
                inplace=True,
            )
            df3["tic"] = ticker
            dates = [transfer_date(df2.index[i]) for i in range(len(df2.index))]
            df3["date"] = dates
            self.dataframe = pd.concat([self.dataframe, df3])
        self.dataframe = self.dataframe.sort_values(by=["date", "tic"]).reset_index(
            drop=True
        )

        self.save_data(save_path)

        print(
            f"Download complete! Dataset saved to {save_path}. \nShape of DataFrame: {self.dataframe.shape}"
        )

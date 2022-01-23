import os
from datetime import datetime
from typing import List

import pandas as pd
import pandas_market_calendars as mcal
import pytz
import requests
from basic_processor import BasicProcessor


class IEXCloudProcessor(BasicProcessor):
    @classmethod
    def _get_base_url(self, mode: str) -> str:

        as1 = "mode must be sandbox or production."
        assert mode in ["sandbox", "production"], as1

        if mode == "sandbox":
            return "https://sandbox.iexapis.com"

        return "https://cloud.iexapis.com"

    def __init__(
        self,
        data_source: str = None,
        **kwargs,
    ):
        BasicProcessor.__init__(self, data_source, **kwargs)
        self.base_url = self._get_base_url(mode=kwargs['mode'])
        self.token = kwargs['token'] or os.environ.get("IEX_TOKEN")

    def download_data(self, ticker_list: List[str], start_date: str, end_date: str, time_interval: str):
        """Returns end of day historical data for up to 15 years.

        Args:
            ticker_list (List[str]): List of the tickers to retrieve information.
            start_date (str): Oldest date of the range.
            end_date (str): Latest date of the range.

        Returns:
            pd.DataFrame: A pandas dataframe with end of day historical data
            for the specified tickers with the following columns:
            date, tic, open, high, low, close, adj_close, volume.

        Examples:
            kwargs['mode'] = 'sandbox'
            kwargs['token'] = 'Tsk_d633e2ff10d463...'
            >>> iex_dloader = IEXCloudProcessor(data_source='iexcloud', **kwargs)
            >>> iex_dloader.download_data(ticker_list=["AAPL", "NVDA"],
                                        start_date='2014-01-01',
                                        end_date='2021-12-12',
                                        time_interval = '1D')
        """
        assert time_interval == '1D'  # one day

        self.dataframe = pd.DataFrame()

        query_params = {
            "token": self.token,
        }

        if start_date and end_date:
            query_params["from"] = start_date
            query_params["to"] = end_date

        for stock in ticker_list:
            end_point = (
                f"{self.base_url}/stable/time-series/HISTORICAL_PRICES/{stock}"
            )

            response = requests.get(
                url=end_point,
                params=query_params,
            )
            if response.status_code == 200:
                temp = pd.DataFrame.from_dict(data=response.json())
                temp["ticker"] = stock
                self.dataframe = self.dataframe.append(temp)
            else:
                raise requests.exceptions.RequestException(response.text)

        self.dataframe = self.dataframe[
            [
                "date",
                "ticker",
                "open",
                "high",
                "low",
                "close",
                "fclose",
                "volume",
            ]
        ]
        self.dataframe = self.dataframe.rename(columns={"ticker": "tic", "date": "time", "fclose": "adj_close"})

        self.dataframe.date = self.dataframe.date.map(
            lambda x: datetime.fromtimestamp(x / 1000, pytz.UTC).strftime(
                "%Y-%m-%d"
            )
        )



    def get_trading_days(self, start: str, end: str) -> List[str]:
        """Retrieves every trading day between two dates.

        Args:
            start (str): Oldest date of the range.
            end (str): Latest date of the range.

        Returns:
            List[str]: List of all trading days in YYYY-dd-mm format.

        Examples:
            >>> iex_dloader = IEXCloudProcessor(data_source='iexcloud',
                                                mode='sandbox',
                                                token='Tsk_d633e2ff10d463...')
            >>> iex_dloader.get_trading_days(start='2014-01-01',
                                             end='2021-12-12')
            ['2021-12-15', '2021-12-16', '2021-12-17']
        """
        nyse = mcal.get_calendar("NYSE")

        df = nyse.schedule(
            start_date=pd.Timestamp(start, tz=pytz.UTC),
            end_date=pd.Timestamp(end, tz=pytz.UTC),
        )
        tradin_days = df.applymap(
            lambda x: x.strftime("%Y-%m-%d")
        ).market_open.to_list()

        return tradin_days

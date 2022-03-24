from typing import List

# from basic_processor import _Base
from finrl_meta.data_processors._base import _Base

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


## The code of this file is used in website, not locally.
class Quantconnect(_Base):
    def __init__(self, data_source: str, start_date: str, end_date: str, time_interval: str, **kwargs):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)

    # def data_fetch(start_time, end_time, stock_list, resolution=Resolution.Daily) :
    #     #resolution: Daily, Hour, Minute, Second
    #     qb = QuantBook()
    #     for stock in stock_list:
    #         qb.AddEquity(stock)
    #     history = qb.History(qb.Securities.Keys, start_time, end_time, resolution)
    #     return history

    def download_data(self, ticker_list: List[str]):
        # self.time_zone = calc_time_zone(ticker_list, TIME_ZONE_SELFDEFINED, USE_TIME_ZONE_SELFDEFINED)

        # start_date = pd.Timestamp(start_date, tz=self.time_zone)
        # end_date = pd.Timestamp(end_date, tz=self.time_zone) + pd.Timedelta(days=1)
        qb = QuantBook()
        for stock in ticker_list:
            qb.AddEquity(stock)
        history = qb.History(qb.Securities.Keys, self.start_date, self.end_date, self.time_interval)
        self.dataframe = history

    # def preprocess(df, stock_list):
    #     df = df[['open','high','low','close','volume']]
    #     if_first_time = True
    #     for stock in stock_list:
    #         if if_first_time:
    #             ary = df.loc[stock].values
    #             if_first_time = False
    #         else:
    #             temp = df.loc[stock].values
    #             ary = np.hstack((ary,temp))
    #     return ary

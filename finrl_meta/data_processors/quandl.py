import quandl

'''Reference: https://github.com/AI4Finance-LLC/FinRL'''

from typing import List
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
try:
    import exchange_calendars as tc
except:
    print('Cannot import exchange_calendars.',
          'If you are using python>=3.7, please install it.')
    import trading_calendars as tc
    print('Use trading_calendars instead for yahoofinance processor..')
# from basic_processor import _Base
from finrl_meta.data_processors._base import _Base
from finrl_meta.data_processors._base import calc_time_zone

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

TIME_ZONE_SELFDEFINED = TIME_ZONE_USEASTERN  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 1  # 0 (default) or 1 (use the self defined)


class Quandl(_Base):

    def __init__(self, data_source: str, start_date: str, end_date: str, time_interval: str, **kwargs):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)

    def download_data(self, ticker_list: List[str]):
        self.time_zone = calc_time_zone(ticker_list, TIME_ZONE_SELFDEFINED, USE_TIME_ZONE_SELFDEFINED)

        # Download and save the data in a pandas DataFrame:
        # data_df = pd.DataFrame()
        # # set paginate to True because Quandl limits tables API to 10,000 rows per call
        # data = quandl.get_table('ZACKS/FC', paginate=True, ticker=ticker_list, per_end_date={'gte': '2021-09-01'}, qopts={'columns': ['ticker', 'per_end_date']})
        # data = quandl.get('ZACKS/FC', ticker=ticker_list,  start_date="2020-12-31", end_date="2021-12-31")
        self.dataframe = quandl.get_table('ZACKS/FC', ticker=ticker_list,
                                qopts={'columns': ['ticker', 'date', 'adjusted_close']},
                                date={'gte': self.start_date, 'lte': self.end_date},
                                paginate=True)
        self.dataframe.dropna(inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)
        print("Shape of DataFrame: ", self.dataframe.shape)
        # print("Display DataFrame: ", data_df.head())

        self.dataframe.sort_values(by=['date', 'ticker'], inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)






    # def get_trading_days(self, start, end):
    #


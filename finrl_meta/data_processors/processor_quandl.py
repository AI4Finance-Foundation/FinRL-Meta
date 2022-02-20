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
# from basic_processor import BasicProcessor
from finrl_meta.data_processors.basic_processor import BasicProcessor
from finrl_meta.data_processors.func import calc_time_zone

TIME_ZONE_SHANGHAI = 'Asia/Shanghai'  ## Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = 'US/Eastern'  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = 'Europe/Paris'  # CAC,
TIME_ZONE_BERLIN = 'Europe/Berlin'  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = 'Asia/Jakarta'  # LQ45
TIME_ZONE_SELFDEFINED = TIME_ZONE_USEASTERN  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 1  # 0 (default) or 1 (use the self defined)


class QuandlProcessor(BasicProcessor):

    def __init__(self, data_source: str, start_date, end_date, time_interval, **kwargs):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)

    def download_data(self, ticker_list: List[str]):
        self.time_zone = calc_time_zone(ticker_list, TIME_ZONE_SELFDEFINED, USE_TIME_ZONE_SELFDEFINED)

        # Download and save the data in a pandas DataFrame:
        data_df = pd.DataFrame()

        # data = quandl.get_table('ZACKS/FC', paginate=True, ticker=ticker_list, per_end_date={'gte': '2021-09-01'}, qopts={'columns': ['ticker', 'per_end_date']})
        # data = quandl.get('ZACKS/FC', ticker=ticker_list,  start_date="2020-12-31", end_date="2021-12-31")
        data = quandl.get_table('WIKI/PRICES', ticker=['AAPL', 'MSFT', 'WMT'],
                                qopts={'columns': ['ticker', 'date', 'adj_close']},
                                date={'gte': self.start_date, 'lte': self.end_date},
                                paginate=True)
        data_df = data_df.dropna()
        data_df = data_df.reset_index(drop=True)
        print("Shape of DataFrame: ", data_df.shape)
        # print("Display DataFrame: ", data_df.head())

        data_df = data_df.sort_values(by=['date', 'tic']).reset_index(drop=True)

        self.dataframe = data_df





    # def get_trading_days(self, start, end):
    #


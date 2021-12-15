import sys
from basic_processor import BasicProcessor
import pandas_market_calendars as mcal
import pandas as pd
import os
import pytz


class IEXCloudProcessor(BasicProcessor):

    @classmethod
    def _get_base_url(self, mode: str) -> str:

        as1 = 'mode must be sandbox or production.'
        assert mode in ['sandbox', 'production'], as1

        if mode == 'sandbox':
            return 'https://sandbox.iexapis.com'

        return 'https://cloud.iexapis.com'

    def __init__(self, data_source: str=None, mode: str=None, token: str = None, **kwargs):
        #BasicProcessor.__init__(self, data_source, **kwargs)
        self.base_url = self._get_base_url(mode=mode)
        self.token = token or os.environ.get("IEX_TOKEN")

    def download_data():
        pass

    def get_trading_days(self, start, end):
        nyse = mcal.get_calendar('NYSE')
        # df = nyse.sessions_in_range(pd.Timestamp(start,tz=pytz.UTC), pd.Timestamp(end,tz=pytz.UTC))

        df = nyse.schedule(start_date=pd.Timestamp(start,tz=pytz.UTC), end_date=pd.Timestamp(end,tz=pytz.UTC))
        tradin_days = df.applymap(lambda x: x.strftime('%Y-%m-%d')).market_open.to_list()
                            
        return tradin_days


if __name__=="__main__":
    iex_dloader = IEXCloudProcessor(data_source='',mode='sandbox')
    #iex_dloader.ohlcv_chart(["AAPL", "NVDA"], '5y')
    td = iex_dloader.get_trading_days(start='2012-07-10', end='2020-03-05')
    print(td)
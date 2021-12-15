from finrl_meta.data_processors.basic_processor import BasicProcessor
import trading_calendars as tc
import pandas as pd
import pytz


class IEXCloudProcessor(BasicProcessor):



    def __init__(self, data_source: str, **kwargs):
        BasicProcessor.__init__(self, data_source, **kwargs)

    def download_data():
        pass

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar('NYSE')
        df = nyse.sessions_in_range(pd.Timestamp(start,tz=pytz.UTC),
                                    pd.Timestamp(end,tz=pytz.UTC))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])
    
        return trading_days


if __name__=="__main__":
    c = IEXCloudProcessor()
    print(c.get_trading_days())
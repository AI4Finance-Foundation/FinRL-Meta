import numpy as np
import pandas as pd
# from basic_processor import BasicProcessor
from finrl_meta.data_processors.basic_processor import BasicProcessor

class QuantConnectEngineer(BasicProcessor):
    def __init__(self, data_source: str, **kwargs):
        BasicProcessor.__init__(self, data_source, **kwargs)
    def data_fetch(start_time, end_time, stock_list, resolution=Resolution.Daily) :
        #resolution: Daily, Hour, Minute, Second
        qb = QuantBook()
        for stock in stock_list:
            qb.AddEquity(stock)
        history = qb.History(qb.Securities.Keys, start_time, end_time, resolution)
        return history
    def preprocess(df, stock_list):
        df = df[['open','high','low','close','volume']]
        if_first_time = True
        for stock in stock_list:
            if if_first_time:
                ary = df.loc[stock].values
                if_first_time = False
            else:
                temp = df.loc[stock].values
                ary = np.hstack((ary,temp))
        return ary

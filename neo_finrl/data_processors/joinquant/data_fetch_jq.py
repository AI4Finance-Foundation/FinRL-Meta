import jqdatasdk as jq
import pandas as pd

def data_fetch(stock_list, num, unit, end_dt):
    df = jq.get_bars(security=stock_list, count=num, unit=unit, 
                     fields=['date','open','high','low','close','volume'],
                     end_dt=end_dt)
    return df



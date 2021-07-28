''' refer from https://techflare.blog/how-to-get-ohlcv-data-for-your-exchange-with-ccxt-library/'''

import numpy as np
import pandas as pd
import ccxt
import calendar
from datetime import datetime

binance = ccxt.binance()

def min_ohlcv(dt, pair, limit):
    # UTC native object
    since = calendar.timegm(dt.utctimetuple())*1000
    ohlcv = binance.fetch_ohlcv(symbol=pair, timeframe='1m', since=since, limit=limit)
    return ohlcv

def ohlcv(dt, pair, period='1d'):
    ohlcv = []
    limit = 1000
    if period == '1m':
        limit = 720
    elif period == '1d':
        limit = 1
    elif period == '1h':
        limit = 24
    elif period == '5m':
        limit = 288
    for i in dt:
        start_dt = i
        since = calendar.timegm(start_dt.utctimetuple())*1000
        if period == '1m':
            ohlcv.extend(min_ohlcv(start_dt, pair, limit))
        else:
            ohlcv.extend(binance.fetch_ohlcv(symbol=pair, timeframe=period, since=since, limit=limit))
    df = pd.DataFrame(ohlcv, columns = ['time', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = [datetime.fromtimestamp(float(time)/1000) for time in df['time']]
    df['open'] = df['open'].astype(np.float64)
    df['high'] = df['high'].astype(np.float64)
    df['low'] = df['low'].astype(np.float64)
    df['close'] = df['close'].astype(np.float64)
    df['volume'] = df['volume'].astype(np.float64)
    return df

def ccxt_fetch_data(start, end, pair = 'BTC/USDT', period = '1d'):
    start_dt = datetime.strptime(start, "%Y%m%d %H:%M:%S")
    end_dt = datetime.strptime(end, "%Y%m%d %H:%M:%S")
    start_timestamp = calendar.timegm(start_dt.utctimetuple())
    end_timestamp = calendar.timegm(end_dt.utctimetuple())
    if period == '1m':
        date_list = [datetime.utcfromtimestamp(float(time)) \
                     for time in range(start_timestamp, end_timestamp, 60*720)]
    else:
        date_list = [datetime.utcfromtimestamp(float(time)) \
                     for time in range(start_timestamp, end_timestamp, 60*1440)]
    df = ohlcv(date_list, pair, period)
    print('Actual end time: ' + str(df['time'].values[-1]))
    return df

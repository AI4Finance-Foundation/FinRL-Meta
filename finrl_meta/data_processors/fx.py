import os
import sys

import pandas as pd
from finta import TA


def add_time_feature(df, symbol, dt_col_name='time'):
    """read csv into df and index on time
    dt_col_name can be any unit from minutes to day. time is the index of pd
    must have pd columns [(time_col),(asset_col), Open,close,High,Low,day]
    data_process will add additional time information: time(index), minute, hour, weekday, week, month,year, day(since 1970)
    use StopLoss and ProfitTaken to simplify the action,
    feed a fixed StopLoss (SL = 200) and PT = SL * ratio
    action space: [action[0,2],ratio[0,10]]
    rewards is point
    
    add hourly, dayofweek(0-6, Sun-Sat)
    Args:
        file (str): file path/name.csv
    """

    df['symbol'] = symbol
    df['dt'] = pd.to_datetime(df[dt_col_name])
    df.index = df['dt']
    df['minute'] = df['dt'].dt.minute
    df['hour'] = df['dt'].dt.hour
    df['weekday'] = df['dt'].dt.dayofweek
    df['week'] = df['dt'].dt.isocalendar().week
    df['month'] = df['dt'].dt.month
    df['year'] = df['dt'].dt.year
    df['day'] = df['dt'].dt.day
    # df = df.set_index('dt')
    return df


# 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30','close_30_sma', 'close_60_sma'
def tech_indictors(df):
    df['macd'] = TA.MACD(df).SIGNAL
    df['boll_ub'] = TA.BBANDS(df).BB_UPPER
    df['boll_lb'] = TA.BBANDS(df).BB_LOWER
    df['rsi_30'] = TA.RSI(df, period=30)
    df['dx_30'] = TA.ADX(df, period=30)
    df['close_30_sma'] = TA.SMA(df, period=30)
    df['close_60_sma'] = TA.SMA(df, period=60)

    # fill NaN to 0
    df = df.fillna(0)
    print(f'--------df head - tail ----------------\n{df.head(3)}\n{df.tail(3)}\n---------------------------------')

    return df


def split_timeserious(df, key_ts='dt', freq='W', symbol=''):
    """import df and split into hour, daily, weekly, monthly based and 
    save into subfolder

    Args:
        df (pandas df with timestamp is part of multi index): 
        spliter (str): H, D, W, M, Y
    """

    freq_name = {'H': 'hourly', 'D': 'daily', 'W': 'weekly', 'M': 'monthly', 'Y': 'Yearly'}
    for count, (n, g) in enumerate(df.groupby(pd.Grouper(level=key_ts, freq=freq))):
        p = f'./data/split/{symbol}/{freq_name[freq]}'
        os.makedirs(p, exist_ok=True)
        # fname = f'{symbol}_{n:%Y%m%d}_{freq}_{count}.csv'
        fname = f'{symbol}_{n:%Y}_{count}.csv'
        fn = f'{p}/{fname}'
        print(f'save to:{fn}')
        g.reset_index(drop=True, inplace=True)
        g.drop(columns=['dt'], inplace=True)
        g.to_csv(fn)
    return


"""
python ./neo_finrl/data_processors/fx.py GBPUSD W ./data/raw/GBPUSD_raw.csv
symbol="GBPUSD"
freq = [H, D, W, M]
file .csv, column names [time, Open, High, Low, Close, Vol]
"""
if __name__ == '__main__':
    symbol, freq, file = sys.argv[1], sys.argv[2], sys.argv[3]
    print(f'processing... symbol:{symbol} freq:{freq} file:{file}')
    try:
        df = pd.read_csv(file)
    except Exception:
        print(f'No such file or directory: {file}')
        exit(0)
    df = add_time_feature(df, symbol=symbol, dt_col_name='time')
    df = tech_indictors(df)
    split_timeserious(df, freq=freq, symbol=symbol)
    print(f'Done!')

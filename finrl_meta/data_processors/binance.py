import datetime as dt
import json
import urllib
from typing import List

import pandas as pd
import requests
import os
import urllib
import zipfile
from datetime import *
from pathlib import Path

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

class Binance(_Base):
    def __init__(self, data_source: str, start_date: str, end_date: str, time_interval: str, **kwargs):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)
        self.url = "https://api.binance.com/api/v3/klines"
        self.time_diff = None

    # main functions
    def download_data(self, ticker_list: List[str]):
        startTime = dt.datetime.strptime(self.start_date, '%Y-%m-%d')
        endTime = dt.datetime.strptime(self.end_date, '%Y-%m-%d')

        self.start_time = self.stringify_dates(startTime)
        self.end_time = self.stringify_dates(endTime)
        self.interval = self.time_interval
        self.limit = 1440

        # 1s for now, will add support for variable time and variable tick soon
        if self.time_interval == "1s":
            # as per https://binance-docs.github.io/apidocs/spot/en/#compressed-aggregate-trades-list
            self.limit = 1000
            final_df = self.fetch_n_combine(self.start_date, self.end_date, ticker_list)
        else:
            final_df = pd.DataFrame()
            for i in ticker_list:
                hist_data = self.dataframe_with_limit(symbol=i)
                df = hist_data.iloc[:-1].dropna()
                df['tic'] = i
                final_df = final_df.append(df)
        self.dataframe = final_df

    # def clean_data(self, df):
    #     df = df.dropna()
    #     return df

    # def add_technical_indicator(self, df, tech_indicator_list):
    #     print('Adding self-defined technical indicators is NOT supported yet.')
    #     print('Use default: MACD, RSI, CCI, DX.')
    #     self.tech_indicator_list = ['open', 'high', 'low', 'close', 'volume',
    #                                 'macd', 'macd_signal', 'macd_hist',
    #                                 'rsi', 'cci', 'dx']
    #     final_df = pd.DataFrame()
    #     for i in df.tic.unique():
    #         tic_df = df[df.tic==i]
    #         tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = MACD(tic_df['close'], fastperiod=12,
    #                                                                             slowperiod=26, signalperiod=9)
    #         tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
    #         tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
    #         tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
    #         final_df = final_df.append(tic_df)
    #
    #     return final_df

    # def add_turbulence(self, df):
    #     print('Turbulence not supported yet. Return original DataFrame.')
    #
    #     return df

    # def add_vix(self, df):
    #     print('VIX is not applicable for cryptocurrencies. Return original DataFrame')
    #
    #     return df

    # def df_to_array(self, df, tech_indicator_list, if_vix):
    #     unique_ticker = df.tic.unique()
    #     price_array = np.column_stack([df[df.tic==tic].close for tic in unique_ticker])
    #     tech_array = np.hstack([df.loc[(df.tic==tic), tech_indicator_list] for tic in unique_ticker])
    #     assert price_array.shape[0] == tech_array.shape[0]
    #     return price_array, tech_array, np.array([])

    # helper functions
    def stringify_dates(self, date: dt.datetime):
        return str(int(date.timestamp() * 1000))

    def get_binance_bars(self, last_datetime, symbol):
        '''
        klines api returns data in the following order:
        open_time, open_price, high_price, low_price, close_price, 
        volume, close_time, quote_asset_volume, n_trades, 
        taker_buy_base_asset_volume, taker_buy_quote_asset_volume, 
        ignore
        '''
        req_params = {"symbol": symbol, 'interval': self.interval,
                      'startTime': last_datetime, 'endTime': self.end_time,
                      'limit': self.limit}
        # For debugging purposes, uncomment these lines and if they throw an error
        # then you may have an error in req_params
        # r = requests.get(self.url, params=req_params)
        # print(r.text) 
        df = pd.DataFrame(requests.get(self.url, params=req_params).json())

        if df.empty:
            return None

        df = df.iloc[:, 0:6]
        df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

        df[['open', 'high', 'low', 'close', 'volume']] = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

        # No stock split and dividend announcement, hence adjusted close is the same as close
        df['adjusted_close'] = df['close']
        df['datetime'] = df.datetime.apply(lambda x: dt.datetime.fromtimestamp(x / 1000.0))
        df.reset_index(drop=True, inplace=True)

        return df
    
    def get_newest_bars(self, symbols, interval, limit):
        merged_df = pd.DataFrame()
        for symbol in symbols:
            req_params = {"symbol": symbol, 'interval': interval, 'limit': limit}
    
            df = pd.DataFrame(requests.get(self.url, params=req_params).json(), index=range(limit))
            
            if df.empty:
                return None
            
            df = df.iloc[:, 0:6]
            df.columns = ['datetime','open','high','low','close','volume']
    
            df[['open','high','low','close','volume']] = df[['open','high','low','close','volume']].astype(float)
    
            # No stock split and dividend announcement, hence adjusted close is the same as close
            df['adjusted_close'] = df['close']
            df['datetime'] = df.datetime.apply(lambda x: dt.datetime.fromtimestamp(x/1000.0))
            df['tic'] = symbol
            df = df.rename(columns = {'datetime':'time'})
            df.reset_index(drop=True, inplace=True)
            merged_df = merged_df.append(df)
            
        return merged_df

    def dataframe_with_limit(self, symbol):
        final_df = pd.DataFrame()
        last_datetime = self.start_time
        while True:

            new_df = self.get_binance_bars(last_datetime, symbol)
            if new_df is None:
                break

            if last_datetime == self.end_time:
                break

            final_df = final_df.append(new_df)
            # last_datetime = max(new_df.datetime) + dt.timedelta(days=1)
            last_datetime = max(new_df.datetime)
            if isinstance(last_datetime,pd.Timestamp):
                last_datetime = last_datetime.to_pydatetime()

            if self.time_diff == None:
                self.time_diff = new_df.loc[1]['datetime'] - new_df.loc[0]['datetime']

            last_datetime = last_datetime + self.time_diff
            last_datetime = self.stringify_dates(last_datetime)

        date_value = final_df['datetime'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        final_df.insert(0, 'time', date_value)
        final_df.drop('datetime', inplace=True, axis=1)
        return final_df

    def get_download_url(self, file_url):
        return f"{BINANCE_BASE_URL}{file_url}"

    # downloads zip, unzips zip and deltes zip
    def download_n_unzip_file(self, base_path, file_name, date_range=None):
        download_path = f"{base_path}{file_name}"
        if date_range:
            date_range = date_range.replace(" ", "_")
            base_path = os.path.join(base_path, date_range)

        # raw_cache_dir = get_destination_dir("./cache/tick_raw")
        raw_cache_dir = "./cache/tick_raw"
        zip_save_path = os.path.join(raw_cache_dir, file_name)

        csv_name = os.path.splitext(file_name)[0] + ".csv"
        csv_save_path = os.path.join(raw_cache_dir, csv_name)

        fhandles = []

        if os.path.exists(csv_save_path):
            print(f"\nfile already exists! {csv_save_path}")
            return [csv_save_path]

        # make the "cache" directory (only)
        if not os.path.exists(raw_cache_dir):
            Path(raw_cache_dir).mkdir(parents=True, exist_ok=True)

        try:
            download_url = self.get_download_url(download_path)
            dl_file = urllib.request.urlopen(download_url)
            length = dl_file.getheader('content-length')
            if length:
                length = int(length)
                blocksize = max(4096, length // 100)

            with open(zip_save_path, 'wb') as out_file:
                dl_progress = 0
                print(f"\nFile Download: {zip_save_path}")
                while True:
                    buf = dl_file.read(blocksize)
                    if not buf:
                        break
                    out_file.write(buf)
                    # visuals
                    # dl_progress += len(buf)
                    # done = int(50 * dl_progress / length)
                    # sys.stdout.write("\r[%s%s]" % ('#' * done, '.' * (50-done)) )
                    # sys.stdout.flush()

            # unzip and delete zip
            file = zipfile.ZipFile(zip_save_path)
            with zipfile.ZipFile(zip_save_path) as zip:
                # guaranteed just 1 csv
                csvpath = zip.extract(zip.namelist()[0], raw_cache_dir)
                fhandles.append(csvpath)
            os.remove(zip_save_path)
            return fhandles

        except urllib.error.HTTPError:
            print(f"\nFile not found: {download_url}")

    def convert_to_date_object(self, d):
        year, month, day = [int(x) for x in d.split('-')]
        return date(year, month, day)

    def get_path(self, trading_type, market_data_type, time_period, symbol, interval=None):
        trading_type_path = 'data/spot'
        # currently just supporting spot
        if trading_type != 'spot':
            trading_type_path = f'data/futures/{trading_type}'
        return (
            f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/{interval}/'
            if interval is not None
            else f'{trading_type_path}/{time_period}/{market_data_type}/{symbol.upper()}/'
        )


    # helpers for manipulating tick level data (1s intervals)
    def download_daily_aggTrades(self, symbols, num_symbols, dates, start_date, end_date):
        trading_type = "spot"
        date_range = start_date + " " + end_date
        start_date = self.convert_to_date_object(start_date)
        end_date = self.convert_to_date_object(end_date)

        print(f"Found {num_symbols} symbols")

        map = {}
        for current, symbol in enumerate(symbols):
            map[symbol] = []
            print(f"[{current + 1}/{num_symbols}] - start download daily {symbol} aggTrades ")
            for date in dates:
                current_date = self.convert_to_date_object(date)
                if current_date >= start_date and current_date <= end_date:
                    path = self.get_path(trading_type, "aggTrades", "daily", symbol)
                    file_name = f"{symbol.upper()}-aggTrades-{date}.zip"
                    fhandle = self.download_n_unzip_file(path, file_name, date_range)
                    map[symbol] += fhandle
        return map

    def fetch_aggTrades(self, startDate: str, endDate: str, tickers: List[str]):
        # all valid symbols traded on v3 api
        response = urllib.request.urlopen("https://api.binance.com/api/v3/exchangeInfo").read()
        valid_symbols = list(map(lambda symbol: symbol['symbol'], json.loads(response)['symbols']))

        for tic in tickers:
            if tic not in valid_symbols:
                print(tic + " not a valid ticker, removing from download")
        tickers = list(set(tickers) & set(valid_symbols))
        num_symbols = len(tickers)
        # not adding tz yet
        # for ffill missing data on starting on first day 00:00:00 (if any)
        tminus1 = (self.convert_to_date_object(startDate) - dt.timedelta(1)).strftime('%Y-%m-%d')
        dates = pd.date_range(start=tminus1, end=endDate)
        dates = [date.strftime("%Y-%m-%d") for date in dates]
        return self.download_daily_aggTrades(tickers, num_symbols, dates, tminus1, endDate)

    # Dict[str]:List[str] -> pd.DataFrame
    def combine_raw(self, map):
        # same format as jingyang's current data format
        final_df = pd.DataFrame()
        # using AggTrades with headers from https://github.com/binance/binance-public-data/
        colNames = ["AggregatetradeId", "Price", "volume", "FirsttradeId", "LasttradeId", "time", "buyerWasMaker",
                    "tradeWasBestPriceMatch"]
        for tic in map.keys():
            security = pd.DataFrame()
            for i, csv in enumerate(map[tic]):
                dailyticks = pd.read_csv(csv,
                                         names=colNames,
                                         index_col=["time"],
                                         parse_dates=['time'],
                                         date_parser=lambda epoch: pd.to_datetime(epoch, unit='ms'))
                dailyfinal = dailyticks.resample('1s').agg({'Price': 'ohlc', 'volume': 'sum'})
                dailyfinal.columns = dailyfinal.columns.droplevel(0)
                # favor continuous series
                # dailyfinal.dropna(inplace=True)

                # implemented T-1 day ffill day start missing values 
                # guaranteed first csv is tminus1 day
                if i == 0:
                    tmr = dailyfinal.index[0].date() + dt.timedelta(1)
                    tmr_dt = dt.datetime.combine(tmr, dt.time.min)
                    last_time_stamp_dt = dailyfinal.index[-1].to_pydatetime()
                    s_delta = (tmr_dt - last_time_stamp_dt).seconds
                    lastsample = dailyfinal.iloc[-1:]
                    lastsample.index = lastsample.index.shift(s_delta, 's')
                else:
                    day_dt = dailyfinal.index[0].date()
                    day_str = day_dt.strftime("%Y-%m-%d")
                    nextday_str = (day_dt + dt.timedelta(1)).strftime("%Y-%m-%d")
                    if dailyfinal.index[0].second != 0:
                        # append last sample
                        dailyfinal = lastsample.append(dailyfinal)
                    # otherwise, just reindex and ffill
                    dailyfinal = dailyfinal.reindex(pd.date_range(day_str, nextday_str, freq="1s")[:-1], method='ffill')
                    # save reference info (guaranteed to be :59)
                    lastsample = dailyfinal.iloc[-1:]
                    lastsample.index = lastsample.index.shift(1, 's')

                    if dailyfinal.shape[0] != 86400:
                        raise ValueError("everyday should have 86400 datapoints")

                    # only save real startDate - endDate
                    security = security.append(dailyfinal)

            security.ffill(inplace=True)
            security['tic'] = tic
            final_df = final_df.append(security)
        return final_df

    def fetch_n_combine(self, startDate, endDate, tickers):
        # return combine_raw(fetchAggTrades(startDate, endDate, tickers))
        mapping = self.fetch_aggTrades(startDate, endDate, tickers)
        return self.combine_raw(mapping)

from finrl_meta.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
from finrl_meta.data_processors.processor_wrds import WrdsProcessor as Wrds
from finrl_meta.data_processors.processor_yahoofinance import YahooFinanceProcessor as YahooFinance
from finrl_meta.data_processors.processor_binance import BinanceProcessor as Binance
from finrl_meta.data_processors.processor_ricequant import RiceQuantProcessor as RiceQuant
from finrl_meta.data_processors.processor_joinquant import JoinquantProcessor
from finrl_meta.data_processors.processor_tusharepro import TushareProProcessor as Tusharepro
from finrl_meta.data_processors.processor_baostock import BaostockProcessor
import pandas as pd
import numpy as np
import os
import pickle

class DataProcessor():
    def __init__(self, data_source, start_date, end_date, time_interval, **kwargs):
        self.data_source = data_source
        self.start_date = start_date
        self.end_date = end_date
        self.time_interval = time_interval
        self.dataframe = pd.DataFrame()
        if self.data_source == 'alpaca':
            try:
                # users should input values: kwargs['API_KEY'], kwargs['API_SECRET'], kwargs['APCA_API_BASE_URL'], kwargs['API']
                self.processor = Alpaca(data_source, start_date, end_date, time_interval, **kwargs)
                print('Alpaca successfully connected')
            except:
                raise ValueError('Please input correct account info for alpaca!')
        elif self.data_source == "baostock":
            try:
                self.processor = BaostockProcessor(data_source, start_date, end_date, time_interval, **kwargs)
                print('Baostock successfully connected')
            except:
                raise ValueError('Please input correct account info for baostock!')
        elif self.data_source == "joinquant":
            try:
                # users should input values: kwargs['username'], kwargs['password']
                self.processor = JoinquantProcessor(data_source, start_date, end_date, time_interval, **kwargs)
                print('Joinquant successfully connected')
            except:
                raise ValueError('Please input correct account info for joinquant!')
        elif self.data_source == 'ricequant':
            try:
                # users should input values: kwargs['username'], kwargs['password']
                self.processor = RiceQuant(data_source, start_date, end_date, time_interval, **kwargs)
                print('Ricequant successfully connected')
            except:
                raise ValueError('Please input correct account info for ricequant!')
        elif self.data_source == 'wrds':
            try:
                # users should input values: kwargs['if_offline']
                self.processor = Wrds(data_source, start_date, end_date, time_interval, **kwargs)
                print('Wrds successfully connected')
            except:
                raise ValueError('Please input correct account info for wrds!')
        elif self.data_source == 'yahoofinance':
            try:
                self.processor = YahooFinance(data_source, start_date, end_date, time_interval, **kwargs)
                print('Yahoofinance successfully connected')
            except:
                raise ValueError('Please input correct account info for yahoofinance!')
        elif self.data_source == 'binance':
            try:
                self.processor = Binance(data_source, start_date, end_date, time_interval, **kwargs)
                print('Binance successfully connected')
            except:
                raise ValueError('Please input correct account info for binance!')
        elif self.data_source == "tusharepro":
            try:
                # users should input values: kwargs['token'], choose to input values: kwargs['adj']
                self.processor = Tusharepro(data_source, start_date, end_date, time_interval, **kwargs)
                print('tusharepro successfully connected')
            except:
                raise ValueError('Please input correct account info for tusharepro!')
        else:
            raise ValueError('Data source input is NOT supported yet.')

    def download_data(self, ticker_list):
        self.processor.download_data(ticker_list=ticker_list)
        self.dataframe = self.processor.dataframe


    def clean_data(self):
        self.processor.dataframe = self.dataframe
        self.processor.clean_data()
        self.dataframe = self.processor.dataframe

    def add_technical_indicator(self, tech_indicator_list, use_stockstats_or_talib: int = 0):
        self.tech_indicator_list = tech_indicator_list
        self.processor.add_technical_indicator(tech_indicator_list, use_stockstats_or_talib)
        self.dataframe = self.processor.dataframe

    def add_turbulence(self):
        self.processor.add_turbulence()
        self.dataframe = self.processor.dataframe

    def add_vix(self):
        self.processor.add_vix()
        self.dataframe = self.processor.dataframe

    def df_to_array(self, if_vix) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(self.tech_indicator_list, if_vix)
        # fill nan with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return price_array, tech_array, turbulence_array

    def run(self, ticker_list, technical_indicator_list, if_vix, cache=False, use_stockstats_or_talib: int = 0):

        if self.time_interval == "1s" and self.data_source != "binance":
            raise ValueError("Currently 1s interval data is only supported with 'binance' as data source")

        cache_filename = '_'.join(ticker_list + [self.data_source, self.start_date, self.end_date, self.time_interval]) + '.pickle'
        cache_dir = './cache'
        cache_path = os.path.join(cache_dir, cache_filename)

        if cache and os.path.isfile(cache_path):
            print(f'Using cached file {cache_path}')
            self.tech_indicator_list = technical_indicator_list
            with open(cache_path, 'rb') as handle:
                self.processor.dataframe = pickle.load(handle)
        else:
            self.download_data(ticker_list)
            self.clean_data()
            if cache:
                if not os.path.exists(cache_dir):
                    os.mkdir(cache_dir)
                with open(cache_path, 'wb') as handle:
                    pickle.dump(self.dataframe, handle, protocol=pickle.HIGHEST_PROTOCOL)


        self.add_technical_indicator(technical_indicator_list, use_stockstats_or_talib)
        if if_vix:
            self.add_vix()
        price_array, tech_array, turbulence_array = self.df_to_array(if_vix)
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return price_array, tech_array, turbulence_array


def test_joinquant():
    path_of_data = "../data"

    # TRADE_START_DATE = "2019-09-01"
    TRADE_START_DATE = "2020-09-01"
    TRADE_END_DATE = "2021-09-11"
    READ_DATA_FROM_LOCAL = 0

    TIME_INTERVAL = '1d'
    TECHNICAL_INDICATOR = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma']

    kwargs = {'username': 'xxx', 'password': 'xxx'}
    p = DataProcessor(data_source='joinquant', start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, time_interval=TIME_INTERVAL, **kwargs)

    # trade_days = p.get_trading_days(TRADE_START_DATE, TRADE_END_DATE)
    # stocknames = ["000612.XSHE", "601808.XSHG"]
    # data = p.download_data_for_stocks(
    #     stocknames, trade_days[0], trade_days[-1], READ_DATA_FROM_LOCAL, path_of_data
    # )
    ticker_list = ["000612.XSHE", "601808.XSHG"]

    p.download_data(ticker_list=ticker_list)

    p.clean_data()
    p.add_turbulence()
    p.add_technical_indicator(TECHNICAL_INDICATOR)
    p.add_vix()

    price_array, tech_array, turbulence_array = p.run(ticker_list, TECHNICAL_INDICATOR, if_vix=False, cache=True)


def test_binance():
    ticker_list = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT']
    start_date = '2021-09-01'
    end_date = '2021-09-20'
    time_interval = '5m'

    DP = DataProcessor('binance', start_date, end_date, time_interval)
    technical_indicator_list = ['macd', 'rsi', 'cci', 'dx']  # self-defined technical indicator list is NOT supported yet
    if_vix = False
    price_array, tech_array, turbulence_array = DP.run(ticker_list, technical_indicator_list, if_vix, cache=True, use_stockstats_or_talib=1)
    print(price_array.shape, tech_array.shape)

def test_yfinance():

    start_date = '2021-01-01'
    end_date = '2021-09-20'
    time_interval = '1D'

    DP = DataProcessor('yahoofinance', start_date, end_date, time_interval)
    ticker_list = [
        "MTX.DE",
        "MRK.DE",
        "LIN.DE",
        "ALV.DE",
        "VNA.DE",
    ]

    technical_indicator_list = ['macd', 'rsi', 'cci', 'dx']  # self-defined technical indicator list is NOT supported yet
    if_vix = False
    price_array, tech_array, turbulence_array = DP.run(ticker_list, technical_indicator_list, if_vix, cache=True, use_stockstats_or_talib=1)
    print(price_array.shape, tech_array.shape)

def test_baostock():
    TRADE_START_DATE = "2019-09-01"
    TRADE_END_DATE = "2021-09-11"

    TIME_INTERVAL = 'd'
    TECHNICAL_INDICATOR = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma']
    kwargs = {}
    p = DataProcessor(data_source='baostock', start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, time_interval=TIME_INTERVAL, **kwargs)

    # trade_days = p.get_trading_days(TRADE_START_DATE, TRADE_END_DATE)
    # stocknames = ["000612.XSHE", "601808.XSHG"]
    # data = p.download_data_for_stocks(
    #     stocknames, trade_days[0], trade_days[-1], READ_DATA_FROM_LOCAL, path_of_data
    # )
    # ticker_list = ["000612.XSHE", "601808.XSHG"]
    ticker_list = ["sh.600000"]

    p.download_data(ticker_list=ticker_list)

    p.clean_data()
    p.add_turbulence()
    p.add_technical_indicator(TECHNICAL_INDICATOR)
    p.add_vix()

    price_array, tech_array, turbulence_array = p.run(ticker_list, TECHNICAL_INDICATOR, if_vix=False, cache=True)
    pass

if __name__ == "__main__":
    # test_joinquant()
    #test_binance()
    # test_yfinance()
    test_baostock()
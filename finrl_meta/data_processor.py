import pandas as pd
import numpy as np


class DataProcessor():
    def __init__(self, data_source, **kwargs):
        if data_source == 'alpaca':
            from finrl_meta.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
            try:
                API_KEY= kwargs.get('API_KEY')
                API_SECRET= kwargs.get('API_SECRET')
                APCA_API_BASE_URL= kwargs.get('APCA_API_BASE_URL')
                self.processor = Alpaca(API_KEY, API_SECRET, APCA_API_BASE_URL)
                print('Alpaca successfully connected')
            except:
                raise ValueError('Please input correct account info for alpaca!')
        elif data_source == "joinquant":
            from finrl_meta.data_processors.processor_joinquant import JoinquantProcessor
            self.processor = JoinquantProcessor(data_source, **kwargs)

        elif data_source =='ricequant':
            from finrl_meta.data_processors.processor_ricequant import RiceQuantProcessor as RiceQuant
            try:
                username = kwargs.get('username')
                password = kwargs.get('password')
                self.processor = RiceQuant(username, password)
            except:
                self.processor = RiceQuant()
                
        elif data_source == 'wrds':
            from finrl_meta.data_processors.processor_wrds import WrdsProcessor as Wrds
            self.processor = Wrds()
            
        elif data_source == 'yahoofinance':
            from finrl_meta.data_processors.processor_yahoofinance import YahooFinanceProcessor as YahooFinance
            self.processor = YahooFinance()
        
        elif data_source =='binance':
            from finrl_meta.data_processors.processor_binance import BinanceProcessor as Binance
            self.processor = Binance()
        
        else:
            raise ValueError('Data source input is NOT supported yet.')
    
    def download_data(self, ticker_list, start_date, end_date, 
                      time_interval) -> pd.DataFrame:
        df = self.processor.download_data(ticker_list = ticker_list, 
                                          start_date = start_date, 
                                          end_date = end_date,
                                          time_interval = time_interval)
        return df
    
    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df)
        
        return df
    
    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)

        return df
    
    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)
        
        return df
    
    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)
        
        return df
    
    def df_to_array(self, df, if_vix) -> np.array:
        price_array,tech_array,turbulence_array = self.processor.df_to_array(df,
                                                self.tech_indicator_list,
                                                if_vix)
        #fill nan with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        
        return price_array, tech_array, turbulence_array
    
    def run(self, ticker_list, start_date, end_date, time_interval, 
            technical_indicator_list, if_vix):
        data = self.download_data(ticker_list, start_date, end_date, time_interval)
        data = self.clean_data(data)
        data = self.add_technical_indicator(data, technical_indicator_list)
        if if_vix:
            data = self.add_vix(data)
        price_array, tech_array, turbulence_array = self.df_to_array(data, if_vix)
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0

        return price_array, tech_array, turbulence_array

def test_joinquant():
    path_of_data = "../data"

    TRADE_START_DATE = "2019-09-01"
    TRADE_END_DATE = "2021-09-11"
    READ_DATA_FROM_LOCAL = 0

    kwargs = {}
    kwargs['username'] = "xxx"  # should input your username
    kwargs['password'] = "xxx"  # should input your password
    p = DataProcessor(data_source='joinquant', **kwargs)

    # trade_days = p.calc_trade_days_by_joinquant(TRADE_START_DATE, TRADE_END_DATE)
    # stocknames = ["000612.XSHE", "601808.XSHG"]
    # data = p.download_data_for_stocks(
    #     stocknames, trade_days[0], trade_days[-1], READ_DATA_FROM_LOCAL, path_of_data
    # )
    ticker_list = ["000612.XSHE", "601808.XSHG"]

    data2 = p.download_data(ticker_list=ticker_list, start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, time_interval='1D')
    # data3 = e.clean_data(data2)
    data4 = p.add_turbulence(data2)
    data6 = p.add_technical_indicator(data4, ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'])
    # data5 = e.add_vix(data4)

    pass

if __name__ == "__main__":
    test_joinquant()
    pass

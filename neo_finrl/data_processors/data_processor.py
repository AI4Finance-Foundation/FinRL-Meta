from alpaca_engineer import AlpacaEngineer as AE
from ccxt_engineer import CCXTEngineer as CE
from joinquant_engineer import JoinQuantEngineer as JE
from quantconnect_engineer import QuantConnectEngineer as QE
from wrds_engineer import WrdsEngineer as WE
from yahoofinance_engineer import YahooFinanceEngineer as YE
import pandas as pd
import numpy as np

class DataProcessor():
    def __init__(self, data_source):
        if data_source == 'alpaca':
            self.processor = AE
        elif data_source == 'ccxt':
            self.processor = CE
        elif data_source == 'joinquant':
            self.processor = JE
        elif data_source == 'quantconnect':
            self.processor = QE
        elif data_source == 'wrds':
            self.processor = WE
        elif data_source == 'yahoofinance':
            self.processor = YE
    
    def data_fetch(self,stock_list=['AAPL'], start_date='2021-05-10',
                   end_date='2021-05-10',time_interval='1Min') -> pd.DataFrame:
        df = self.processor.data_fetch(self, stock_list, start_date, end_date,
                                       time_interval)
        return df
    
    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df)
        
        return df
    
    def add_technical_indicators(self, df, tech_indicator_list = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30',
            'close_30_sma', 'close_60_sma']) -> pd.DataFrame:
        df = self.processor.add_technical_indicator(df, tech_indicator_list)
        
        return df
    
    def add_turbulence(self,df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)
        
        return df
    
    def df_to_ary(self, df) -> np.array:
        price_ary,tech_ary,turbulence_ary = self.df_to_ary(df)
        
        return price_ary,tech_ary,turbulence_ary
        
        
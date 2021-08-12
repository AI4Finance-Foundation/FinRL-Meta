from neo_finrl.data_processors.processor_alpaca import AlpacaProcessor as Alpaca
from neo_finrl.data_processors.processor_wrds import WrdsProcessor as Wrds
from neo_finrl.data_processors.processor_yahoofinance import YahooFinanceProcessor as YahooFinance
import pandas as pd
import numpy as np

class DataProcessor():
    def __init__(self, data_source, **kwargs):
        if data_source == 'alpaca':
            
            try:
                API_KEY= kwargs.get('API_KEY')
                API_SECRET= kwargs.get('API_SECRET')
                APCA_API_BASE_URL= kwargs.get('APCA_API_BASE_URL')
                self.processor = Alpaca(API_KEY, API_SECRET, APCA_API_BASE_URL)
                print('alpaca successfully connect')
            except:
                raise ValueError('Please input correct account info for alpaca!')
                
        elif data_source == 'wrds':
            self.processor = Wrds
            
        elif data_source == 'yahoofinance':
            self.processor = YahooFinance
        
        else:
            raise ValueError('Data source input is NOT supported yet.')
    
    def download_data(self, ticker_list, start_date, end_date, 
                      time_interval='1D') -> pd.DataFrame:
        df = self.processor.download_data(ticker_list, start_date, end_date,
                                       time_interval)
        return df
    
    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df)
        
        return df
    
    def add_technical_indicators(self, df, tech_indicator_list) -> pd.DataFrame:
        df = self.processor.add_technical_indicator(df, tech_indicator_list)
        
        return df
    
    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)
        
        return df
    
    def df_to_array(self, df) -> np.array:
        price_array,tech_array,turbulence_array = self.df_to_array(df)
        
        return price_array,tech_array,turbulence_array
        

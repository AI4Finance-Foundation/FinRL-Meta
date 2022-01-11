
import numpy as np
import pandas as pd
import tushare as ts
from tqdm import tqdm
from stockstats import StockDataFrame as Sdf
import stockstats
from finrl_meta.data_processors.basic_processor import BasicProcessor
from typing import List
import time
import copy
import warnings
from talib.abstract import CCI, DX, MACD, RSI
warnings.filterwarnings("ignore")

class TushareProProcessor(BasicProcessor):
    """Provides methods for retrieving daily stock data from tusharepro API
    Attributes
    ----------
        start_date : str
            start date of the data
        end_date : str
            end date of the data
        ticker_list : list
            a list of stock tickers 
        token : str
            get from https://waditu.com/ after registration
        adj: str
            Whether to use adjusted closing price. Default is None. 
            If you want to use forward adjusted closing price or 前复权. pleses use 'qfq'
            If you want to use backward adjusted closing price or 后复权. pleses use 'hfq'
    Methods
    -------
    download_data()
        Fetches data from tusharepro API
    
    """
    def __init__(self, data_source: str, **kwargs):
        BasicProcessor.__init__(self, data_source, **kwargs)
        if  'token' not in kwargs.keys() :
            raise ValueError("pleses input token!")
        self.token=kwargs["token"]
        if  'adj' in kwargs.keys() :
            self.adj=kwargs["adj"]
            print(f"Using {self.adj} method")
        else:
            self.adj=None
            
    
    def get_data(self,id) -> pd.DataFrame: 
        dfb = ts.pro_bar(ts_code=id, start_date=self.start,end_date=self.end,adj=self.adj)
        #df1 = ts.pro_bar(ts_code=id, start_date=self.start_date,end_date='20180101')
        #dfb=pd.concat([df, df1], ignore_index=True)
        #print(dfb.shape)
        return dfb

    def download_data(self, ticker_list: List[str], start_date: str, end_date: str, time_interval: str) \
            -> pd.DataFrame:
        """Fetches data from tusharepro API
        Parameters
        ----------
        Returns
        -------
        `pd.DataFrame`
            7 columns: A tick symbol, date, open, high, low, close and volume 
            for the specified stock ticker
        """
        self.ticker_list = ticker_list
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        if self.time_interval!="1D":
            raise ValueError('not supported currently')
        
        ts.set_token(self.token)
        
        self.df=pd.DataFrame()
        for i in tqdm(self.ticker_list,total=len(self.ticker_list)):
            df_temp=self.get_data(i)
            self.df=self.df.append(df_temp)
            #print("{} ok".format(i))
            time.sleep(0.25)
        
        self.df.columns=['tic','date','open','high','low','close','pre_close','change','pct_chg','volume','amount']
        self.df = self.df.sort_values(by=['date','tic']).reset_index(drop=True)
        
        df=self.df[['tic', 'date' , 'open' , 'high' , 'low' , 'close' , 'volume' ]]
        df["date"]= pd.to_datetime(df["date"],format="%Y%m%d")
        df["day"] = df["date"].dt.dayofweek 
        df["date"] = df.date.apply(lambda x: x.strftime("%Y-%m-%d"))
        
        df = df.dropna()
        df = df.sort_values(by=['date','tic']).reset_index(drop=True)

        print("Shape of DataFrame: ", df.shape)

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        dfc=copy.deepcopy(df)
        
        dfcode=pd.DataFrame(columns=['tic'])
        dfdate=pd.DataFrame(columns=['date'])

        dfcode.tic=dfc.tic.unique()
        dfdate.date=dfc.date.unique()
        dfdate.sort_values(by="date",ascending=False,ignore_index=True,inplace=True)
        
        # the old pandas may not support pd.merge(how="cross")
        try: 
            df1=pd.merge(dfcode,dfdate,how="cross")
        except:
            print("Please wait for a few seconds...")
            df1=pd.DataFrame(columns=["tic","date"])
            for i in range(dfcode.shape[0]):
                for j in range(dfdate.shape[0]):
                    df1=df1.append(pd.DataFrame(data={"tic":dfcode.iat[i,0], "date":dfdate.iat[j,0]},index=[(i+1)*(j+1)-1]))
            
        df2=pd.merge(df1,dfc,how="left",on=["tic","date"])
        

        # back fill missing data then front fill
        df3=pd.DataFrame(columns=df2.columns)
        for i in self.ticker_list:
            df4=df2[df2.tic==i].fillna(method="bfill").fillna(method="ffill")
            df3=pd.concat([df3, df4], ignore_index=True)

        df3=df3.fillna(0)

        # reshape dataframe
        df3 = df3.sort_values(by=['date','tic']).reset_index(drop=True)

        print("Shape of DataFrame: ", df3.shape)

        return df3

    def add_technical_indicator(self, data: pd.DataFrame, tech_indicator_list: List[str], use_stockstats: bool=True) \
            -> pd.DataFrame:
        """
        calculate technical indicators
        use stockstats/talib package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        if "date" in df.columns.values.tolist():
            df = df.rename(columns={'date': 'time'})
        
        if self.data_source == "ccxt":
            df = df.rename(columns={'index': 'time'})

        # df = df.reset_index(drop=False)
        # df = df.drop(columns=["level_1"])
        # df = df.rename(columns={"level_0": "tic", "date": "time"})
        if use_stockstats:  # use stockstats
            stock = stockstats.StockDataFrame.retype(df.copy())
            unique_ticker = stock.tic.unique()
            #print(unique_ticker)
            for indicator in tech_indicator_list:
                indicator_df = pd.DataFrame()
                for i in range(len(unique_ticker)):
                    try:
                        temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                        temp_indicator = pd.DataFrame(temp_indicator)
                        temp_indicator["tic"] = unique_ticker[i]
                        temp_indicator["time"] = df[df.tic == unique_ticker[i]][
                            "time"
                        ].to_list()
                        indicator_df = indicator_df.append(
                            temp_indicator, ignore_index=True
                        )
                    except Exception as e:
                        print(e)
                #print(indicator_df)
                df = df.merge(
                    indicator_df[["tic", "time", indicator]], on=["tic", "time"], how="left"
                )
        else:  # use talib
            final_df = pd.DataFrame()
            for i in df.tic.unique():
                tic_df = df[df.tic == i]
                tic_df['macd'], tic_df['macd_signal'], tic_df['macd_hist'] = MACD(tic_df['close'], fastperiod=12,
                                                                                  slowperiod=26, signalperiod=9)
                tic_df['rsi'] = RSI(tic_df['close'], timeperiod=14)
                tic_df['cci'] = CCI(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
                tic_df['dx'] = DX(tic_df['high'], tic_df['low'], tic_df['close'], timeperiod=14)
                final_df = final_df.append(tic_df)
            df = final_df

        df = df.sort_values(by=["time", "tic"])
        df = df.dropna()
        print("Succesfully add technical indicators")
        return df

    def get_trading_days(self, start: str, end: str) -> List[str]:
        print('not supported currently!')
        return ['not supported currently!']
    
    # def add_turbulence(self, data: pd.DataFrame) \
    #         -> pd.DataFrame:
    #     print('not supported currently!')
    #     return pd.DataFrame(['not supported currently!'])
    #
    # def calculate_turbulence(self, data: pd.DataFrame, time_period: int = 252) \
    #         -> pd.DataFrame:
    #     print('not supported currently!')
    #     return pd.DataFrame(['not supported currently!'])
    
    def add_vix(self, data: pd.DataFrame) \
            -> pd.DataFrame:
        print('not supported currently!')
        return pd.DataFrame(['not supported currently!'])

    def df_to_array(self, df: pd.DataFrame, tech_indicator_list: List[str], if_vix: bool) \
            -> List[np.array]:
        print('not supported currently!')
        return pd.DataFrame(['not supported currently!'])
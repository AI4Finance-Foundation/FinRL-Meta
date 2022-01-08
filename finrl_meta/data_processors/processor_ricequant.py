import rqdatac as ricequant
import pandas as pd
from stockstats import StockDataFrame as Sdf
import numpy as np
from typing import List
# from basic_processor import BasicProcessor
from finrl_meta.data_processors.basic_processor import BasicProcessor

class RiceQuantProcessor(BasicProcessor):
    # def __init__(self, username = None, password = None):
    #     # initialize ricequant
    #     if username== None or password == None:
    #         ricequant.init()   #if the lisence is already set, you can init without username and password
    #     else:
    #         ricequant.init(username, password) #init with username and password
    def __init__(self, data_source: str, **kwargs):
        BasicProcessor.__init__(self, data_source, **kwargs)
        if kwargs['username']== None or kwargs['password'] == None:
            ricequant.init()   #if the lisence is already set, you can init without username and password
        else:
            ricequant.init(kwargs['username'], kwargs['password']) #init with username and password
            
    def download_data(self, ticker_list: List[str], start_date: str, end_date: str, time_interval: str) -> pd.DataFrame:
        # download data by calling RiceQuant API
        dataframe = ricequant.get_price(ticker_list, frequency = time_interval, 
                            start_date = start_date, end_date = end_date)
        
        return dataframe
    
    def clean_data(self, df) -> pd.DataFrame:
        ''' RiceQuant data is already cleaned, we only need to transform data format here.
        No need for filling NaN data'''
        df = df.copy()
        # raw df uses multi-index (tic,time), reset it to single index (time) 
        df = df.reset_index(level=[0,1])
        # rename column order_book_id to tic
        # df = df.rename(columns={'order_book_id':'tic', 'datetime':'time'})
        # reserve columns needed
        df = df[['tic','time','open','high','low','close','volume']]
        # check if there is NaN values
        assert not df.isnull().values.any()
        
        return df 
    
    # def add_technical_indicator(self, df, tech_indicator_list = [
    #         'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30',
    #         'close_30_sma', 'close_60_sma']):
    #     df = df.copy()
    #     df = df.rename(columns={'time':'date'})
    #     df = df.sort_values(by=['tic', 'date'])
    #     stock = Sdf.retype(df.copy())
    #     unique_ticker = stock.tic.unique()
    #     tech_indicator_list = tech_indicator_list
    #
    #     for indicator in tech_indicator_list:
    #         indicator_df = pd.DataFrame()
    #         for i in range(len(unique_ticker)):
    #             # print(unique_ticker[i], i)
    #             temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
    #             temp_indicator = pd.DataFrame(temp_indicator)
    #             temp_indicator['tic'] = unique_ticker[i]
    #             # print(len(df[df.tic == unique_ticker[i]]['date'].to_list()))
    #             temp_indicator['date'] = df[df.tic == unique_ticker[i]]['date'].to_list()
    #             indicator_df = indicator_df.append(
    #                 temp_indicator, ignore_index=True
    #             )
    #         df = df.merge(indicator_df[['tic', 'date', indicator]], on=['tic', 'date'], how='left')
    #     df = df.sort_values(by=['date', 'tic'])
    #     df = df.rename(columns={'date':'time'})
    #     print('Succesfully add technical indicators')
    #     return df
    
    def add_vix(self, data):
        print('VIX is NOT applicable to China A-shares')
        return data
    
    def calculate_turbulence(self, data, time_period=252):
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()
    
        unique_date = df.date.unique()
        # start after a fixed time period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
                ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)
    
            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(filtered_hist_price, axis=0)
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)
    
        turbulence_index = pd.DataFrame(
            {"date": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index
    
    def add_turbulence(self,data, time_period=252):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(self, df, tech_indicator_list, if_vix):
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic==tic][['close']].values
                tech_array = df[df.tic==tic][tech_indicator_list].values
                #risk_array = df[df.tic==tic]['turbulence'].values
                if_first_time = False
            else:
                price_array = np.hstack([price_array, df[df.tic==tic][['close']].values])
                tech_array = np.hstack([tech_array, df[df.tic==tic][tech_indicator_list].values])
        print('Successfully transformed into array')
        return price_array, tech_array, None
    
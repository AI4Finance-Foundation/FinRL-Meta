import alpaca_trade_api as tradeapi
import pandas as pd
import numpy as np
from stockstats import StockDataFrame as Sdf
import trading_calendars as tc
import pytz

class AlpacaEngineer():
    def __init__(self, API_KEY=None, API_SECRET=None, APCA_API_BASE_URL=None, api=None):
        if api == None:
            try:
                self.api = tradeapi.REST(API_KEY, API_SECRET, APCA_API_BASE_URL, 'v2')
            except:
                raise ValueError('Wrong Account Info!')
        else:
            self.api = api
            
    def data_fetch(self,stock_list=['AAPL'], start_date='2021-05-10',
                   end_date='2021-05-10',time_interval='15Min') -> pd.DataFrame:
        
        self.start = start_date
        self.end = end_date
        
        NY = 'America/New_York'
        start_date = pd.Timestamp(start_date, tz=NY)
        end_date = pd.Timestamp(end_date, tz=NY) + pd.Timedelta(days=1)
        date = start_date
        data_df = pd.DataFrame()
        while date != end_date:
            start_time=(date + pd.Timedelta('09:30:00')).isoformat()
            end_time=(date + pd.Timedelta('15:59:00')).isoformat()
            for tic in stock_list:
                barset = self.api.get_barset([tic], time_interval, 
                                             start=start_time, end=end_time, 
                                             limit=500).df[tic]
                barset['tic'] = tic
                barset = barset.reset_index()
                data_df = data_df.append(barset)
            print(('Data before ') + end_time + ' is successfully fetched')
            date = date + pd.Timedelta(days=1)
            if date.isoformat()[-14:-6] == '01:00:00':
                date = date - pd.Timedelta('01:00:00')
            elif date.isoformat()[-14:-6] == '23:00:00':
                date = date + pd.Timedelta('01:00:00')
            if date.isoformat()[-14:-6] != '00:00:00':
                raise ValueError('Timezone Error')
        '''times = data_df['time'].values
        for i in range(len(times)):
            times[i] = str(times[i])
        data_df['time'] = times'''
        return data_df
    
    def clean_data(self, df):
        tic_list = np.unique(df.tic.values)
    
        trading_days = self.get_trading_days(start=self.start, end=self.end)
        
        times = []
        for day in trading_days:
            NY = 'America/New_York'
            current_time = pd.Timestamp(day+' 09:30:00').tz_localize(NY)
            for i in range(390):
                times.append(current_time)
                current_time += pd.Timedelta(minutes=1)
        
        new_df = pd.DataFrame()
        for tic in tic_list:
            tmp_df = pd.DataFrame(columns=['open','high','low','close','volume'], 
                                  index=times)
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]['time']] = tic_df.iloc[i][['open','high',
                                                                     'low','close',
                                                                     'volume']]
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]['close']) == 'nan':
                    previous_close = tmp_df.iloc[i-1]['close']
                    if str(previous_close) == 'nan':
                        raise ValueError
                    tmp_df.iloc[i] = [previous_close, previous_close, previous_close,
                                      previous_close, 0.0]
            tmp_df = tmp_df.astype(float)
            tmp_df['tic'] = tic
            new_df = new_df.append(tmp_df)
        
        print('Data clean finished!')
        
        return new_df
    
    def add_technical_indicators(self, df, stock_list, tech_indicator_list = [
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30',
            'close_30_sma', 'close_60_sma']):
        df = df.dropna()
        df = df.copy()
        column_list = [stock_list, ['open','high','low','close','volume']+(tech_indicator_list)]
        column = pd.MultiIndex.from_product(column_list)
        index_list = df.index
        dataset = pd.DataFrame(columns=column,index=index_list)
        for stock in stock_list:
            stock_column = pd.MultiIndex.from_product([[stock],['open','high','low','close','volume']])
            dataset[stock_column] = df[stock]
            temp_df = df[stock].reset_index().sort_values(by=['time'])
            temp_df = temp_df.rename(columns={'time':'date'})
            stock_df = Sdf.retype(temp_df.copy())  
            for indicator in tech_indicator_list:
                temp_indicator = stock_df[indicator].values.tolist()
                dataset[(stock,indicator)] = temp_indicator
        print('Succesfully add technical indicators')
        return dataset
    
    def df_to_ary(self, df, stock_list, tech_indicator_list=[
            'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30',
            'close_30_sma', 'close_60_sma']):
        price_array = df[pd.MultiIndex.from_product([stock_list,['close']])].values
        tech_array = df[pd.MultiIndex.from_product([stock_list,tech_indicator_list])].values
        return price_array, tech_array
    
    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data):
        """calculate turbulence index based on dow 30"""
        # can add other market assets
        df = data.copy()
        df = df.dropna()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = 60
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - 60])
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

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar('NYSE')
        df = nyse.sessions_in_range(pd.Timestamp(start,tz=pytz.UTC),
                                    pd.Timestamp(end,tz=pytz.UTC))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])
    
        return trading_days
        
        

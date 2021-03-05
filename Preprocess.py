# from finrl.preprocessing.preprocessors import pd, data_split, preprocess_data, add_turbulence
from finrl.config import config
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.marketdata.yahoodownloader import YahooDownloader
import pandas as pd
import numpy as np
import os


class Preprocessor:
    """
    Get Data and preprocess data
    support YahooData now
    """

    def __init__(self, start_date, end_date, data_source="yahoo"):
        self.start_date = start_date
        self.end_date = end_date
        self.data_source = data_source

    def fetch_yahoo_data(self):
        df = YahooDownloader(start_date=self.start_date,
                             end_date=self.end_date,
                             ticker_list=config.DOW_30_TICKER).fetch_data()
        df.to_csv("yahoo_data.csv")

    @staticmethod
    def process():
        processor = FeatureEngineer()
        # the following is same as part of run_model()
        preprocessed_path = "done_data.csv"
        if os.path.exists(preprocessed_path):
            data = pd.read_csv(preprocessed_path, index_col=0)
        else:
            data = pd.read_csv("yahoo_data.csv", index_col=0)
            data = processor.preprocess_data(data)
            data = processor.add_turbulence(data)
            data.to_csv(preprocessed_path)

        df = data
        rebalance_window = 63
        validation_window = 63
        i = rebalance_window + validation_window

        unique_trade_date = data[(data.datadate > 20151001) & (data.datadate <= 20200707)].datadate.unique()
        train__df = processor.data_split(df, start=20090000,
                                         end=unique_trade_date[i - rebalance_window - validation_window])
        # print(train__df) # df: DataFrame of Pandas

        train_ary = train__df.to_numpy().reshape((-1, 30, 12))
        '''state_dim = 1 + 6 * stock_dim, stock_dim=30
        n   item    index
        1   ACCOUNT -
        30  adjcp   2
        30  stock   -
        30  macd    7
        30  rsi     8
        30  cci     9
        30  adx     10
        '''
        data_ary = np.empty((train_ary.shape[0], 5, 30), dtype=np.float32)
        data_ary[:, 0] = train_ary[:, :, 2]  # adjcp
        data_ary[:, 1] = train_ary[:, :, 7]  # macd
        data_ary[:, 2] = train_ary[:, :, 8]  # rsi
        data_ary[:, 3] = train_ary[:, :, 9]  # cci
        data_ary[:, 4] = train_ary[:, :, 10]  # adx

        data_ary = data_ary.reshape((-1, 5 * 30))

        os.makedirs(npy_path[:npy_path.rfind('/')])
        np.save(npy_path, data_ary.astype(np.float16))  # save as float16 (0.5 MB), float32 (1.0 MB)
        print('| FinanceMultiStockEnv(): save in:', npy_path)

        return data_arys

    def preprocess(self):
        if self.data_source == "yahoo":
            raw_data_path = "yahoo_data.csv"
            if not os.path.exists(raw_data_path):
                fetchYahooData(self.start_date, self.end_date)
        self.process()


if __name__ == '__main__':
    data_processor = Preprocessor(start_date='2009-01-01', end_date='2021-01-01')
    data_processor.preprocess()

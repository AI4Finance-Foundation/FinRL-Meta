from finrl.config import config
from finrl.preprocessing.preprocessors import FeatureEngineer
import pandas as pd
import numpy as np

def preprocess_btc(df):
    df = df.rename(columns={'time':'date'})
    df.insert(loc=1, column='tic',value='BTC/USDT')
    df = df[['date','tic','open','high','low','close','volume']]
    print(df.head())
    
    
    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                        use_turbulence=False,
                        user_defined_feature = False)
            
    processed = fe.preprocess_data(df)
    ary = processed.values[:,range(2,15)]
    data_ary = ary.astype(np.float32)
    return data_ary

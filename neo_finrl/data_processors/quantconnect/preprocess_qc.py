import numpy as np
import pandas as pd
def preprocess(df, stock_list):
    df = df[['open','high','low','close','volume']]
    if_first_time = True
    for stock in stock_list:
        if if_first_time:
            ary = df.loc[stock].values
            if_first_time = False
        else:
            temp = df.loc[stock].values
            ary = np.hstack((ary,temp))
    return ary
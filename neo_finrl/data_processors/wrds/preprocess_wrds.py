import pandas as pd
import numpy as np

def preprocess(df):
    df = df.sort_values(by=['date','time_i','sym_root'])
    N = df.shape[0]
    assert N%3 == 0
    n = int(N/3)
    ary = np.zeros(shape=(n,3))
    for i in range(0,n):
        row_AAPL = df.iloc[3*i]
        row_IBM = df.iloc[3*i+1]
        row_SPY = df.iloc[3*i+2]
        ary[i][0] = row_AAPL['price']
        ary[i][1] = row_IBM['price']
        ary[i][2] = row_SPY['price']
    print('preprocess finished')
    return ary

        

import pandas as pd
import numpy as np

def preprocess(df, stock_list):
    n = len(stock_list)
    N = df.shape[0]
    assert N%n == 0
    d = int(N/n)
    stock1_ary = df.iloc[0:d,1:].values
    temp_ary = stock1_ary
    for j in range(1, n):
        stocki_ary = df.iloc[j*d:(j+1)*d,1:].values
        temp_ary = np.hstack((temp_ary,stocki_ary))
    return temp_ary

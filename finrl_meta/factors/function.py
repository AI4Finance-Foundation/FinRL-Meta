import numpy as np
import pandas as pd
import scipy


def filterNa(df):
    '''
    各特征NaN的个数
    '''
    naCount_dict = {}
    for col in df.columns.values:
        if df[col].dtypes==float:
            naCount_dict[col] = len(np.where(np.isnan(df[col]).values)[0])
    for i in naCount_dict:
        if(naCount_dict[i]>df.shape[0]/10):
            print(i, naCount_dict[i])
    return naCount_dict

def delNa(data, columns):
    '''
    保留指定因子, 并从中去除含有NaN的项
    '''
    df = data[columns]
    df.dropna(axis=1, how='all', inplace=True)
    df.dropna(axis=0, how='any', inplace=True)
    return df

def pearson_corr(df_, target):
    '''
    计算因子与目标值的皮尔逊系数相关性
    '''
    Pearson_dict={}
    df_.replace([np.inf, -np.inf], np.nan,inplace=True)
    df = df_.dropna(axis=1, how='all')
    df = df_.dropna(axis=0, how='any')
    for i in df.columns.values:
        if (type(df[i].values[-1])==float or type(df[i].values[-1])==np.float64) and i != 'alpha084' and i != 'alpha191-017':
            Pearson_dict[i] = scipy.stats.pearsonr(df[target].values, df[i].values)[0]
            
    df_Pearson = pd.DataFrame(data=Pearson_dict, index=[0]).T
    return abs(df_Pearson).sort_values(by=[0], ascending=False)

def spearmanr_corr(df_, target):
    '''
    计算因子与目标值的斯皮尔曼系数相关性
    '''
    Spearmanr_dict={}
    df_.replace([np.inf, -np.inf], np.nan,inplace=True)
    df = df_.dropna(axis=1, how='all')
    df = df_.dropna(axis=0, how='any')
    for i in df.columns.values:
        if (type(df[i].values[-1])==float or type(df[i].values[-1])==np.float64) and i != 'alpha084' and i != 'alpha191-017':
            Spearmanr_dict[i] = scipy.stats.spearmanr(df[target].values, df[i].values)[0]
            
    df_Spearmanr= pd.DataFrame(data=Spearmanr_dict, index=[0]).T
    return abs(df_Spearmanr).sort_values(by=[0], ascending=False)
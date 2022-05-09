import numpy as np
import statsmodels.api as sm
import pandas as pd
import scipy

def filterNa(df):
    '''
    各特征NaN的个数
    '''
    naCount_dict = {}
    for col in df.columns.values:
        if df[col].dtypes == float:
            naCount_dict[col] = len(np.where(np.isnan(df[col]).values)[0])
    for i in naCount_dict:
        if (naCount_dict[i] > df.shape[0] / 10):
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
    Pearson_dict = {}
    df_.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df_.dropna(axis=1, how='all')
    df = df_.dropna(axis=0, how='any')
    for i in df.columns.values:
        if (type(df[i].values[-1]) == float or type(df[i].values[-1]) == np.float64) and i != 'alpha084' and i != 'alpha191-017':
            Pearson_dict[i] = scipy.stats.pearsonr(df[target].values, df[i].values)[0]

    df_Pearson = pd.DataFrame(data=Pearson_dict, index=[0]).T
    return abs(df_Pearson).sort_values(by=[0], ascending=False)


def spearmanr_corr(df_, target):
    '''
    计算因子与目标值的斯皮尔曼系数相关性
    '''
    Spearmanr_dict = {}
    df_.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df_.dropna(axis=1, how='all')
    df = df_.dropna(axis=0, how='any')
    for i in df.columns.values:
        if (type(df[i].values[-1]) == float or type(df[i].values[-1]) == np.float64) and i != 'alpha084' and i != 'alpha191-017':
            Spearmanr_dict[i] = scipy.stats.spearmanr(df[target].values, df[i].values)[0]

    df_Spearmanr = pd.DataFrame(data=Spearmanr_dict, index=[0]).T
    return abs(df_Spearmanr).sort_values(by=[0], ascending=False)


# total_volume: 发行总量

def RD(N,D=3):   return np.round(N,D)        #四舍五入取3位小数 
def ABS(S):      return np.abs(S)            #返回N的绝对值
def MAX(S1,S2):  return np.maximum(S1,S2)    #序列max
def MIN(S1,S2):  return np.minimum(S1,S2)    #序列min
def IF(S,A,B):   return np.where(S,A,B)      #序列布尔判断 return=A  if S==True  else  B


def SUM(S, N):            #对序列求N天累计和，返回序列    N=0对序列所有依次求和         
    return pd.Series(S).rolling(N).sum().values if N>0 else pd.Series(S).cumsum().values  

def COUNT(S, N):                       # COUNT(CLOSE>O, N):  最近N天满足S_BOO的天数  True的天数
    return SUM(S,N)  

def REF(S, N=1):          #对序列整体下移动N,返回序列(shift后会产生NAN)    
    return pd.Series(S).shift(N).values  

def MA(S,N):              #求序列的N日简单移动平均值，返回序列                    
    return pd.Series(S).rolling(N).mean().values  

def EMA(S,N):             #指数移动平均,为了精度 S>4*N  EMA至少需要120周期     alpha=2/(span+1)    
    return pd.Series(S).ewm(span=N, adjust=False).mean().values    

def AVEDEV(S, N):         #平均绝对偏差  (序列与其平均值的绝对差的平均值)   
    return pd.Series(S).rolling(N).apply(lambda x: (np.abs(x - x.mean())).mean()).values 

def STD(S,N):             #求序列的N日标准差，返回序列    
    return  pd.Series(S).rolling(N).std(ddof=0).values   

def LLV(S,N):             #LLV(C, 5) 最近5天收盘最低价     
    return pd.Series(S).rolling(N).min().values  

def HHV(S,N):             #HHV(C, 5) 最近5天收盘最高价        
    return pd.Series(S).rolling(N).max().values 

def SMA(S, N, M=1):       #中国式的SMA,至少需要120周期才精确 (雪球180周期)    alpha=1/(1+com)    
    return pd.Series(S).ewm(alpha=M/N,adjust=False).mean().values           #com=N-M/M

def ATR(CLOSE,HIGH,LOW, N=20):                    #真实波动N日平均值
    TR = MAX(MAX((HIGH - LOW), ABS(REF(CLOSE, 1) - HIGH)), ABS(REF(CLOSE, 1) - LOW))
    return MA(TR, N)

def DMA(S, A):            #求S的动态移动平均，A作平滑因子,必须 0<A<1  (此为核心函数，非指标）
    if isinstance(A,(int,float)):  return pd.Series(S).ewm(alpha=A,adjust=False).mean().values    
    A=np.array(A)   
    A[np.isnan(A)]=1.0   
    Y= np.zeros(len(S))   
    Y[0]=S[0]     
    for i in range(1,len(S)): Y[i]=A[i]*S[i]+(1-A[i])*Y[i-1]      #A支持序列 by jqz1226         
    return Y       

class MomentumFactors:
    '''
    动量类因子
    '''
    # 5日乖离率 'ic_mean': '-0.045657'
    def BIAS5(self, close, N=5):
        #（收盘价-收盘价的N日简单平均）/ 收盘价的N日简单平均*100，在此n取5
        mac = MA(close, N)
        return (close - mac) / (mac * 100)
    
    # 10日乖离率  'ic_mean': '-0.043967'
    def BIAS10(self, close, N=10):
        # （收盘价-收盘价的N日简单平均）/ 收盘价的N日简单平均*100，在此n取10
        mac =  MA(close, N)
        return (close - mac) / (mac * 100)
    
    # 60日乖离率 'ic_mean': '-0.039533'
    def BIAS60(self, close, N=60):
        # （收盘价-收盘价的N日简单平均）/ 收盘价的N日简单平均*100，在此n取60
        mac =  MA(close, N)
        return (close - mac) / (mac * 100)
    
    # 当前股价除以过去一个月股价均值再减1 'ic_mean': '-0.039303'
    def Price1M(self, close, N=21):
        # 当日收盘价 / mean(过去一个月(21天)的收盘价) -1
        return close / close.rolling(N).mean() - 1
    
    # 当前股价除以过去三个月股价均值再减1 'ic_mean': '-0.034927'
    def Price3M(self, close, N=61):
        # 当日收盘价 / mean(过去三个月(61天)的收盘价) -1
        return close / close.rolling(N).mean() - 1

    # # 6日收盘价格与日期线性回归系数 'ic_mean': '-0.030587'
    # def PLRC6(self):
    #     #  计算 6 日收盘价格，与日期序号（1-6）的线性回归系数，(close / mean(close)) = beta * t + alpha
    #     return
    #
    # # 12日收盘价格与日期线性回归系数 'ic_mean': '0.047087'
    # def PLRC12():
    #     #  计算 12 日收盘价格，与日期序号（1-12）的线性回归系数，(close / mean(close)) = beta * t + alpha
    #     return
     
    # 6日变动速率（Price Rate of Change） 'ic_mean': '-0.030587'
    def ROC6(self, close, N=6):
        # ①AX=今天的收盘价—6天前的收盘价
        # ②BX=6天前的收盘价
        # ③ROC=AX/BX*100
        BX = close.shift(N)
        AX = close - BX
        return AX / (BX * 100)
    
    # 12日变动速率（Price Rate of Change） 'ic_mean': '-0.034748'
    def ROC12(self, close, N=12):
        # ①AX=今天的收盘价—12天前的收盘价 ②BX=12天前的收盘价 ③ROC=AX/BX*100
        BX = close.shift(N)
        AX = close - BX
        return AX / (BX * 100)
    
    # 20日变动速率（Price Rate of Change）  'ic_mean': '-0.031276'
    def ROC20(self, close, N=20):
        #  ①AX=今天的收盘价—20天前的收盘价 ②BX=20天前的收盘价 ③ROC=AX/BX*100
        BX = close.shift(N)
        AX = close - BX
        return AX / (BX * 100)
    
    # 单日价量趋势  'ic_mean': '-0.051037'
    def single_day_VPT(self, df):
        # （今日收盘价 - 昨日收盘价）/ 昨日收盘价 * 当日成交量  # (复权方法为基于当日前复权)
        sft = df['close_price'].shift(1)
        return (df['close_price'] - sft) / sft * df['volume']
    
    # 单日价量趋势6日均值 'ic_mean': '-0.032458'
    def single_day_VPT_6(self, df):
        # MA(single_day_VPT, 6)
        sft = df['close_price'].shift(1)
        return pd.Series(MA((df['close_price'] - sft) / sft * df['volume'], 6))
    
    # 单日价量趋势12均值 'ic_mean': '-0.031016'
    def single_day_VPT_12(self, df):
        # MA(single_day_VPT, 12)
        sft = df['close_price'].shift(1)
        return pd.Series(MA((df['close_price'] - sft) / sft * df['volume'], 12))
    
    # 10日顺势指标 'ic_mean': '-0.038179'
    def CCI10(self, df, N=10):
        #  CCI:=(TYP-MA(TYP,N))/(0.015*AVEDEV(TYP,N)) TYP:=(HIGH+LOW+CLOSE)/3 N:=10
        TYP = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        return (TYP - MA(TYP,N)) / (0.015 * AVEDEV(TYP,N))
    
    # 15日顺势指标 'ic_mean': '-0.035973'
    def CCI15(self, df, N=15):
        #  CCI:=(TYP-MA(TYP,N))/(0.015*AVEDEV(TYP,N)) TYP:=(HIGH+LOW+CLOSE)/3 N:=15
        TYP = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        return (TYP - MA(TYP, N)) / (0.015 * AVEDEV(TYP, N))
    
    # 20日顺势指标 'ic_mean': '-0.033437'
    def CCI20(self, df, N=20):
        # CCI:=(TYP-MA(TYP,N))/(0.015*AVEDEV(TYP,N)) TYP:=(HIGH+LOW+CLOSE)/3 N:=20
        TYP = (df['high_price'] + df['low_price'] + df['close_price']) / 3
        return (TYP - MA(TYP, N)) / (0.015 * AVEDEV(TYP, N))
    
    # 当前交易量相比过去1个月日均交易量 与过去过去20日日均收益率乘积 'ic_mean': '-0.032789'
    # def Volume1M(volume, profit):
    #     # 当日交易量 / 过去20日交易量MEAN * 过去20日收益率MEAN
    def Volume1M(self, df, N=21):
        # 当日交易量 / 过去20日交易量MEAN * 过去20日收益率MEAN
        return df['volume'] / df['volume'].rolling(N).mean() * df['target'].rolling(N).mean()
    
    # # 1减去 过去一个月收益率排名与股票总数的比值 'ic_mean': '0.030226'
    # def Rank1M(self, ):
    #     # 1-(Rank(个股20日收益) / 股票总数)
    #     return
    
    # 多头力道 'ic_mean': '-0.039968'
    def bull_power(self, df, timeperiod=13):
        return (df['high_price'] - EMA(df['close_price'], timeperiod)) / df['close_price']
    
class EmotionFactors:
    '''
    情绪类因子
    '''
    # 换手率： 某一段时期内的成交量/发行总股数×100%
    # 5日平均换手率 'ic_mean': '-0.044'
    def VOL5(self, S, total_volume, N=5):
        # 5日换手率均值
        S = S / total_volume
        return pd.Series(S).rolling(N).mean()

    # 10日平均换手率 'ic_mean': '-0.040'
    def VOL10(self, S, total_volume, N=10):
        # 10日换手率的均值
        S = S / total_volume
        return pd.Series(S).rolling(N).mean()

    # 20日平均换手率 'ic_mean': '-0.035'
    def VOL20(self, S, total_volume, N=20):
        # 20日换手率的均值,单位为%
        S = S / total_volume
        return pd.Series(S).rolling(N).mean()

    # 周期太长 暂时不要了
#     # 5日平均换手率与120日平均换手率 'ic_mean': '-0.039'
#     def DAVOL5(S):
#         # 5日平均换手率 / 120日平均换手率
#         return VOL5(S) / VOL5(S, N=120)

#     # 10日平均换手率与120日平均换手率之比 'ic_mean': '-0.033'
#     def DAVOL10(S):
#         # 10日平均换手率 / 120日平均换手率
#         return VOL10(S) / VOL5(S, N=120)

    # 10日成交量标准差 'ic_mean': '-0.037'
    def VSTD10(self, volume, N=10):
        # 10日成交量去标准差
        return pd.Series(STD(volume, N))

    # 20日成交量标准差 'ic_mean': '-0.033'
    def VSTD20(self, volume, N=20):
        # 20日成交量去标准差
        return pd.Series(STD(volume, N))

    # 6日成交金额的标准差 'ic_mean': '-0.044'
    def TVSTD6(self, df, N=6):
        # 6日成交额的标准差
        trades = df['close_price'] * df['volume']
        return pd.Series(STD(trades, N))

    # 20日成交金额的标准差 'ic_mean': '-0.038'
    def TVSTD20(self, df, N=20):
        # 20日成交额的标准差
        trades = df['close_price'] * df['volume']
        return pd.Series(STD(trades, N))
    
    # 成交量的5日指数移动平均 'ic_mean': '-0.035'
    def VEMA5(self, volume, N=5):
        # 
        return pd.Series(EMA(volume, N))

    # 成交量的10日指数移动平均 'ic_mean': '-0.032'
    def VEMA10(self, volume, N=10):
        # 
        return pd.Series(EMA(volume, N))

    # 12日成交量的移动平均值 'ic_mean': '-0.031'
    def VEMA12(self, volume, N=12):
        # 
        return pd.Series(EMA(volume, N))

    # 成交量震荡 'ic_mean': '-0.039'
    def VOSC(self, volume):
        # 'VEMA12'和'VEMA26'两者的差值，再求差值与'VEMA12'的比，最后将比值放大100倍，得到VOSC值
        ema12 =  EMA(volume, 12)
        return pd.Series((EMA(volume, 26) - ema12 / (ema12 * 100)))

    # 6日量变动速率指标 'ic_mean': '-0.032'
    def VROC6(self, volume, N=6):
        # 成交量减N日前的成交量，再除以N日前的成交量，放大100倍，得到VROC值 ，n=6
        sft = volume.shift(N)
        return pd.Series((volume - sft) / (sft * 100))

    # 12日量变动速率指标 'ic_mean': '-0.040'
    def VROC12(self, volume, N=12):
        # 成交量减N日前的成交量，再除以N日前的成交量，放大100倍，得到VROC值 ，n=12
        sft = volume.shift(N)
        return pd.Series((volume - sft) / (sft * 100))

    # 6日成交金额的移动平均值 'ic_mean': '-0.038'
    def TVMA6(self, df, N=6):
        # 6日成交金额的移动平均值
        trades = df['close_price'] * df['volume']
        return pd.Series(MA(trades, N))

    # 威廉变异离散量 'ic_mean': '-0.031'
    def WVAD(self, df, N=6):
        # (收盘价－开盘价)/(最高价－最低价)×成交量，再做加和，使用过去6个交易日的数据
        WVA = (df['close_price'] - df['open_price']) / (df['high_price'] - df['low_price']) * df['volume']
        return WVA.rolling(N).sum()

    # 换手率相对波动率 'ic_mean': '-0.042'
    def turnover_volatility(self, volume, total_volume, N=20):
        # 取20个交易日个股换手率的标准差
        turnover = volume / total_volume
        return pd.Series(STD(turnover, N))

    # 人气指标 'ic_mean': '-0.031'
    def AR(self, df, N=26):
        # AR=N日内（当日最高价—当日开市价）之和 / N日内（当日开市价—当日最低价）之和 * 100，n设定为26
        ho = (df['high_price'] - df['open_price']).rolling(N).sum()
        ol = (df['open_price'] - df['low_price']).rolling(N).sum()
        return ho / (ol * 100)
    
class extra_facters:
    '''
    特殊因子
    '''
    def RSRS(self, df, N):
        #用于记录回归后的beta值，即斜率
        ans = []
        #用于计算被决定系数加权修正后的贝塔值
        ans_rightdev= []
        # 一：RSRS指标的构建过程：
        # 1，取前N日的最高价序列与最低价序列。
        # 2，将两列数据按前述OLS线性回归模型拟合出当日的斜率值（Beta）。
        # 3，取前M日的斜率时间序列，计算当日斜率的标准分z。
        # 4，将z与拟合方程的决定系数相乘，作为当日RSRS指标值。
        X = sm.add_constant(df['low_price'])
        model = sm.OLS(df['high_price'], X)
        beta = model.fit().params
        r2=model.fit().rsquared
        ans.append(beta)
        # 计算标准化的RSRS指标
        # 计算均值序列    
        section = ans[-N:]
        # 计算均值序列
        mu = np.mean(section)
        # 计算标准化RSRS指标序列
        sigma = np.std(section)
        zscore = (section[-1]-mu)/sigma  
        #计算右偏RSRS标准分
        return pd.Series(zscore*beta*r2)
    
    # def VIX():
    #     pass

class GeneralFactors:
    '''
    常见因子
    '''
    def MACD(self, CLOSE, SHORT=12, LONG=26, M=9):             # EMA的关系，S取120日，和雪球小数点2位相同
        DIF = EMA(CLOSE,SHORT)-EMA(CLOSE,LONG)  
        DEA = EMA(DIF,M)      
        MACD=(DIF-DEA)*2
        return RD(MACD)

    def KDJ(self, df, KDJ_type, N=9, M1=3, M2=3):           # KDJ指标
        RSV = (df['close_price'] - LLV(df['low_price'], N)) / (HHV(df['high_price'], N) - LLV(df['low_price'], N)) * 100
        K = EMA(RSV, (M1*2-1))
        if KDJ_type == 'KDJ_K':
            return K
        elif KDJ_type == 'KDJ_D':
            return EMA(K, (M2*2-1))
        elif KDJ_type == 'KDJ_J':
            D = EMA(K, (M2*2-1))        
            return K*3-D*2

    def RSI(self, CLOSE, N=24):                           # RSI指标,和通达信小数点2位相同
        DIF = CLOSE-REF(CLOSE, 1) 
        return RD(SMA(MAX(DIF, 0), N) / SMA(ABS(DIF), N) * 100)  

    def WR(self, df, N=10):            #W&R 威廉指标
        WR = (HHV(df['high_price'], N) - df['close_price']) / (HHV(df['high_price'], N) - LLV(df['low_price'], N)) * 100
        return RD(WR)

    def BOLL(self, CLOSE, BOLL_type, N=20, P=2):                       #BOLL指标，布林带
        MID = MA(CLOSE, N) 
        if BOLL_type == 'BOLL_mid':
            return MID
        elif BOLL_type == 'BOLL_upper':
            return MID + STD(CLOSE, N) * P
        elif BOLL_type == 'BOLL_lower':
            return MID - STD(CLOSE, N) * P
        # return RD(UPPER), RD(MID), RD(LOWER)

    def PSY(self, CLOSE, PSY_type, N=12, M=6):
        PSY=COUNT(CLOSE>REF(CLOSE,1),N)/N*100
        if PSY_type == 'PSY':
            return PSY
        elif PSY_type == 'PSYMA':
            return MA(PSY,M)
        # return RD(PSY), RD(PSYMA)

    def ATR(self, df, N=20):                    #真实波动N日平均值
        TR = MAX(MAX((df['high_price'] - df['low_price']), ABS(REF(df['close_price'], 1) - df['high_price'])), ABS(REF(df['close_price'], 1) - df['low_price']))
        return MA(TR, N)

    def BBI(self, CLOSE,M1=3,M2=6,M3=12,M4=20):             #BBI多空指标
        return (MA(CLOSE,M1)+MA(CLOSE,M2)+MA(CLOSE,M3)+MA(CLOSE,M4))/4    

    def DMI(self, df, DMI_type, M1=14,M2=6):               #动向指标：结果和同花顺，通达信完全一致
        TR = SUM(MAX(MAX(df['high_price'] - df['low_price'], ABS(df['high_price'] - REF(df['close_price'], 1))), ABS(df['low_price'] - REF(df['close_price'], 1))), M1)
        HD = df['high_price'] - REF(df['high_price'], 1)
        LD = REF(df['low_price'], 1) - df['low_price']
        DMP = SUM(IF((HD > 0) & (HD > LD), HD, 0), M1)
        DMM = SUM(IF((LD > 0) & (LD > HD), LD, 0), M1)
        PDI = DMP * 100 / TR
        MDI = DMM * 100 / TR
        if DMI_type == 'DMI_PDI':
            return PDI
        elif DMI_type == 'DMI_MDI':
            return MDI
        elif DMI_type == 'DMI_ADX':
            return MA(ABS(MDI - PDI) / (PDI + MDI) * 100, M2)
        elif DMI_type == 'DMI_ADXR':
            ADX = MA(ABS(MDI - PDI) / (PDI + MDI) * 100, M2)
            return (ADX + REF(ADX, M2)) / 2
        # return PDI, MDI, ADX, ADXR  

    def TAQ(self, df, TAQ_type, N=6):                               #唐安奇通道(海龟)交易指标，大道至简，能穿越牛熊
        UP=HHV(df['high_price'], N)
        DOWN=LLV(df['low_price'], N)
        #MID=(UP+DOWN)/2
        if TAQ_type == 'TAQ_UP':
            return UP
        elif TAQ_type == 'TAQ_DOWN':
            return DOWN
        elif TAQ_type == 'TAQ_MID':
            return (UP+DOWN)/2
        # return UP,MID,DOWN

    def KTN(self, df, KTN_type, N=20, M=10):                 #肯特纳交易通道, N选20日，ATR选10日
        MID=EMA((df['high_price'] + df['low_price'] + df['close_price'])/3, N)
        if KTN_type == 'KTN_mid':
            return MID
        elif KTN_type == 'KTN_upper':
            ATRN=ATR(df['close_price'], df['high_price'], df['low_price'], M)
            return MID+2*ATRN
        elif KTN_type == 'KTN_lower':
            ATRN=ATR(df['close_price'], df['high_price'], df['low_price'], M)
            return MID-2*ATRN
        # return UPPER,MID,LOWER       

    def TRIX(self, CLOSE, TRIX_type, M1=12, M2=20):                      #三重指数平滑平均线
        TR = EMA(EMA(EMA(CLOSE, M1), M1), M1)
        TRIX = (TR - REF(TR, 1)) / REF(TR, 1) * 100
        if TRIX_type == 'TRIX':
            return TRIX
        elif TRIX_type == 'TRMA':
            return MA(TRIX, M2)
        # return TRIX, TRMA

    def VR(self, df, M1=26):                            #VR容量比率
        LC = REF(df['close_price'], 1)
        return SUM(IF(df['close_price'] > LC, df['volume'], 0), M1) / SUM(IF(df['close_price'] <= LC, df['volume'], 0), M1) * 100

    def EMV(self, df, EMV_type, N=14, M=9):                     #简易波动指标
        VOLUME=MA(df['volume'], N)/df['volume']
        MID=100*(df['high_price'] + df['low_price'] - REF(df['high_price'] + df['low_price'], 1))/(df['high_price'] + df['low_price'])
        EMV=MA(MID*VOLUME*(df['high_price'] - df['low_price'])/MA(df['high_price'] - df['low_price'], N), N)
        if EMV_type == 'EMV':
            return EMV
        elif EMV_type == 'MAEMV':
            return MA(EMV,M)
        # return EMV,MAEMV

    def DPO(self, CLOSE, DPO_type, M1=20, M2=10, M3=6):                  #区间震荡线
        DPO = CLOSE - REF(MA(CLOSE, M1), M2)
        if DPO_type == 'DPO':
            return DPO
        elif DPO_type == 'MADPO':
            return MA(DPO, M3)
        # return DPO, MADPO

    def BRAR(self, df, M1=26):                 #BRAR-ARBR 情绪指标
        # AR = SUM(HIGH - OPEN, M1) / SUM(OPEN - LOW, M1) * 100
        return SUM(MAX(0, df['high_price'] - REF(df['close_price'], 1)), M1) / SUM(MAX(0, REF(df['close_price'], 1) - df['low_price']), M1) * 100
        # return AR, BR

    def DFMA(self, CLOSE, N1=10, N2=50, M=10):                    #平行线差指标
        DIF=MA(CLOSE,N1)-MA(CLOSE,N2) 
        DIFMA=MA(DIF,M)   #通达信指标叫DMA 同花顺叫新DMA
        return DIFMA

    def MTM(self, CLOSE, MTM_type, N=12,M=6):                             #动量指标
        MTM=CLOSE-REF(CLOSE, N)
        if MTM_type == 'MTM':
            return MTM
        elif MTM_type == 'MTMMA':
            return MA(MTM,M)
        # return MTM,MTMMA

    def MASS(self, df, MASS_type, N1=9, N2=25, M=6):                   #梅斯线
        MASS=SUM(MA(df['high_price'] - df['low_price'], N1)/MA(MA(df['high_price'] - df['low_price'], N1), N1), N2)
        if MASS_type == 'MASS':
            return MASS
        elif MASS_type == 'MA_MASS':
            return MA(MASS,M)
        # MA_MASS=MA(MASS,M)
        # return MASS,MA_MASS

    def OBV(self, df):                                  #能量潮指标
        return SUM(IF(df['close_price']>REF(df['close_price'], 1), df['volume'], IF(df['close_price']<REF(df['close_price'], 1), - df['volume'], 0)), 0)/10000

    def MFI(self, df, N=14):                    #MFI指标是成交量的RSI指标
        TYP = (df['high_price'] + df['low_price'] + df['close_price'])/3
        V1 = SUM(IF(TYP>REF(TYP,1), TYP*df['volume'], 0), N)/SUM(IF(TYP<REF(TYP, 1), TYP*df['volume'], 0), N)  
        return 100-(100/(1+V1))     

    def ASI(self, df, ASI_type, M1=26, M2=10):            #振动升降指标
        LC = REF(df['close_price'], 1)
        AA = ABS(df['high_price'] - LC)
        BB = ABS(df['low_price'] - LC)
        CC = ABS(df['high_price'] - REF(df['low_price'],1))
        DD = ABS(LC-REF(df['open_price'], 1))
        R = IF( (AA>BB) & (AA>CC), AA+BB/2+DD/4, IF( (BB>CC) & (BB>AA),BB+AA/2+DD/4,CC+DD/4))
        X = (df['close_price'] - LC + (df['close_price'] - df['open_price'])/2 + LC - REF(df['open_price'], 1))
        SI = 16*X/R*MAX(AA, BB)
        ASI = SUM(SI, M1)
        if ASI_type == 'ASI':
            return ASI
        elif ASI_type == 'ASIT':
            return MA(ASI, M2)
        # return ASI,ASIT   

    def XSII(self, df, XSII_type, N=102, M=7):              #薛斯通道II
        AA  = MA((2*df['close_price'] + df['high_price'] + df['low_price'])/4, 5)            #最新版DMA才支持 2021-12-4
        # TD1 = AA*N/100   TD2 = AA*(200-N) / 100
        if XSII_type == 'XSII_TD1':
            return AA*N/100
        elif XSII_type == 'XSII_TD2':
            return AA*(200-N) / 100
        elif XSII_type == 'XSII_TD3':
            CC =  ABS((2*df['close_price'] + df['high_price'] + df['low_price'])/4 - MA(df['close_price'], 20))/MA(df['close_price'], 20)
            BB = df['close_price'].reset_index()['close_price']
            DD =  DMA(BB, CC)
            return (1+M/100)*DD
        elif XSII_type == 'XSII_TD4':
            CC =  ABS((2*df['close_price'] + df['high_price'] + df['low_price'])/4 - MA(df['close_price'], 20))/MA(df['close_price'], 20)
            BB = df['close_price'].reset_index()['close_price']
            DD =  DMA(BB, CC)
            return (1-M/100)*DD
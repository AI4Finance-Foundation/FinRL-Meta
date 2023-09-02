from config import cfg
from meta.data_processor import DataProcessor
import tushare as ts
import pandas as pd

def getFilteredShareList(df):
    '''
    column:
    ts_code,symbol,name,area,industry,market,exchange,list_status,list_date,is_hs
    根据market，选出主板的股票
    '''
    df = df[df["market"] == "主板"]
    return df




def getRawData(source, start_date, end_date, token):
    if source == "tushare":
        kwargs = {}
        kwargs["token"] = token
        kwargs["adj"] = "qfq"
        p = DataProcessor(
            data_source="tushare",
            start_date=start_date,
            end_date=end_date,
            time_interval="1d",
            **kwargs,
        )

        # download and clean
        p.download_data(ticker_list=ticker_list, save_path=cfg["save_path"])
        p.clean_data()
        p.fillna()
    
    if source == "baostock":

        kwargs = {}
        p = DataProcessor(
            data_source="baostock",
            start_date=start_date,
            end_date=end_date,
            time_interval="d",
            **kwargs,
        )

        ticker_list = ["600000.XSHG"]

        p.download_data(ticker_list=ticker_list)
        p.clean_data()
        p.add_turbulence()

    return p

def getIndicators(p, tech_indicators):
    # add_technical_indicator
    p.add_technical_indicator(tech_indicators)
    p.fillna()
    print(f"p.dataframe: {p.dataframe}")
    return p

def getDataset(p, start_date, end_date):
    train = p.data_split(p.dataframe, start_date, end_date)
    print(f"train.head(): {train.head()}")
    print(f"train.shape: {train.shape}")


def getStockTradeData():
    df = pro.daily(**{
            "ts_code": "",
            "trade_date": "",
            "start_date": "",
            "end_date": "",
            "offset": "",
            "limit": ""
        }, fields=[
            "ts_code",
            "trade_date",
            "open",
            "high",
            "low",
            "close",
            "pre_close",
            "change",
            "pct_chg",
            "vol",
            "amount"
        ])

if __name__ == "__main__":
    # set tushare token
    kwargs = {}
    kwargs["token"] = cfg["tushare_token"]

    # init processor
    p = DataProcessor(
        data_source="tushare",
        **kwargs
    )

    # get share list in 主板
    #p.download_share_list(save_path=cfg["save_path_tic_list"],
    #                     filter_item=["market_type"],
    #                     filter_value=["主板"])
    p.load_share_list(csv_path=cfg["save_path_tic_list"],
                      filter_item=["market_type"],
                        filter_value=["主板"])
    p.download_market_daily(p.sharelist,
                            cfg['TRAIN_START_DATE'],
                            cfg['TRADE_END_DATE'],
                            cfg['adj'],
                            cfg["save_path_data"])

from typing import List

# from basic_processor import BasicProcessor
from finrl_meta.data_processors.basic_processor import BasicProcessor

TIME_ZONE_SHANGHAI = 'Asia/Shanghai'  ## Hang Seng HSI, SSE, CSI
TIME_ZONE_USEASTERN = 'US/Eastern'  # Dow, Nasdaq, SP
TIME_ZONE_PARIS = 'Europe/Paris'  # CAC,
TIME_ZONE_BERLIN = 'Europe/Berlin'  # DAX, TECDAX, MDAX, SDAX
TIME_ZONE_JAKARTA = 'Asia/Jakarta'  # LQ45
TIME_ZONE_SELFDEFINED = 'xxx'  # If neither of the above is your time zone, you should define it, and set USE_TIME_ZONE_SELFDEFINED 1.
USE_TIME_ZONE_SELFDEFINED = 0  # 0 (default) or 1 (use the self defined)


## The code of this file is used in website, not locally.
class QuantConnectProcessor(BasicProcessor):
    def __init__(self, data_source: str, start_date, end_date, time_interval, **kwargs):
        super().__init__(data_source, start_date, end_date, time_interval, **kwargs)

    # def data_fetch(start_time, end_time, stock_list, resolution=Resolution.Daily) :
    #     #resolution: Daily, Hour, Minute, Second
    #     qb = QuantBook()
    #     for stock in stock_list:
    #         qb.AddEquity(stock)
    #     history = qb.History(qb.Securities.Keys, start_time, end_time, resolution)
    #     return history

    def download_data(self, ticker_list: List[str]):
        # self.time_zone = calc_time_zone(ticker_list, TIME_ZONE_SELFDEFINED, USE_TIME_ZONE_SELFDEFINED)

        # start_date = pd.Timestamp(start_date, tz=self.time_zone)
        # end_date = pd.Timestamp(end_date, tz=self.time_zone) + pd.Timedelta(days=1)
        qb = QuantBook()
        for stock in ticker_list:
            qb.AddEquity(stock)
        history = qb.History(qb.Securities.Keys, self.start_date, self.end_date, self.time_interval)

        # on later calls (for example for getting the vix) merge
        if self.dataframe.empty:
            self.dataframe = history
        else:
            self.dataframe = self.dataframe.merge(
                history, on=["tic", "time"], how="left"
            )


    # def preprocess(df, stock_list):
    #     df = df[['open','high','low','close','volume']]
    #     if_first_time = True
    #     for stock in stock_list:
    #         if if_first_time:
    #             ary = df.loc[stock].values
    #             if_first_time = False
    #         else:
    #             temp = df.loc[stock].values
    #             ary = np.hstack((ary,temp))
    #     return ary

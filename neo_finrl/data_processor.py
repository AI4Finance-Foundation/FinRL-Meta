import numpy as np
import pandas as pd
# from neo_finrl.data_processors.processor_alpaca import AlpacaProcessor
# from neo_finrl.data_processors.processor_ccxt import CcxtProcessor
from data_processors.processor_joinquant import JoinquantProcessor
# from neo_finrl.data_processors.processor_wrds import WrdsProcessor
# from neo_finrl.data_processors.processor_yahoofinance import YahooFinanceProcessor
from typing import List
TIME_INTERVAL = '1D'
# www.csindex.com.cn, for SSE and CSI adjustments
# SSE 50 Index constituents at 2019
SSE_50_TICKER = [
    "600000.XSHG",
    "600036.XSHG",
    "600104.XSHG",
    "600030.XSHG",
    "601628.XSHG",
    "601166.XSHG",
    "601318.XSHG",
    "601328.XSHG",
    "601088.XSHG",
    "601857.XSHG",
    "601601.XSHG",
    "601668.XSHG",
    "601288.XSHG",
    "601818.XSHG",
    "601989.XSHG",
    "601398.XSHG",
    "600048.XSHG",
    "600028.XSHG",
    "600050.XSHG",
    "600519.XSHG",
    "600016.XSHG",
    "600887.XSHG",
    "601688.XSHG",
    "601186.XSHG",
    "601988.XSHG",
    "601211.XSHG",
    "601336.XSHG",
    "600309.XSHG",
    "603993.XSHG",
    "600690.XSHG",
    "600276.XSHG",
    "600703.XSHG",
    "600585.XSHG",
    "603259.XSHG",
    "601888.XSHG",
    "601138.XSHG",
    "600196.XSHG",
    "601766.XSHG",
    "600340.XSHG",
    "601390.XSHG",
    "601939.XSHG",
    "601111.XSHG",
    "600029.XSHG",
    "600019.XSHG",
    "601229.XSHG",
    "601800.XSHG",
    "600547.XSHG",
    "601006.XSHG",
    "601360.XSHG",
    "600606.XSHG",
    "601319.XSHG",
    "600837.XSHG",
    "600031.XSHG",
    "601066.XSHG",
    "600009.XSHG",
    "601236.XSHG",
    "601012.XSHG",
    "600745.XSHG",
    "600588.XSHG",
    "601658.XSHG",
    "601816.XSHG",
    "603160.XSHG",
]

# CSI 300 Index constituents at 2019
CSI_300_TICKER = [
    "600000.XSHG",
    "600004.XSHG",
    "600009.XSHG",
    "600010.XSHG",
    "600011.XSHG",
    "600015.XSHG",
    "600016.XSHG",
    "600018.XSHG",
    "600019.XSHG",
    "600025.XSHG",
    "600027.XSHG",
    "600028.XSHG",
    "600029.XSHG",
    "600030.XSHG",
    "600031.XSHG",
    "600036.XSHG",
    "600038.XSHG",
    "600048.XSHG",
    "600050.XSHG",
    "600061.XSHG",
    "600066.XSHG",
    "600068.XSHG",
    "600085.XSHG",
    "600089.XSHG",
    "600104.XSHG",
    "600109.XSHG",
    "600111.XSHG",
    "600115.XSHG",
    "600118.XSHG",
    "600170.XSHG",
    "600176.XSHG",
    "600177.XSHG",
    "600183.XSHG",
    "600188.XSHG",
    "600196.XSHG",
    "600208.XSHG",
    "600219.XSHG",
    "600221.XSHG",
    "600233.XSHG",
    "600271.XSHG",
    "600276.XSHG",
    "600297.XSHG",
    "600299.XSHG",
    "600309.XSHG",
    "600332.XSHG",
    "600340.XSHG",
    "600346.XSHG",
    "600352.XSHG",
    "600362.XSHG",
    "600369.XSHG",
    "600372.XSHG",
    "600383.XSHG",
    "600390.XSHG",
    "600398.XSHG",
    "600406.XSHG",
    "600436.XSHG",
    "600438.XSHG",
    "600482.XSHG",
    "600487.XSHG",
    "600489.XSHG",
    "600498.XSHG",
    "600516.XSHG",
    "600519.XSHG",
    "600522.XSHG",
    "600547.XSHG",
    "600570.XSHG",
    "600583.XSHG",
    "600585.XSHG",
    "600588.XSHG",
    "600606.XSHG",
    "600637.XSHG",
    "600655.XSHG",
    "600660.XSHG",
    "600674.XSHG",
    "600690.XSHG",
    "600703.XSHG",
    "600705.XSHG",
    "600741.XSHG",
    "600745.XSHG",
    "600760.XSHG",
    "600795.XSHG",
    "600809.XSHG",
    "600837.XSHG",
    "600848.XSHG",
    "600867.XSHG",
    "600886.XSHG",
    "600887.XSHG",
    "600893.XSHG",
    "600900.XSHG",
    "600919.XSHG",
    "600926.XSHG",
    "600928.XSHG",
    "600958.XSHG",
    "600968.XSHG",
    "600977.XSHG",
    "600989.XSHG",
    "600998.XSHG",
    "600999.XSHG",
    "601006.XSHG",
    "601009.XSHG",
    "601012.XSHG",
    "601018.XSHG",
    "601021.XSHG",
    "601066.XSHG",
    "601077.XSHG",
    "601088.XSHG",
    "601100.XSHG",
    "601108.XSHG",
    "601111.XSHG",
    "601117.XSHG",
    "601138.XSHG",
    "601155.XSHG",
    "601162.XSHG",
    "601166.XSHG",
    "601169.XSHG",
    "601186.XSHG",
    "601198.XSHG",
    "601211.XSHG",
    "601212.XSHG",
    "601216.XSHG",
    "601225.XSHG",
    "601229.XSHG",
    "601231.XSHG",
    "601236.XSHG",
    "601238.XSHG",
    "601288.XSHG",
    "601298.XSHG",
    "601318.XSHG",
    "601319.XSHG",
    "601328.XSHG",
    "601336.XSHG",
    "601360.XSHG",
    "601377.XSHG",
    "601390.XSHG",
    "601398.XSHG",
    "601555.XSHG",
    "601577.XSHG",
    "601600.XSHG",
    "601601.XSHG",
    "601607.XSHG",
    "601618.XSHG",
    "601628.XSHG",
    "601633.XSHG",
    "601658.XSHG",
    "601668.XSHG",
    "601669.XSHG",
    "601688.XSHG",
    "601698.XSHG",
    "601727.XSHG",
    "601766.XSHG",
    "601788.XSHG",
    "601800.XSHG",
    "601808.XSHG",
    "601816.XSHG",
    "601818.XSHG",
    "601828.XSHG",
    "601838.XSHG",
    "601857.XSHG",
    "601877.XSHG",
    "601878.XSHG",
    "601881.XSHG",
    "601888.XSHG",
    "601898.XSHG",
    "601899.XSHG",
    "601901.XSHG",
    "601916.XSHG",
    "601919.XSHG",
    "601933.XSHG",
    "601939.XSHG",
    "601985.XSHG",
    "601988.XSHG",
    "601989.XSHG",
    "601992.XSHG",
    "601997.XSHG",
    "601998.XSHG",
    "603019.XSHG",
    "603156.XSHG",
    "603160.XSHG",
    "603259.XSHG",
    "603260.XSHG",
    "603288.XSHG",
    "603369.XSHG",
    "603501.XSHG",
    "603658.XSHG",
    "603799.XSHG",
    "603833.XSHG",
    "603899.XSHG",
    "603986.XSHG",
    "603993.XSHG",
    "000001.XSHE",
    "000002.XSHE",
    "000063.XSHE",
    "000066.XSHE",
    "000069.XSHE",
    "000100.XSHE",
    "000157.XSHE",
    "000166.XSHE",
    "000333.XSHE",
    "000338.XSHE",
    "000425.XSHE",
    "000538.XSHE",
    "000568.XSHE",
    "000596.XSHE",
    "000625.XSHE",
    "000627.XSHE",
    "000651.XSHE",
    "000656.XSHE",
    "000661.XSHE",
    "000671.XSHE",
    "000703.XSHE",
    "000708.XSHE",
    "000709.XSHE",
    "000723.XSHE",
    "000725.XSHE",
    "000728.XSHE",
    "000768.XSHE",
    "000776.XSHE",
    "000783.XSHE",
    "000786.XSHE",
    "000858.XSHE",
    "000860.XSHE",
    "000876.XSHE",
    "000895.XSHE",
    "000938.XSHE",
    "000961.XSHE",
    "000963.XSHE",
    "000977.XSHE",
    "001979.XSHE",
    "002001.XSHE",
    "002007.XSHE",
    "002008.XSHE",
    "002024.XSHE",
    "002027.XSHE",
    "002032.XSHE",
    "002044.XSHE",
    "002050.XSHE",
    "002120.XSHE",
    "002129.XSHE",
    "002142.XSHE",
    "002146.XSHE",
    "002153.XSHE",
    "002157.XSHE",
    "002179.XSHE",
    "002202.XSHE",
    "002230.XSHE",
    "002236.XSHE",
    "002241.XSHE",
    "002252.XSHE",
    "002271.XSHE",
    "002304.XSHE",
    "002311.XSHE",
    "002352.XSHE",
    "002371.XSHE",
    "002410.XSHE",
    "002415.XSHE",
    "002422.XSHE",
    "002456.XSHE",
    "002460.XSHE",
    "002463.XSHE",
    "002466.XSHE",
    "002468.XSHE",
    "002475.XSHE",
    "002493.XSHE",
    "002508.XSHE",
    "002555.XSHE",
    "002558.XSHE",
    "002594.XSHE",
    "002601.XSHE",
    "002602.XSHE",
    "002607.XSHE",
    "002624.XSHE",
    "002673.XSHE",
    "002714.XSHE",
    "002736.XSHE",
    "002739.XSHE",
    "002773.XSHE",
    "002841.XSHE",
    "002916.XSHE",
    "002938.XSHE",
    "002939.XSHE",
    "002945.XSHE",
    "002958.XSHE",
    "003816.XSHE",
    "300003.XSHE",
    "300014.XSHE",
    "300015.XSHE",
    "300033.XSHE",
    "300059.XSHE",
    "300122.XSHE",
    "300124.XSHE",
    "300136.XSHE",
    "300142.XSHE",
    "300144.XSHE",
    "300347.XSHE",
    "300408.XSHE",
    "300413.XSHE",
    "300433.XSHE",
    "300498.XSHE",
    "300601.XSHE",
    "300628.XSHE",
]
class DataProcessor:
    def __init__(self, data_source, **kwargs):
        self.time_interval = TIME_INTERVAL
        if data_source == "alpaca":
            try:
                API_KEY = kwargs.get("API_KEY")
                API_SECRET = kwargs.get("API_SECRET")
                APCA_API_BASE_URL = kwargs.get("APCA_API_BASE_URL")
                self.processor = AlpacaProcessor(API_KEY, API_SECRET, APCA_API_BASE_URL)
                print("AlpacaProcessor successfully connected")
            except BaseException:
                raise ValueError("Please input correct account info for alpaca!")

        elif data_source == "ccxt":
            self.processor = CcxtProcessor(data_source, **kwargs)

        elif data_source == "joinquant":
            self.processor = JoinquantProcessor(data_source, **kwargs)

        elif data_source == "wrds":
            self.processor = WrdsProcessor(data_source, **kwargs)

        elif data_source == "yahoofinance":
            self.processor = YahooFinanceProcessor(data_source, **kwargs)

        else:
            raise ValueError("Data source input is NOT supported yet.")

    def download_data(
        self, ticker_list: List[str], start_date: str, end_date: str, time_interval: str
    ) -> pd.DataFrame:
        df = self.processor.download_data(
            ticker_list=ticker_list,
            start_date=start_date,
            end_date=end_date,
            time_interval=time_interval,
        )
        return df

    def clean_data(self, df) -> pd.DataFrame:
        df = self.processor.clean_data(df)

        return df

    def add_technical_indicator(self, df, tech_indicator_list) -> pd.DataFrame:
        self.tech_indicator_list = tech_indicator_list
        df = self.processor.add_technical_indicator(df, tech_indicator_list)
        df = df.dropna()
        return df

    def add_turbulence(self, df) -> pd.DataFrame:
        df = self.processor.add_turbulence(df)

        return df

    def add_vix(self, df) -> pd.DataFrame:
        df = self.processor.add_vix(df)

        return df

    def df_to_array(self, df, if_vix) -> np.array:
        price_array, tech_array, turbulence_array = self.processor.df_to_array(
            df, self.tech_indicator_list, if_vix
        )
        # fill nan and inf values with 0 for technical indicators
        tech_nan_positions = np.isnan(tech_array)
        tech_array[tech_nan_positions] = 0
        tech_inf_positions = np.isinf(tech_array)
        tech_array[tech_inf_positions] = 0

        return price_array, tech_array, turbulence_array


def test_joinquant():
    path_of_data = "../" + "data"

    TRADE_START_DATE = "2020-08-03"
    TRADE_END_DATE = "2021-09-10"
    READ_DATA_FROM_LOCAL = 1

    username = "18117580099"  # should input your username
    password = "Bl2020quant"  # should input your password
    kwargs = {'username': username, 'password': password}
    e = DataProcessor(data_source="joinquant", **kwargs)

    # trade_days = e.calc_trade_days_by_joinquant(TRADE_START_DATE, TRADE_END_DATE)
    # stocknames = ["000612.XSHE", "601808.XSHG"]
    # data = e.download_data(
    #     stocknames, trade_days[0], trade_days[-1], READ_DATA_FROM_LOCAL, path_of_data
    # )

    data2 = e.download_data(ticker_list=SSE_50_TICKER, start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, time_interval='1D')
    data3 = e.clean_data(data2)
    data4 = e.add_technical_indicator(data2, ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'])
    # data5 = e.add_vix(data4)
    data6 = e.add_turbulence(data4)
    pass

def test_yahoo():
    TRADE_START_DATE = "2020-08-03"
    TRADE_END_DATE = "2021-09-10"
    READ_DATA_FROM_LOCAL = 1

    kwargs = {}
    e = DataProcessor(data_source="yahoofinance")

    data2 = e.download_data(ticker_list=["AXP", "AMGN"], start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, time_interval='1D')
    # data3 = e.clean_data(data2)
    data4 = e.add_technical_indicator(data2, ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'])
    # data5 = e.add_vix(data4)
    data6 = e.add_turbulence(data4)
    pass

if __name__ == "__main__":
    pass
    test_joinquant()
    # test_yahoo()






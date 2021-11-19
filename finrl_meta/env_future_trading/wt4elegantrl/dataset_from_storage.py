import pandas as pd
from ctypes import POINTER
from os import makedirs

from wtpy.wrapper.WtDtHelper import WtDataHelper
from wtpy.WtCoreDefs import WTSBarStruct
from wtpy import WtDtServo


class DataReader:
    df: pd.DataFrame

    def __len__(self) -> int:
        return len(self.df)

    def set_bars(self, data: pd.DataFrame) -> tuple:
        dtype = {
            'date': int,
            'time': int,
            'open': float,
            'high': float,
            'low': float,
            'close': float,
            'money': float,
            'vol': int,
            'hold': int,
            'diff': int,
        }
        self.df = data[dtype.keys()].astype(dtype)

    def get_bars(self, curBar: POINTER(WTSBarStruct), idx: int) -> bool:
        curBar.contents.date = self.df['date'].iloc[idx]
        curBar.contents.time = self.df['time'].iloc[idx]
        curBar.contents.open = self.df['open'].iloc[idx]
        curBar.contents.high = self.df['high'].iloc[idx]
        curBar.contents.low = self.df['low'].iloc[idx]
        curBar.contents.close = self.df['close'].iloc[idx]
        curBar.contents.money = self.df['money'].iloc[idx]
        curBar.contents.vol = self.df['vol'].iloc[idx]
        curBar.contents.hold = self.df['hold'].iloc[idx]
        curBar.contents.diff = self.df['diff'].iloc[idx]

        return True


reader: DataReader = DataReader()
helper: WtDataHelper = WtDataHelper()
dt: WtDtServo = WtDtServo()
dt.setBasefiles(
    commfile='./config/01commom/commodities.json',
    contractfile='./config/01commom/contracts.json',
    holidayfile='./config/01commom/holidays.json',
    sessionfile='./config/01commom/sessions.json',
    hotfile='./config/01commom/hots.json'
)
dt.setStorage('D:/Github/test/Storage/')
dt.commitConfig()

securities: tuple = (
    'CFFEX.IC.HOT',
    'CFFEX.IF.HOT',
    'CFFEX.IH.HOT',
    'CFFEX.T.HOT',
    'CFFEX.TF.HOT',
    'CZCE.RM.HOT',#
    'CZCE.JR.HOT',#
    'CZCE.AP.HOT',
    'CZCE.CF.HOT',
    'CZCE.MA.HOT',
    'CZCE.SF.HOT',
    'CZCE.SR.HOT',
    'CZCE.TA.HOT',#
    'CZCE.ZC.HOT',
    'DCE.a.HOT',
    'DCE.c.HOT',#
    'DCE.cs.HOT',#
    'DCE.i.HOT',
    'DCE.jd.HOT',
    'DCE.l.HOT',
    'DCE.m.HOT',#
    'DCE.p.HOT',
    'DCE.pp.HOT',
    'DCE.y.HOT',
    'DCE.rr.HOT',#
    'INE.sc.HOT',
    'INE.lu.HOT',
    'SHFE.ag.HOT',
    'SHFE.al.HOT',
    'SHFE.bu.HOT',
    'SHFE.cu.HOT',
    'SHFE.fu.HOT',
    'SHFE.hc.HOT',
    'SHFE.ni.HOT',
    'SHFE.pb.HOT',
    'SHFE.rb.HOT',
    'SHFE.zn.HOT',
)

for period, name in {'d1': 'day', 'm5': 'min5', }.items():#, 'm1': 'min1'
    for code in securities:
        print(period, code)
        path = './dataset/his/%s/%s' % (name, code.split('.')[0])
        makedirs(path, exist_ok=True)
        df = dt.get_bars(code, period=period, fromTime=201601011600,
                         endTime=202110131600).to_pandas()
        reader.set_bars(df)
        helper.trans_bars(barFile='%s/%s.dsb' % (path, code.replace('.HOT', '_HOT')), getter=reader.get_bars,
                          count=len(reader), period='d' if period == 'd1' else period)
        # break
    # break

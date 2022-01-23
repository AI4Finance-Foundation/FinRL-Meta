import datetime

import mplfinance as mpf
import pandas as pd


class TradingChart():
    """An ohlc trading visualization using matplotlib made to render tgym environment"""

    def __init__(self, df, transaction_history, **kwargs):
        self.ohlc = df[['_time', 'Open', 'High', 'Low', 'Close', 'symbol']].copy()
        self.ohlc = self.ohlc.rename(columns={'_time': 'Date'})
        self.ohlc.index = pd.DatetimeIndex(self.ohlc['Date'])
        self.transaction_history = transaction_history
        self.parameters = {"figscale": 6.0, "style": "nightclouds", "type": "hollow_and_filled",
                           "warn_too_much_data": 2000}
        self.symbols = self.ohlc['symbol'].unique()

    def transaction_line(self, symbol):
        _wlines = []
        _wcolors = []
        _llines = []
        _lcolors = []

        rewards = 0
        for tr in self.transaction_history:
            if tr["Symbol"] == symbol:
                rd = tr['Reward']
                rewards += rd
                if tr['ClosePrice'] > 0:
                    if tr['Type'] == 0:
                        if rd > 0:
                            _wlines.append([(tr['ActionTime'], tr['ActionPrice']), (tr['CloseTime'], tr['ClosePrice'])])
                            _wcolors.append('c')

                        else:
                            _llines.append([(tr['ActionTime'], tr['ActionPrice']), (tr['CloseTime'], tr['ClosePrice'])])
                            _lcolors.append('c')
                    elif tr['Type'] == 1:
                        if rd > 0:
                            _wlines.append([(tr['ActionTime'], tr['ActionPrice']), (tr['CloseTime'], tr['ClosePrice'])])
                            _wcolors.append('k')
                        else:
                            _llines.append([(tr['ActionTime'], tr['ActionPrice']), (tr['CloseTime'], tr['ClosePrice'])])
                            _lcolors.append('k')
        return _wlines, _wcolors, _llines, _lcolors, rewards

    def plot(self):

        for s in self.symbols:
            _wlines, _wcolors, _llines, _lcolors, rewards = self.transaction_line(s)
            _wseq = dict(alines=_wlines, colors=_wcolors)
            _lseq = dict(alines=_llines, colors=_lcolors, linestyle='--')
            _ohlc = self.ohlc.query(f'symbol=="{s}"')
            _style = mpf.make_mpf_style(base_mpl_style='seaborn', rc={'axes.grid': True})
            fig = mpf.figure(style=_style, figsize=(40, 20))
            ax1 = fig.subplot()
            ax2 = ax1.twinx()
            mpf.plot(_ohlc, alines=_lseq, mav=(10, 20), ax=ax1, type='ohlc', style='default')
            mpf.plot(_ohlc, alines=_wseq, ax=ax2, type='candle', style='yahoo', axtitle=f'{s} reward: {rewards}')
            fig.savefig(f'./data/log/{s}-{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}')

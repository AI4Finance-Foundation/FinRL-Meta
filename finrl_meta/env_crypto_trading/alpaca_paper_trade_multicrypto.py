# /*
#
# MIT License
#
# Copyright (c) 2021 AI4Finance
#
# Author: Berend Gort
#
# Year: 2021
#
# GitHub_link_author: https://github.com/Burntt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import math
import threading
import time
from datetime import datetime, timedelta

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import torch

from finrl_meta.data_processors.ccxt import Ccxt


class AlpacaPaperTradingMultiCrypto():

    def __init__(self, ticker_list, time_interval, drl_lib, agent, cwd, net_dim,
                 state_dim, action_dim, API_KEY, API_SECRET,
                 API_BASE_URL, tech_indicator_list,
                 max_stock=1e2, latency=None):
        # load agent
        self.drl_lib = drl_lib
        if agent != 'ppo':
            raise ValueError('Agent input is NOT supported yet.')

        if drl_lib == 'elegantrl':
            from elegantrl.agent import AgentPPO
            # load agent
            try:
                agent = AgentPPO()
                agent.init(net_dim, state_dim, action_dim)
                agent.save_or_load_agent(cwd=cwd, if_save=False)
                self.act = agent.act
                self.device = agent.device
            except:
                raise ValueError('Fail to load agent!')
        # connect to Alpaca trading API
        try:
            self.alpaca = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, 'v2')
            print('Connected to Alpaca API!')
        except:
            raise ValueError('Fail to connect Alpaca. Please check account info and internet connection.')

        # CCXT uses different time_interval than Alpaca (confusing I know)
        self.CCTX_time_interval = time_interval

        # read trading time interval
        if self.CCTX_time_interval == '1m':
            self.time_interval = 60
        elif self.CCTX_time_interval == '1h':
            self.time_interval = 60 ** 2
        elif self.CCTX_time_interval == '1d':
            self.time_interval = 60 ** 2 * 24
        else:
            raise ValueError('Time interval input is NOT supported yet.')

        # read trading settings
        self.tech_indicator_list = tech_indicator_list
        self.max_stock = max_stock
        self.previous_candles = 250
        self.lookback = 1
        self.action_dim = action_dim
        self.action_decimals = 2

        # initialize account
        self.stocks = np.asarray([0] * len(ticker_list))  # stocks holding
        self.stocks_cd = np.zeros_like(self.stocks)
        self.cash = None  # cash record
        self.stocks_df = pd.DataFrame(self.stocks, columns=['stocks'], index=ticker_list)
        self.asset_list = []
        self.price = np.asarray([0] * len(ticker_list))

        stockUniverse = []
        for stock in ticker_list:
            stock = stock.replace("USDT", "USD")
            stockUniverse.append(stock)

        self.ticker_list = ticker_list
        self.stockUniverse = stockUniverse
        self.equities = []

    def test_latency(self, test_times=10):
        total_time = 0
        for _ in range(test_times):
            time0 = time.time()
            self.get_state()
            time1 = time.time()
            temp_time = time1 - time0
            total_time += temp_time
        latency = total_time / test_times
        print('latency for data processing: ', latency)
        return latency

    def run(self):
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
            self.alpaca.cancel_order(order.id)
        while True:
            print('\n' + '#################### NEW CANDLE ####################')
            print('#################### NEW CANDLE ####################' + '\n')

            trade = threading.Thread(target=self.trade)
            trade.start()
            trade.join()
            last_equity = float(self.alpaca.get_account().last_equity)
            cur_time = time.time()
            self.equities.append([cur_time, last_equity])
            time.sleep(self.time_interval)

    def trade(self):

        # Get state
        state = self.get_state()

        # Get action
        if self.drl_lib != 'elegantrl':
            raise ValueError('The DRL library input is NOT supported yet. Please check your input.')

        with torch.no_grad():
            s_tensor = torch.as_tensor((state,), device=self.device)
            a_tensor = self.act(s_tensor)
            action = a_tensor.detach().cpu().numpy()[0]
        action = (action * self.max_stock).astype(float)
        print('\n' + 'ACTION:    ', action, '\n')
        # Normalize action
        action_norm_vector = []
        for price in self.price:
            print('PRICE:    ', price)
            x = math.floor(math.log(price, 10)) - 2
            print('MAG:      ', x)
            action_norm_vector.append(1 / ((10) ** x))
            print('NORM VEC: ', action_norm_vector)

        for i in range(self.action_dim):
            norm_vector_i = action_norm_vector[i]
            action[i] = action[i] * norm_vector_i

        print('\n' + 'NORMALIZED ACTION:    ', action, '\n')

        # Trade
        self.stocks_cd += 1
        min_action = 10 ** -(self.action_decimals)  # stock_cd
        for index in np.where(action < -min_action)[0]:  # sell_index:
            sell_num_shares = min(self.stocks[index], -action[index])

            qty = abs(float(sell_num_shares))
            qty = round(qty, self.action_decimals)
            print('SELL, qty:', qty)

            respSO = []
            tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'sell', respSO))
            tSubmitOrder.start()
            tSubmitOrder.join()
            self.cash = float(self.alpaca.get_account().cash)
            self.stocks_cd[index] = 0

        for index in np.where(action > min_action)[0]:  # buy_index:
            tmp_cash = max(self.cash, 0)
            print('current cash:', tmp_cash)
            # Adjusted part to accept decimal places up to two
            buy_num_shares = min(tmp_cash / self.price[index], abs(float(action[index])))

            qty = abs(float(buy_num_shares))
            qty = round(qty, self.action_decimals)
            print('BUY, qty:', qty)

            respSO = []
            tSubmitOrder = threading.Thread(target=self.submitOrder(qty, self.stockUniverse[index], 'buy', respSO))
            tSubmitOrder.start()
            tSubmitOrder.join()
            self.cash = float(self.alpaca.get_account().cash)
            self.stocks_cd[index] = 0

        print('Trade finished')

    def get_state(self):
        datetime_today = datetime.today()

        if self.CCTX_time_interval == '1m':
            start_date = (datetime_today - timedelta(minutes=self.previous_candles)).strftime("%Y%m%d %H:%M:%S")
            end_date = datetime.today().strftime("%Y%m%d %H:%M:%S")
        elif self.CCTX_time_interval == '1h':
            start_date = (datetime_today - timedelta(hours=self.previous_candles)).strftime("%Y%m%d %H:%M:%S")
            end_date = datetime.today().strftime("%Y%m%d %H:%M:%S")
        elif self.CCTX_time_interval == '1d':
            start_date = (datetime_today - timedelta(days=self.previous_candles)).strftime("%Y%m%d %H:%M:%S")
            end_date = datetime.today().strftime("%Y%m%d %H:%M:%S")

        print('fetching latest ' + str(self.previous_candles) + ' candles..')
        CCXT_instance = Ccxt()
        CCXT_instance.download_data(self.ticker_list, start_date, end_date, self.CCTX_time_interval)

        CCXT_instance.add_technical_indicators(self.ticker_list, self.tech_indicator_list)

        price_array, tech_array, _ = CCXT_instance.df_to_ary(self.ticker_list, self.tech_indicator_list)

        self.price_array = price_array
        self.tech_array = tech_array

        print('downloaded candles..')

        positions = self.alpaca.list_positions()
        stocks = [0] * len(self.stockUniverse)

        for position in positions:
            ind = self.stockUniverse.index(position.symbol)
            stocks[ind] = (abs(int(float(position.qty))))

        stocks = np.asarray(stocks, dtype=float)
        cash = float(self.alpaca.get_account().cash)
        self.cash = cash
        self.stocks = stocks

        # latest price and tech arrays
        self.price = price_array[-1]

        # Stack cash and stocks
        state = np.hstack((self.cash * 2 ** -18, self.stocks * 2 ** -3))
        for i in range(self.lookback):
            tech_i = self.tech_array[-1 - i]
            normalized_tech_i = tech_i * 2 ** -15
            state = np.hstack((state, normalized_tech_i)).astype(np.float32)

        print('\n' + 'STATE:')
        print(state)

        return state

    def submitOrder(self, qty, stock, side, resp):
        if (qty > 0):
            try:
                self.alpaca.submit_order(stock, qty, side, "market", "day")
                print("Market order of | "
                      + str(qty)
                      + " "
                      + stock
                      + " " + side + " | completed.")
                resp.append(True)
            except Exception as e:
                print('ALPACA API ERROR: ', e)
                print("Order of | " + str(qty) + " " + stock + " " + side + " | did not go through.")
                resp.append(False)
        else:
            print("Quantity is 0, order of | " + str(qty) + " " + stock + " " + side + " | not completed.")
            resp.append(True)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh

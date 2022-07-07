import datetime
import threading
import time

import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd

from meta.data_processors.alpaca import Alpaca


class AlpacaPaperTrading_sb3:
    def __init__(
        self,
        ticker_list,
        time_interval,
        agent,
        cwd,
        net_dim,
        state_dim,
        action_dim,
        API_KEY,
        API_SECRET,
        API_BASE_URL,
        tech_indicator_list,
        turbulence_thresh=30,
        max_stock=1e2,
    ):
        # load agent
        if agent == "ppo":
            from stable_baselines3 import PPO

            try:
                # load agent
                self.model = PPO.load(cwd)
                print("Successfully load model", cwd)
            except:
                raise ValueError("Fail to load agent!")
        else:
            raise ValueError("Agent input is NOT supported yet.")

        # connect to Alpaca trading API
        try:
            self.alpaca = tradeapi.REST(API_KEY, API_SECRET, API_BASE_URL, "v2")
        except:
            raise ValueError(
                "Fail to connect Alpaca. Please check account info and internet connection."
            )

        # read trading time interval
        if time_interval == "1s":
            self.time_interval = 1
        elif time_interval == "5s":
            self.time_interval = 5
        elif time_interval == "1Min":
            self.time_interval = 60
        elif time_interval == "5Min":
            self.time_interval = 60 * 5
        elif time_interval == "15Min":
            self.time_interval = 60 * 15
        else:
            raise ValueError("Time interval input is NOT supported yet.")

        # read trading settings
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_thresh = turbulence_thresh
        self.max_stock = max_stock

        # initialize account
        self.stocks = np.asarray([0] * len(ticker_list))  # stocks holding
        self.stocks_cd = np.zeros_like(self.stocks)
        self.cash = None  # cash record
        self.stocks_df = pd.DataFrame(
            self.stocks, columns=["stocks"], index=ticker_list
        )
        self.asset_list = []
        self.price = np.asarray([0] * len(ticker_list))
        self.stockUniverse = ticker_list
        self.turbulence_bool = 0
        self.equities = []

    def run(self):
        orders = self.alpaca.list_orders(status="open")
        for order in orders:
            self.alpaca.cancel_order(order.id)

        # Wait for market to open.
        print("Waiting for market to open...")
        tAMO = threading.Thread(target=self.awaitMarketOpen)
        tAMO.start()
        tAMO.join()
        print("Market opened.")
        while True:

            # Figure out when the market will close so we can prepare to sell beforehand.
            clock = self.alpaca.get_clock()
            closingTime = clock.next_close.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()
            currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            self.timeToClose = closingTime - currTime

            if self.timeToClose < (60):
                # Close all positions when 1 minutes til market close.
                print("Market closing soon. Stop trading.")
                break

                """# Close all positions when 1 minutes til market close.
                print("Market closing soon.  Closing positions.")

                positions = self.alpaca.list_positions()
                for position in positions:
                  if(position.side == 'long'):
                    orderSide = 'sell'
                  else:
                    orderSide = 'buy'
                  qty = abs(int(float(position.qty)))
                  respSO = []
                  tSubmitOrder = threading.Thread(target=self.submitOrder(qty, position.symbol, orderSide, respSO))
                  tSubmitOrder.start()
                  tSubmitOrder.join()

                # Run script again after market close for next trading day.
                print("Sleeping until market close (15 minutes).")
                time.sleep(60 * 15)"""

            else:
                trade = threading.Thread(target=self.trade)
                trade.start()
                trade.join()
                last_equity = float(self.alpaca.get_account().last_equity)
                cur_time = time.time()
                self.equities.append([cur_time, last_equity])
                np.save(
                    "paper_trading_records.npy",
                    np.asarray(self.equities, dtype=float),
                )
                time.sleep(self.time_interval)

    def awaitMarketOpen(self):
        isOpen = self.alpaca.get_clock().is_open
        while not isOpen:
            clock = self.alpaca.get_clock()
            openingTime = clock.next_open.replace(
                tzinfo=datetime.timezone.utc
            ).timestamp()
            currTime = clock.timestamp.replace(tzinfo=datetime.timezone.utc).timestamp()
            timeToOpen = int((openingTime - currTime) / 60)
            print(str(timeToOpen) + " minutes til market open.")
            time.sleep(60)
            isOpen = self.alpaca.get_clock().is_open

    def trade(self):
        state = self.get_state()
        action = self.model.predict(state)[0]
        action = (action * self.max_stock).astype(int)

        self.stocks_cd += 1
        if self.turbulence_bool == 0:
            min_action = 10  # stock_cd
            for index in np.where(action < -min_action)[0]:  # sell_index:
                sell_num_shares = min(self.stocks[index], -action[index])
                qty = abs(int(sell_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(
                        qty, self.stockUniverse[index], "sell", respSO
                    )
                )
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0

            for index in np.where(action > min_action)[0]:  # buy_index:
                if self.cash < 0:
                    tmp_cash = 0
                else:
                    tmp_cash = self.cash
                buy_num_shares = min(
                    tmp_cash // self.price[index], abs(int(action[index]))
                )
                qty = abs(int(buy_num_shares))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(
                        qty, self.stockUniverse[index], "buy", respSO
                    )
                )
                tSubmitOrder.start()
                tSubmitOrder.join()
                self.cash = float(self.alpaca.get_account().cash)
                self.stocks_cd[index] = 0

        else:  # sell all when turbulence
            positions = self.alpaca.list_positions()
            for position in positions:
                if position.side == "long":
                    orderSide = "sell"
                else:
                    orderSide = "buy"
                qty = abs(int(float(position.qty)))
                respSO = []
                tSubmitOrder = threading.Thread(
                    target=self.submitOrder(qty, position.symbol, orderSide, respSO)
                )
                tSubmitOrder.start()
                tSubmitOrder.join()

            self.stocks_cd[:] = 0

    def get_state(self):
        alpaca = Alpaca(api=self.alpaca)
        price, tech, turbulence = alpaca.fetch_latest_data(
            ticker_list=self.stockUniverse,
            time_interval="1Min",
            tech_indicator_list=self.tech_indicator_list,
        )
        turbulence_bool = 1 if turbulence >= self.turbulence_thresh else 0

        turbulence = (
            self.sigmoid_sign(turbulence, self.turbulence_thresh) * 2**-5
        ).astype(np.float32)

        tech = tech * 2**-7
        positions = self.alpaca.list_positions()
        stocks = [0] * len(self.stockUniverse)
        for position in positions:
            ind = self.stockUniverse.index(position.symbol)
            stocks[ind] = abs(int(float(position.qty)))

        stocks = np.asarray(stocks, dtype=float)
        cash = float(self.alpaca.get_account().cash)
        self.cash = cash
        self.stocks = stocks
        self.turbulence_bool = turbulence_bool
        self.price = price

        amount = np.array(max(self.cash, 1e4) * (2**-12), dtype=np.float32)
        scale = np.array(2**-6, dtype=np.float32)
        state = np.hstack(
            (
                amount,
                turbulence,
                self.turbulence_bool,
                price * scale,
                self.stocks * scale,
                self.stocks_cd,
                tech,
            )
        ).astype(np.float32)
        print(len(self.stockUniverse))
        return state

    def submitOrder(self, qty, stock, side, resp):
        if qty > 0:
            try:
                self.alpaca.submit_order(stock, qty, side, "market", "day")
                print(
                    "Market order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | completed."
                )
                resp.append(True)
            except:
                print(
                    "Order of | "
                    + str(qty)
                    + " "
                    + stock
                    + " "
                    + side
                    + " | did not go through."
                )
                resp.append(False)
        else:
            print(
                "Quantity is 0, order of | "
                + str(qty)
                + " "
                + stock
                + " "
                + side
                + " | not completed."
            )
            resp.append(True)

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh

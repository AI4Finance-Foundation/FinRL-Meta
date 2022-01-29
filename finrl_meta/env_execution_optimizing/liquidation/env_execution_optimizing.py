'''Source: https://github.com/AI4Finance-Foundation/Liquidation-Analysis-using-Multi-Agent-Reinforcement-Learning-ICML-2019/blob/master/syntheticChrissAlmgren.py'''
'''Paper: Multi-agent reinforcement learning for liquidation strategy analysis accepted by ICML 2019 AI in Finance: Applications and Infrastructure for Multi-Agent Learning. (https://arxiv.org/abs/1906.11046)'''

import collections
import random

import numpy as np

# ------------------------------------------------ Financial Parameters --------------------------------------------------- #

ANNUAL_VOLAT = 0.12  # Annual volatility in stock price
BID_ASK_SP = 1 / 8  # Bid-ask spread
DAILY_TRADE_VOL = 5e6  # Average Daily trading volume
TRAD_DAYS = 250  # Number of trading days in a year
DAILY_VOLAT = ANNUAL_VOLAT / np.sqrt(TRAD_DAYS)  # Daily volatility in stock price

# ----------------------------- Parameters for the Almgren and Chriss Optimal Execution Model ----------------------------- #

TOTAL_SHARES1 = 500000  # Total number of shares to sell
TOTAL_SHARES2 = 500000  # Total number of shares to sell
STARTING_PRICE = 50  # Starting price per share
LLAMBDA1 = 1e-6  # Trader's risk aversion
LLAMBDA2 = 1e-4
LIQUIDATION_TIME = 60  # How many days to sell all the shares.
NUM_N = 60  # Number of trades
EPSILON = BID_ASK_SP / 2  # Fixed Cost of Selling.
SINGLE_STEP_VARIANCE = (DAILY_VOLAT * STARTING_PRICE) ** 2  # Calculate single step variance
ETA = BID_ASK_SP / (0.01 * DAILY_TRADE_VOL)  # Price Impact for Each 1% of Daily Volume Traded
GAMMA = BID_ASK_SP / (0.1 * DAILY_TRADE_VOL)  # Permanent Impact Constant


# ----------------------------------------------------------------------------------------------------------------------- #


# Simulation Environment

class MarketEnvironment():

    def __init__(self, randomSeed=0,
                 lqd_time=LIQUIDATION_TIME,
                 num_tr=NUM_N,
                 lambd1=LLAMBDA1,
                 lambd2=LLAMBDA2):

        # Set the random seed
        random.seed(randomSeed)

        # Initialize the financial parameters so we can access them later
        self.anv = ANNUAL_VOLAT
        self.basp = BID_ASK_SP
        self.dtv = DAILY_TRADE_VOL
        self.dpv = DAILY_VOLAT

        # Initialize the Almgren-Chriss parameters so we can access them later
        self.total_shares1 = TOTAL_SHARES1
        self.total_shares2 = TOTAL_SHARES2
        self.startingPrice = STARTING_PRICE
        self.llambda1 = lambd1
        self.llambda2 = lambd2
        self.liquidation_time = lqd_time
        self.num_n = num_tr
        self.epsilon = EPSILON
        self.singleStepVariance = SINGLE_STEP_VARIANCE
        self.eta = ETA
        self.gamma = GAMMA

        # Calculate some Almgren-Chriss parameters
        self.tau = self.liquidation_time / self.num_n
        self.eta_hat = self.eta - (0.5 * self.gamma * self.tau)
        self.kappa_hat1 = np.sqrt((self.llambda1 * self.singleStepVariance) / self.eta_hat)
        self.kappa_hat2 = np.sqrt((self.llambda2 * self.singleStepVariance) / self.eta_hat)
        self.kappa1 = np.arccosh((((self.kappa_hat1 ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau
        self.kappa2 = np.arccosh((((self.kappa_hat2 ** 2) * (self.tau ** 2)) / 2) + 1) / self.tau

        # Set the variables for the initial state
        self.shares_remaining1 = self.total_shares1
        self.shares_remaining2 = self.total_shares2
        self.timeHorizon = self.num_n
        self.logReturns = collections.deque(np.zeros(6))

        # Set the initial impacted price to the starting price
        self.prevImpactedPrice = self.startingPrice

        # Set the initial transaction state to False
        self.transacting1 = False
        self.transacting2 = False

        # Set a variable to keep trak of the trade number
        self.k = 0

    def reset(self, seed=0, liquid_time=LIQUIDATION_TIME, num_trades=NUM_N, lamb1=LLAMBDA1, lamb2=LLAMBDA2):

        # Initialize the environment with the given parameters
        self.__init__(randomSeed=seed, lqd_time=liquid_time, num_tr=num_trades, lambd1=lamb1, lambd2=lamb2)

        # Set the initial state to [0,0,0,0,0,0,1,1]
        self.initial_state = np.array(list(self.logReturns) + [self.timeHorizon / self.num_n, \
                                                               self.shares_remaining1 / self.total_shares1, \
                                                               self.shares_remaining2 / self.total_shares2])
        return self.initial_state

    def start_transactions(self):

        # Set transactions on
        self.transacting1 = True
        self.transacting2 = True

        # Set the minimum number of stocks one can sell
        self.tolerance = 1

        # Set the initial capture to zero
        self.totalCapture1 = 0
        self.totalCapture2 = 0

        # Set the initial previous price to the starting price
        self.prevPrice = self.startingPrice

        # Set the initial square of the shares to sell to zero
        self.totalSSSQ1 = 0
        self.totalSSSQ2 = 0
        # Set the initial square of the remaing shares to sell to zero
        self.totalSRSQ1 = 0
        self.totalSRSQ2 = 0
        # Set the initial AC utility
        self.prevUtility1 = self.compute_AC_utility(self.total_shares1, self.kappa1, self.llambda1)
        self.prevUtility2 = self.compute_AC_utility(self.total_shares2, self.kappa2, self.llambda2)

    def step(self, action1, action2):

        # Create a class that will be used to keep track of information about the transaction
        class Info(object):
            pass

        info = Info()

        # Set the done flag to False. This indicates that we haven't sold all the shares yet.
        info.done1 = False
        info.done2 = False

        # During training, if the DDPG fails to sell all the stocks before the given 
        # number of trades or if the total number shares remaining is less than 1, then stop transacting,
        # set the done Flag to True, return the current implementation shortfall, and give a negative reward.
        # The negative reward is given in the else statement below.
        if self.transacting1 and (self.timeHorizon == 0 or (abs(self.shares_remaining1) < self.tolerance)):
            self.transacting1 = False
            info.done1 = True
            info.implementation_shortfall1 = self.total_shares1 * self.startingPrice - self.totalCapture1
            info.expected_shortfall1 = self.get_expected_shortfall(self.total_shares1, self.totalSSSQ1)
            info.expected_variance1 = self.singleStepVariance * self.tau * self.totalSRSQ1
            info.utility1 = info.expected_shortfall1 + self.llambda1 * info.expected_variance1

        if self.transacting2 and (self.timeHorizon == 0 or (abs(self.shares_remaining2) < self.tolerance)):
            self.transacting2 = False
            info.done2 = True
            info.implementation_shortfall2 = self.total_shares2 * self.startingPrice - self.totalCapture2
            info.expected_shortfall2 = self.get_expected_shortfall(self.total_shares2, self.totalSSSQ2)
            info.expected_variance2 = self.singleStepVariance * self.tau * self.totalSRSQ2
            info.utility2 = info.expected_shortfall2 + self.llambda2 * info.expected_variance2

        # We don't add noise before the first trade    
        if self.k == 0:
            info.price = self.prevImpactedPrice
        else:
            # Calculate the current stock price using arithmetic brownian motion
            info.price = self.prevImpactedPrice + np.sqrt(self.singleStepVariance * self.tau) * random.normalvariate(0,
                                                                                                                     1)

        # If we are transacting, the stock price is affected by the number of shares we sell. The price evolves 
        # according to the Almgren and Chriss price dynamics model. 
        if self.transacting1:

            # If action is an ndarray then extract the number from the array
            if isinstance(action1, np.ndarray):
                action1 = action1.item()

                # Convert the action to the number of shares to sell in the current step
            sharesToSellNow1 = self.shares_remaining1 * action1

            if self.timeHorizon < 2:
                sharesToSellNow1 = self.shares_remaining1
        else:
            sharesToSellNow1 = 0
        #             sharesToSellNow = min(self.shares_remaining * action, self.shares_remaining)
        if self.transacting2:

            # If action is an ndarray then extract the number from the array
            if isinstance(action2, np.ndarray):
                action2 = action2.item()

                # Convert the action to the number of shares to sell in the current step
            sharesToSellNow2 = self.shares_remaining2 * action2

            if self.timeHorizon < 2:
                sharesToSellNow2 = self.shares_remaining2
        else:
            sharesToSellNow2 = 0

        if self.transacting1 or self.transacting2:

            # Since we are not selling fractions of shares, round up the total number of shares to sell to the nearest integer. 
            info.share_to_sell_now1 = np.around(sharesToSellNow1)
            info.share_to_sell_now2 = np.around(sharesToSellNow2)
            # Calculate the permanent and temporary impact on the stock price according the AC price dynamics model
            info.currentPermanentImpact = self.permanentImpact(info.share_to_sell_now1 + info.share_to_sell_now2)
            info.currentTemporaryImpact = self.temporaryImpact(info.share_to_sell_now1 + info.share_to_sell_now2)

            # Apply the temporary impact on the current stock price    
            info.exec_price = info.price - info.currentTemporaryImpact

            # Calculate the current total capture
            self.totalCapture1 += info.share_to_sell_now1 * info.exec_price
            self.totalCapture2 += info.share_to_sell_now2 * info.exec_price

            # Calculate the log return for the current step and save it in the logReturn deque
            self.logReturns.append(np.log(info.price / self.prevPrice))
            self.logReturns.popleft()

            # Update the number of shares remaining
            self.shares_remaining1 -= info.share_to_sell_now1
            self.shares_remaining2 -= info.share_to_sell_now2

            # Calculate the runnig total of the squares of shares sold and shares remaining
            self.totalSSSQ1 += info.share_to_sell_now1 ** 2
            self.totalSRSQ1 += self.shares_remaining1 ** 2

            self.totalSSSQ2 += info.share_to_sell_now2 ** 2
            self.totalSRSQ2 += self.shares_remaining2 ** 2

            # Update the variables required for the next step
            self.timeHorizon -= 1
            self.prevPrice = info.price
            self.prevImpactedPrice = info.price - info.currentPermanentImpact

            # Calculate the reward
            currentUtility1 = self.compute_AC_utility(self.shares_remaining1, self.kappa1, self.llambda1)
            currentUtility2 = self.compute_AC_utility(self.shares_remaining2, self.kappa2, self.llambda2)
            if self.prevUtility1 == 0:
                reward1 = 0
            else:
                reward1 = (abs(self.prevUtility1) - abs(currentUtility1)) / abs(self.prevUtility1)
            if self.prevUtility2 == 0:
                reward2 = 0
            else:
                reward2 = (abs(self.prevUtility2) - abs(currentUtility2)) / abs(self.prevUtility2)

            if reward1 > reward2:
                reward2 -= reward1
                # reward2 += reward1
                # reward2 *= 0.5
                reward2 *= 0.5
            else:
                # reward1 += reward2
                # reward1 *= 0.5
                reward1 -= reward2
                reward1 *= 0.5
            # reward1 = max(reward1 - reward2, 0)
            # reward2 = max(reward2 - reward1, 0)

            self.prevUtility1 = currentUtility1
            self.prevUtility2 = currentUtility2

            # If all the shares have been sold calculate E, V, and U, and give a positive reward.
            if self.shares_remaining1 <= 0:
                # Calculate the implementation shortfall
                info.implementation_shortfall1 = self.total_shares1 * self.startingPrice - self.totalCapture1
                info.done1 = True

            if self.shares_remaining2 <= 0:
                # Calculate the implementation shortfall
                info.implementation_shortfall2 = self.total_shares2 * self.startingPrice - self.totalCapture2
                info.done2 = True

                # Set the done flag to True. This indicates that we have sold all the shares
        else:
            reward1 = 0.0
            reward2 = 0.0

        self.k += 1

        # Set the new state
        state = np.array(
            list(self.logReturns) + [self.timeHorizon / self.num_n, self.shares_remaining1 / self.total_shares1,
                                     self.shares_remaining2 / self.total_shares2])

        return (state, np.array([reward1]), np.array([reward2]), info.done1, info.done2, info)

    def permanentImpact(self, sharesToSell):
        # Calculate the permanent impact according to equations (6) and (1) of the AC paper
        return self.gamma * sharesToSell

    def temporaryImpact(self, sharesToSell):
        # Calculate the temporary impact according to equation (7) of the AC paper
        return (self.epsilon * np.sign(sharesToSell)) + ((self.eta / self.tau) * sharesToSell)

    def get_expected_shortfall(self, sharesToSell, totalSSSQ):
        # Calculate the expected shortfall according to equation (8) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)
        st = self.epsilon * sharesToSell
        tt = (self.eta_hat / self.tau) * totalSSSQ
        return ft + st + tt

    def get_AC_expected_shortfall(self, sharesToSell, kappa):
        # Calculate the expected shortfall for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * self.gamma * (sharesToSell ** 2)
        st = self.epsilon * sharesToSell
        tt = self.eta_hat * (sharesToSell ** 2)
        nft = np.tanh(0.5 * kappa * self.tau) * (self.tau * np.sinh(2 * kappa * self.liquidation_time) \
                                                 + 2 * self.liquidation_time * np.sinh(kappa * self.tau))
        dft = 2 * (self.tau ** 2) * (np.sinh(kappa * self.liquidation_time) ** 2)
        fot = nft / dft
        return ft + st + (tt * fot)

    def get_AC_variance(self, sharesToSell, kappa):
        # Calculate the variance for the optimal strategy according to equation (20) of the AC paper
        ft = 0.5 * (self.singleStepVariance) * (sharesToSell ** 2)
        nst = self.tau * np.sinh(kappa * self.liquidation_time) * np.cosh(kappa * (self.liquidation_time - self.tau)) \
              - self.liquidation_time * np.sinh(kappa * self.tau)
        dst = (np.sinh(kappa * self.liquidation_time) ** 2) * np.sinh(kappa * self.tau)
        st = nst / dst
        return ft * st

    def compute_AC_utility(self, sharesToSell, kappa, llambda):
        # Calculate the AC Utility according to pg. 13 of the AC paper
        if self.liquidation_time == 0:
            return 0
        E = self.get_AC_expected_shortfall(sharesToSell, kappa)
        V = self.get_AC_variance(sharesToSell, kappa)
        return E + llambda * V

    def get_trade_list(self, kappa):
        # Calculate the trade list for the optimal strategy according to equation (18) of the AC paper
        trade_list = np.zeros(self.num_n)
        ftn = 2 * np.sinh(0.5 * kappa * self.tau)
        ftd = np.sinh(kappa * self.liquidation_time)
        ft = (ftn / ftd) * self.total_shares1
        for i in range(1, self.num_n + 1):
            st = np.cosh(kappa * (self.liquidation_time - (i - 0.5) * self.tau))
            trade_list[i - 1] = st
        trade_list *= ft
        return trade_list

    def observation_space_dimension(self):
        # Return the dimension of the state
        return 8

    def action_space_dimension(self):
        # Return the dimension of the action
        return 1

    def stop_transactions(self):
        # Stop transacting
        self.transacting = False

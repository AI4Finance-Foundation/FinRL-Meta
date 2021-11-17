# TradingGym

# Trading Gym (>Python 3.7)
Trading Gym is an open-source project for the development of deep reinforcement learning algorithms in the context of forex trading.
## Highlight of the project 
1. customized gym.env for forex trading, with three actions (buy,sell, nothing). rewards is forex point based and realized by stop loss (SL) and profit taken (PT). The SL is fixed by passing parameter and PT is AI learning to maximize the rewards. 
2. multiply pairs with same time frame, time frame can setup from 1M, 5M, 30M, 1H, 4H. Both over night cash penalty, transaction fee, transaction overnight penalty (SWAP) can be configured.
3. data process will split the data into daily, weekly or monthly time serial based. and will training in parallel by using Isaac or Ray (coming soon)
4. file log, console print-out and live graph are available for render 
##  Major Feature
1. forex feature
    
    1.1.  using Point (5 decimal forex )  for reward and balance calculation
    ### Basis point
    The last decimal place to which a particular exchange rate is usually quoted is referred to as a point. 
       
2. data process: (./data/data_process.py)
    
    2.1. processing csv (time, open, high, low, close), the source I used is MetaTrader.
    
    2.2. adding a few required features, such as symbol, day, etc...
    
    2.3. adding TA features by using finta
    
    2.4. splitting data based on time serial into weekly or monthly
    
    2.5 combining different trading pair (e.g GBPUSD, EURUSD...) into on csv for vector process. (so far manually)
3. environment:
    
    3.1. action_space = spaces.Box(low=0, high=3, shape=(len(assets), ))
    
    3.2. _action = floor(action) {0:buy, 1:sell,2:nothing}
    
    3.3. action_price is the current candlestick close price.
    
    3.4. observation_space contains current balance, .. (draw down) * assets + (TA features)*assets
    
    3.5. using fix stop_loss (SL) parameter and fraction calculation for profit_taken (PT) as a part of action
    
    3.6. overnight cash penalty, transaction fee, transaction holding over night penalty
    
    3.7. rewards will be realized once SL or PT trigger at next step
    
    3.8. max holding
    
    3.9. done when balance <= x  or step == len(df)  [reach the weekend]
        if step == len(df) close all holding position at close price.

## The trading environment:

`Candle Trading` is a trading environment with input ohlc (open, high,low,close candlestick/bar) data, it is very useful to forex (currency) trading. We use profit-taking (machine learning) and fixed stop-loss.

## Create your own `data_process`

To create your own data, you can use `data_process` base class which can be found in the file 'data/data_process.py'. 

## Compatibility with OpenAI gym

The tgym (trading environment) is inherited from OpenAI Gym. We aim to entirely base it upon OpenAI Gym architecture and propose Trading Gym as an additional OpenAI environment.

## Examples
. ./neo_finrl/env_fx_trading/env.sh
ppo_test.ipynb
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

### Parameters (./neo_finrl/env_fx_trading/config/your.json)
| Scope | Parameter   | Description | Example|
| -------- | ----------- | ----------- | -------- |
| env | observation_list |A list numerical info feed back to env, you can add addition TAs from data process, it must contain:<br>"observation_list": ["Open","High","Low","Close""day"] |        "observation_list": [<br>"Open",<br>"High",<br>"Low",<br>"Close",<br>"minute",<br>"hour",<br>"day",<br>"macd",<br>"boll_ub","boll_lb",<br>"rsi_30",<br>"dx_30",<br>"close_30_sma",<br>"close_60_sma"<br>],|
| env |over_night_cash_penalty | penalty of holding cash over night in Point |5|
| env |balance | initial balance in Point | 1000 |
| env |asset_col| asset colunm name from source data | ASSET_ID or symbol|
| env |time_col| timestamp column name source data |Date|
| env |random_start| when training, reset start from seed |true |
| env |log_filename |log file location, make sure the folder is available  |  |
| env |title| plot grpah title | |
| env |description | free text | |
| symbol |symbol name |you must define your trading pair |GBPUSD |
| symbol |point |number of decimal of your trading pair. <br>for crypto/stock =1? |100000 |
| symbol |transaction_fee|transaction fee or cost in Point|5|
| symbol |over_night_penalty|holding position overnight cost (intrest charge)|1|
| symbol |stop_loss_max|stop loss, this is hard-coded for this pair trading lost exit,<br>for example, for a buy position, if current bar low price, low <= order_price - stop_loss_max/Point, position will force to close with lost.|200 (Point)|
| symbol |profit_taken_max|max profit taken, bot will set a PT betwen abs(stop_loss_max) and profit_taken_max |1000|
| symbol |max_current_holding|max current holding of this trading pair, if current holding > max_current_holding, it will stop order |5|
| symbol |limit_order| if use limit order|true|
| symbol |limit_order_expiration|limit order will withdrawn if exceed n bars. it will be expiration|5 (bars)|
| symbol |max_spread|=ask-bid, if exceed max_spred, not order |not implemented yet|


## The trading environment:

`Candle Trading` is a trading environment with input ohlc (open, high,low,close candlestick/bar) data, it is very useful to forex (currency) trading. We use profit-taking (machine learning) and fixed stop-loss.

## Create your own `data_process`

To create your own data, you can use `data_process` base class which can be found in the file 'data/data_process.py'. 

## Compatibility with OpenAI gym

The tgym (trading environment) is inherited from OpenAI Gym. We aim to entirely base it upon OpenAI Gym architecture and propose Trading Gym as an additional OpenAI environment.

## Examples
```shell
. ./neo_finrl/env_fx_trading/env.sh
ppo_test.ipynb
```
## Download Datasets from Dukascopy
```shell
duka EURUSD -s 2010-01-01 -e 2012-12-31 -c M5 
duka GBPUSD -s 2010-01-01 -e 2012-12-31 -c M5
duka USDJPY -s 2010-01-01 -e 2012-12-31 -c M5
duka EURJPY -s 2010-01-01 -e 2012-12-31 -c M5
duka AUDUSD -s 2010-01-01 -e 2012-12-31 -c M5
duka NZDUSD -s 2010-01-01 -e 2012-12-31 -c M5
duka USDCAD -s 2010-01-01 -e 2012-12-31 -c M5
duka USDCHF -s 2010-01-01 -e 2012-12-31 -c M5
duka XAGUSD -s 2010-01-01 -e 2012-12-31 -c M5
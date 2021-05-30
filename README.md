# NeoFinRL
A suite of universal near real-market environments for DRL in quant finance.


**Why NeoFinRL (FinRL-Gym): Motivation**

● To close the sim-reality gap: existing academic AI+Finance papers use backtesting on historical data, the reported performance (annualized return, shape ratio, max dropdown, etc.) may be quite different when we apply the algorithms to a paper/live trading, or real markets.

● To reduce the data pre-processing burden, so that quants can focus on developing their strategies.

● To facilitate fair comparisons or benchmark performances, providing standardized envs will allow researchers to evaluate different strategies in some way. Also, it would help researchers to better understand the “black-box” algorithm.

**Design Principles**

● Plug-and-play (PnP): Modularity○Handle different markets (say T0 vs. T+1)

● Avoid hard-coded parameters

● Closing the sim-real gap by “simulation-validation-trading”: simulation for training andconnecting real-time APIs for trading;  here a “virtual env” may be a good solution.

● Efficient sampling: accelerate sampling is key! Here, we can learn ideas from the ElegantRL project. Note that multi-processing is a key to reducing training time. (scheduling between CPU + GPU)

● Transparency: a virtual env which is invisible to the upper layer

● Completeness and universal:
   
  Different markets; 
  
  Various data sources (API, Excel, etc)○User-friendly variables (complete and allow user-define): may use heritage of class

● Flexibility and extensibility: Inheritance might be helpful here

**What is Gym?**
● Gym by OpenAI

Gym is a toolkit for developing and comparing DRL algorithms.
Gym’s main purpose is to provide a large collection of environments that expose a unified interface and allow fair comparisons.

Two main constraints of Gym:
1. The need for better benchmarks
2. Lack of standardization of environments used in publications

Resources:
https://github.com/openai/gym/blob/master/docs/creating-environments.md


**CCXT**

run demo_btc_ccxt.py

**WRDS**

See demo_stock_wrds.ipynb

**Alpaca**

run demo_stock_alpaca.py

**JoinQuant**

run demo_stock_jq.py

**QuantConnect**

See demo_stock_qc.ipynb
(Run it on QuantConnect web)

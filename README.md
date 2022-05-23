# FinRL-Meta: A Universe of Market Environments and Benchmarks for Data-Driven Financial Reinforcement Learning


[![Downloads](https://pepy.tech/badge/finrl_meta)](https://pepy.tech/project/finrl_meta)
[![Downloads](https://pepy.tech/badge/finrl_meta/week)](https://pepy.tech/project/finrl_meta)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl_meta.svg)](https://pypi.org/project/finrl_meta/)

FinRL-Meta is a universe of market environments for data-driven financial reinforcement learning. Users can use FinRL-Meta as the metaverse of their financial  environments.

1. FinRL-Meta separates financial data processing from the design pipeline of DRL-based strategy and provides open-source data engineering tools for financial big data.
2. FinRL-Meta provides hundreds of market environments for various trading tasks.
3. FinRL-Meta enables multiprocessing simulation and training by exploiting thousands of GPU cores.


Also called **Neo_FinRL**: **N**ear real-market **E**nvironments f**o**r data-driven **Fin**ancial **R**einforcement **L**earning.

## Outline
- [News and Tutorials](#news-and-tutorials)
- [Our Goals](#our-goals)
- [Design Principles](#design-principles)
- [Overview](#overview)
- [Plug-and-Play](#plug-and-play)
- [Training-Testing-Trading](#training-testing-trading-pipeline)
- [Our Vision](#our-vision)

## News and Tutorials

+ [DataDrivenInvestor] [FinRL-Meta: A Universe of Near Real-Market En­vironments for Data­-Driven Financial Reinforcement Learning](https://medium.datadriveninvestor.com/finrl-meta-a-universe-of-near-real-market-en-vironments-for-data-driven-financial-reinforcement-e1894e1ebfbd)
+ [央广网] [2021 IDEA大会于福田圆满落幕：群英荟萃论道AI 多项目发布亮点纷呈](http://tech.cnr.cn/techph/20211123/t20211123_525669092.shtml)
+ [央广网] [2021 IDEA大会开启AI思想盛宴 沈向洋理事长发布六大前沿产品](https://baijiahao.baidu.com/s?id=1717101783873523790&wfr=spider&for=pc)
+ [IDEA新闻] [2021 IDEA大会发布产品FinRL-Meta——基于数据驱动的强化学习金融风险模拟系统](https://idea.edu.cn/news/20211213143128.html)
+ [知乎] [FinRL-Meta基于数据驱动的强化学习金融元宇宙](https://zhuanlan.zhihu.com/p/437804814)

## Our Goals
+ To reduce the simulation-reality gap: existing works use backtesting on historical data, while the real performance may be quite different when applying the algorithms to paper/live trading.
+ To reduce the data pre-processing burden, so that quants can focus on developing and optimizing strategies.
+ To provide benchmark performance and facilitate fair comparisons, providing a standardized environment will allow researchers to evaluate different strategies in the same way. Also, it would help researchers to better understand the “black-box” nature (deep neural network-based) of DRL algorithms.

## Design Principles
+ Plug-and-Play (PnP): Modularity; Handle different markets (say T0 vs. T+1)
+ Completeness and universal:
  Multiple markets; Various data sources (APIs, Excel, etc); User-friendly variables.
+ Avoid hard-coded parameters
+ Closing the sim-real gap using the “training-testing-trading” pipeline: simulation for training and connecting real-time APIs for testing/trading.
+ Efficient data sampling: accelerate the data sampling process is the key to DRL training!  From the ElegantRL project. we know that multi-processing is powerful to reduce the training time (scheduling between CPU + GPU).
+ Transparency: a virtual env that is invisible to the upper layer
+ Flexibility and extensibility: Inheritance might be helpful here

## Overview 
![Overview image of FinRL-Meta](https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/figs/neofinrl_overview.png)
We utilize a layered structure in FinRL-metaverse, as shown in the figure above. FinRL-metaverse consists of three layers: data layer, environment layer, and agent layer. Each layer executes its functions and is independent. Meanwhile, layers interact through end-to-end interfaces to implement the complete workflow of algorithm trading.

## DataOps
DataOps is a series of principles and practices to improve the quality and reduce the cycle time of data science. It inherits the ideas of Agile development, DevOps, and lean manufacturing and applies them to the data science and machine learning field. FinRL-Meta follows the DataOps paradigm.


<div align="center">
<img align="center" src=figs/finrl_meta_dataops.png width="800">
</div>


Supported Data Sources: 
|Data Source |Type |Range and Frequency |Request Limits|Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|[Alpaca](https://alpaca.markets/docs/introduction/)| US Stocks, ETFs| 2015-now, 1min| Account-specific| OHLCV| Prices&Indicators|
|[Baostock](http://baostock.com/baostock/index.php/Python_API%E6%96%87%E6%A1%A3)| CN Securities| 1990-12-19-now, 5min| Account-specific| OHLCV| Prices&Indicators|
|[Binance](https://binance-docs.github.io/apidocs/spot/en/#public-api-definitions)| Cryptocurrency| API-specific, 1s, 1min| API-specific| Tick-level daily aggegrated trades, OHLCV| Prices&Indicators|
|[CCXT](https://docs.ccxt.com/en/latest/manual.html)| Cryptocurrency| API-specific, 1min| API-specific| OHLCV| Prices&Indicators|
|[IEXCloud](https://iexcloud.io/docs/api/)| NMS US securities|1970-now, 1 day|100 per second per IP|OHLCV| Prices&Indicators|
|[JoinQuant](https://www.joinquant.com/)| CN Securities| 2005-now, 1min| 3 requests each time| OHLCV| Prices&Indicators|
|[QuantConnect](https://www.quantconnect.com/docs/home/home)| US Securities| 1998-now, 1s| NA| OHLCV| Prices&Indicators|
|[RiceQuant](https://www.ricequant.com/doc/rqdata/python/)| CN Securities| 2005-now, 1ms| Account-specific| OHLCV| Prices&Indicators|
|[Tushare](https://tushare.pro/document/1?doc_id=131)| CN Securities, A share| -now, 1 min| Account-specific| OHLCV| Prices&Indicators|
|[WRDS](https://wrds-www.wharton.upenn.edu/pages/about/data-vendors/nyse-trade-and-quote-taq/)| US Securities| 2003-now, 1ms| 5 requests each time| Intraday Trades|Prices&Indicators|
|[YahooFinance](https://pypi.org/project/yfinance/)| US Securities| Frequency-specific, 1min| 2,000/hour| OHLCV | Prices&Indicators|



OHLCV: open, high, low, and close prices; volume 

adjusted_close: adjusted close price

Technical indicators users can add: 'macd', 'boll_ub', 'boll_lb', 'rsi_30', 'dx_30', 'close_30_sma', 'close_60_sma'. Users also can add their features. 


## Plug-and-Play (PnP)
In the development pipeline, we separate market environments from the data layer and the agent layer. Any DRL agent can be directly plugged into our environments, then trained and tested. Different agents/algorithms can be compared by running on the same benchmark environment for fair evaluations. 

A demonstration notebook for plug-and-play with ElegantRL, Stable Baselines3 and RLlib: [Plug and Play with DRL Agents](https://colab.research.google.com/github/AI4Finance-Foundation/FinRL-Meta/blob/main/Demo_Plug_and_Play_with_DRL_Libraries.ipynb)

## "Training-Testing-Trading" Pipeline

A DRL agent learns by interacting with the training environment, is validated in the validation environment for parameter tuning. Then, the agent is tested in historical datasets (backtesting).  Finally, the agent will be deployed in paper trading or live trading markets. 

This pipeline solves the **information leakage problem** because the trading data are never leaked when training/tuning the agents. 

Such a unified pipeline allows fair comparisons among different algorithms and strategies. 


<div align="center">
<img align="center" src=figs/timeline.png width="800">
</div>
	

## Our Vision

For future work, we plan to build a multi-agent-based market simulator that consists of over ten thousands of agents, namely, a FinRL-Metaverse. First, FinRL-Metaverse aims to build a universe of market environments, like the XLand environment ([source](https://deepmind.com/research/publications/2021/open-ended-learning-leads-to-generally-capable-agents)) and planet-scale climate forecast ([source](https://www.nature.com/articles/s41586-021-03854-z)) by DeepMind. To improve the performance for large-scale markets, we will employ GPU-based massive parallel simulation just as Isaac Gym ([source](https://arxiv.org/abs/2108.10470)). Moreover, it will be interesting to explore the deep evolutionary RL framework ([source](https://doaj.org/article/4dd31838732842439cc1301e52613d1c)) to simulate the markets. Our final goal is to provide insights into complex market phenomena and offer guidance for financial regulations through FinRL-Metaverse.

<div align="center">
<img align="center" src=figs/finrl_metaverse.png width="800">
</div>
	

## Citing FinRL-Meta
```
@article{finrl_meta_2021,
    author = {Liu, Xiao-Yang and Rui, Jingyang and Gao, Jiechao and Yang, Liuqing and Yang, Hongyang and Wang, Zhaoran and Wang, Christina Dan and Guo Jian},
    title   = {{FinRL-Meta}: Data-Driven Deep ReinforcementLearning in Quantitative Finance},
    journal = {Data-Centric AI Workshop, NeurIPS},
    year    = {2021}
}

```

## Collaborators

<div align="center">
<img align="center" src=figs/Columbia_logo.jpg width="120"> &nbsp;&nbsp;
<img align="center" src=figs/IDEA_Logo.png width="200"> &nbsp;&nbsp;
<img align="center" src=figs/Northwestern_University.png width="120"> &nbsp;&nbsp;
<img align="center" src=figs/NYU_Shanghai_Logo.png width="200">	&nbsp;&nbsp;
</div>

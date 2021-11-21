# FinRL-Meta: A Universe of Market Environments.


[![Downloads](https://pepy.tech/badge/finrl_meta)](https://pepy.tech/project/finrl_meta)
[![Downloads](https://pepy.tech/badge/finrl_meta/week)](https://pepy.tech/project/finrl_meta)
[![Python 3.6](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/downloads/release/python-360/)
[![PyPI](https://img.shields.io/pypi/v/finrl_meta.svg)](https://pypi.org/project/finrl_meta/)

FinRL-Meta is a universe of market environments for data-driven financial reinforcement learning.
1. FinRL-Meta separates financial data processing from the design pipeline of DRL-based strategy and provides open-source data engineering tools for financial big data.
2. FinRL-Meta provides hundreds of market environments for various trading tasks.
3. FinRL-Meta enables multiprocessing simulation and training by exploiting thousands of GPU cores.


Also called **Neo_FinRL**: **N**ear real-market **E**nvironments f**o**r data-driven **Fin**ancial **R**einforcement **L**earning.

## Outline
- [Our Goals](#our-goals)
- [Design Principles](#design-principles)
- [Overview](#overview)
- [Plug-and-Play](#plug-and-play)
- [Training-Testing-Trading](#training-testing-trading-pipeline)
- [Our Vision](#our-vision)

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
![Overview image of NeoFinRL](https://github.com/AI4Finance-Foundation/NeoFinRL/blob/main/figs/neofinrl_overview.png)
We utilize a layered structure in FinRL-metaverse, as shown in the figure above. FinRL-metaverse consists of three layers: data layer, environment layer, and agent layer. Each layer executes its functions and is independent. Meanwhile, layers interact through end-to-end interfaces to implement the complete workflow of algorithm trading.

## DataOps
DataOps is a series of principles and practices to improve the quality and reduce the cycle time of data science. It inherits the ideas of Agile development, DevOps, and lean manufacturing and applies them to the data science and machine learning field. FinRL-Meta follows the DataOps paradigm.


<div align="center">
<img align="center" src=figs/finrl_meta_dataops.png width="800">
</div>


Supported Data Sources: 
|Data Source |Type |Range and Frequency |Request Limits|Raw Data|Preprocessed Data|
|  ----  |  ----  |  ----  |  ----  |  ----  |  ----  |
|Yahoo! Finance| US Securities| Frequency-specific, 1min| 2,000/hour| OHLCV | Prices&Indicators|
|CCXT| Cryptocurrency| API-specific, 1min| API-specific| OHLCV| Prices&Indicators|
|WRDS.TAQ| US Securities| 2003-now, 1ms| 5 requests each time| Intraday Trades|Prices&Indicators|
|Alpaca| US Stocks, ETFs| 2015-now, 1min| Account-specific| OHLCV| Prices&Indicators|
|RiceQuant| CN Securities| 2005-now, 1ms| Account-specific| OHLCV| Prices&Indicators|
|JoinQuant| CN Securities| 2005-now, 1min| 3 requests each time| OHLCV| Prices&Indicators|
|QuantConnect| US Securities| 1998-now, 1s| NA| OHLCV| Prices&Indicators|


## Plug-and-Play
In the development pipeline, we separate market environments from the data layer and the agent layer. Any DRL agent can be directly plugged into our environments, then trained and tested. Different agents/algorithms can be compared by running on the same benchmark environment for fair evaluations. 

A demonstration notebook for plug-and-play with ElegantRL, Stable Baselines3 and RLlib: [Play and Play with DRL Agents](https://colab.research.google.com/github/AI4Finance-Foundation/FinRL-Meta/blob/main/Demo_Plug_and_Play_with_DRL_Libraries.ipynb)

## "Training-Testing-Trading" Pipeline

A DRL agent learns by interacting with the training environment, is validated in the validation environment for parameter tuning. Then, the agent is tested in historical datasets (backtesting).  Finally, the agent will be deployed in paper trading or live trading markets. 

This pipeline solves the **information leakage problem** because the trading data are never leaked when training/tuning the agents. 

Such a unified pipeline allows fair comparisons among different algorithms and strategies. 


<div align="center">
<img align="center" src=figs/timeline.png width="800">
</div>
	

## Our Vision

For future work, we plan to build a multi-agent-based market simulator that consists of over ten thousands of agents, namely, a FinRL-Metaverse. First, FinRL-Metaverse aims to build a universe of market environments, like the XLand environment ([source](https://deepmind.com/research/publications/2021/open-ended-learning-leads-to-generally-capable-agents)) and planet-scale climate forecast ([source](https://www.nature.com/articles/s41586-021-03854-z)) by DeepMind. To improve the performance for large-scale markets, we will employ GPU-based massive parallel simulation as Isaac Gym ([source](https://arxiv.org/abs/2108.10470)). Moreover, it will be interesting to explore the deep evolutionary RL framework ([source](https://doaj.org/article/4dd31838732842439cc1301e52613d1c)) to simulate the markets. Our final goal is to provide insights into complex market phenomena and offer guidance for financial regulations through FinRL-Metaverse.

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
<img align="center" src=figs/Columbia_logo.jpg width="150"> &nbsp;&nbsp;
<img align="center" src=figs/IDEA_Logo.png width="200"> &nbsp;&nbsp;
<img align="center" src=figs/Northwestern_University.png width="150"> &nbsp;&nbsp;
<img align="center" src=figs/NYU_Shanghai_Logo.png width="200">	&nbsp;&nbsp;
</div>

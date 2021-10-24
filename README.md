# NeoFinRL: Tens of Market Environments for Financial Reinforcement Learning

**N**ear real-market **E**nvironments f**o**r data-driven **Fin**ancial **R**einforcement **L**earning (**NeoFinRL**) is an open-source framework that provides tens of standard market environments for various trading tasks and standard APIs to connect with different financial data sources, trading platforms and DRL algorithms.

## Outline
- [Our Goals](#our-goals)
- [Design Principles](#design-principles)
- [Overview](#overview)
- [Plug-and-Play](#plug-and-play)
- [Training-Testing-Trading](#training-testing-trading)

## Our Goals
+ To reduce the simulation-reality gap: existing works use backtesting on historical data, while the real performance may be quite different when applying the algorithms to paper/live trading.
+ To reduce the data pre-processing burden, so that quants can focus on developing and optimizing strategies.
+ To provide benchmark performance and facilitate fair comparisons, providing a standardized environment will allow researchers to evaluate different strategies in the same way. Also, it would help researchers to better understand the “black-box” nature (deep neural network-based) of DRL algorithms.

## Design Principles
+ Plug-and-Play (PnP): Modularity; Handle different markets (say T0 vs. T+1)
+ Avoid hard-coded parameters
+ Closing the sim-real gap by the “training-testing-trading” pipeline: simulation for training and connecting real-time APIs for testing/trading;  here a “virtual env” may be the solution.
+ Efficient sampling: accelerate the data sampling process is the key!  From the ElegantRL project. we know that multi-processing is a key to reducing training time (scheduling between CPU + GPU).
+ Transparency: a virtual env that is invisible to the upper layer
+ Completeness and universal:
  Multiple markets; Various data sources (APIs, Excel, etc); User-friendly variables (complete and allow user-define): may use the heritage of class
+ Flexibility and extensibility: Inheritance might be helpful here

## Overview 
![Overview image of NeoFinRL](https://github.com/AI4Finance-Foundation/NeoFinRL/blob/main/figs/neofinrl_overview.png)
We adopt a layered structure for DRL in finance in NeoFinRL, as shown in the figure above. NeoFinRL consists of three layers: data layer, environment layer, and agent layer. Each layer executes its functions and is relatively independent. Meanwhile, layers interact through end-to-end interfaces to implement the complete workflow of algorithm trading.

## Plug-and-Play
In the development pipeline, we separate market environments from the data layer and the agent layer. Any DRL agent can be directly plugged into our environments, then trained and tested. Different agents/algorithms can be compared by running on the same benchmark environment to achieve fair evaluations. 

A demonstration notebook for plug-and-play with ElegantRL, Stable Baselines3 and RLlib: [NeoFinRL: Plug-and-Play with DRL libraries](https://colab.research.google.com/drive/1zd10aPKT7fE9lr3G2hozT-SeRRYRU2rg?usp=sharing)

## Training-Testing-Trading
We employ a training-testing-trading pipeline. The DRL agent first learns from the training environment and is then validated in the validation environment for further adjustment. Then the validated agent is tested in historical datasets. Finally, the tested agent will be deployed in paper trading or live trading markets. First, this pipeline solves the information leakage problem because the trading data are never leaked when adjusting agents. Second, a unified pipeline allows fair comparisons among different algorithms and strategies. 



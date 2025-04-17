=========================================
Task 1 FinRL-DeepSeek for Stock Trading
=========================================

FinRL-DeepSeek for Stock Trading is a task that challenges you to build trading agents by combining FinRL and LLMs. Your mission: train agents that can trade stocks using both market data and financial news while effectively managing risk. This task builds on the FinRL-DeepSeek [1]_ — an integration of LLMs and  RL algorithms designed for real-world stock trading scenarios.

Task Overview
----------------
In this task, participants are invited to develop stock trading agents that integrate LLM-generated signals in FinRL using both market and news data. Building on the `FinRL-DeepSeek <https://github.com/benstaf/FinRL_DeepSeek>`_, participants can explore a variety of directions, including but not limited to:

- Designing new LLM prompting strategies to extract signals from financial news
- Exploring novel ways to inject LLM-generated signals into the RL pipeline (e.g., into policy networks, reward shaping, or environment dynamics)
- Applying alternative RL algorithms, such as Generalized Reward Policy Optimization (GRPO).
- Investigating computationally intensive approaches, including adapting instruction-tuned variants of the DeepSeek R1 training methodology to this stock trading task

Participants are encouraged to propose creative improvements and extensions that further advance the integration of LLM-generated signals in FinRL for financial decision-making.

Why This Matters
----------------
Modern RL trading agents often rely on structured market data and overlook unstructured financial documents, such as news. They also tend to optimize returns without controlling risk — leading to poor performance in volatile markets.

FinRL-DeepSeek introduces a hybrid solution with:

- LLM-generated signals: extract sentiment score and risk level from financial news.
- CVaR-PPO Integration: Risk-constrained learning to cap losses while pursuing gains.
- Action Modulation: Agent actions are scaled by LLM-generated sentiment scores.
- Reward Adjustment: High-risk actions are penalized using LLM-generated risk scores.

This is the first time CVaR-PPO has been adapted to stock trading with LLMs. We invite you to explore, improve, and extend it.


Data
--------

FNSPID Dataset
~~~~~~~~~~~~~~~
The `Financial News and Stock Price Integration Dataset (FNSPID) <https://huggingface.co/datasets/Zihan1004/FNSPID>`_ is a comprehensive financial news dataset. It contains historical stock prices and over 15 million time-aligned financial news articles related to Nasdaq-listed companies, spanning from 1999 to 2023. 

Here is a preview of the dataset:

.. list-table::
   :header-rows: 1
   :widths: 15 30 10 50 20 10 10 10 10 10 10

   * - Date
     - Article_title
     - Stock_symbol
     - Url
     - Publisher
     - Author
     - Article
     - Lsa_summary
     - Luhn_summary
     - Textrank_summary
     - Lexrank_summary
   * - 2020-06-05 06:30:54 UTC
     - Stocks That Hit 52-Week Highs On Friday
     - A
     - https://www.benzinga.com/news/20/06/16190091/stocks-that-hit-52-week-highs-on-friday
     - Benzinga Insights
     - null
     - null
     - null
     - null
     - null
     - null
   * - 2020-06-03 06:45:20 UTC
     - Stocks That Hit 52-Week Highs On Wednesday
     - A
     - https://www.benzinga.com/news/20/06/16170189/stocks-that-hit-52-week-highs-on-wednesday
     - Benzinga Insights
     - null
     - null
     - null
     - null
     - null
     - null
   * - 2020-05-26 00:30:07 UTC
     - 71 Biggest Movers From Friday
     - A
     - https://www.benzinga.com/news/20/05/16103463/71-biggest-movers-from-friday
     - Lisa Levin
     - null
     - null
     - null
     - null
     - null
     - null

To reduce LLM API costs, we randomly select a news article per stock per day, reducing the dataset to 2 million records. The final filtered dataset contains news for 89 Nasdaq stocks from 2013 to 2023.

LLM-Generated Signals
~~~~~~~~~~~~~~~~~~~~~~~
From the financial news, we leverage DeepSeek-V3 to generate sentiment scores and risk levels.

Sentiment scores
^^^^^^^^^^^^^^^^^
DeepSeek-V3 assigns a sentiment score u of 1 to 5 according to the news, with 1 for negative and 5 for positive.

.. raw:: html

   <div style="background-color:#f0f0f0; padding:12px; border-radius:6px;">
     <strong>Prompt:</strong><br>
       You are a financial expert with sentiment analysis and stock recommendation experience. Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive
   </div>

Risk levels
^^^^^^^^^^^^^^^^
DeepSeek-V3 assigns a risk level q of 1 to 5 from the news, with 1 for low risk and 5 high risk.

.. raw:: html

   <div style="background-color:#f0f0f0; padding:12px; border-radius:6px;">
     <strong>Prompt:</strong><br>
       You are a financial expert specializing in risk assessment. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk.
   </div>


OHLCV dataset
~~~~~~~~~~~~~~~~~~~~~~~
In addition, we download the OHLCV data from Yahoo Finance for the same stocks and period. The dataset includes daily open, high, low, close prices, and volume data for each stock. We also include the 10 market indicators through feature engineering, as shown in :ref:`technical-indicators`

We combine the OHLCV dataset and LLM-generated signals together:

.. list-table::
   :header-rows: 1
   :widths: 10 10 10 10 10 10 12 10 10 10 10 10 10 10 10 8 8 8 8

   * - date
     - tic
     - close
     - high
     - low
     - open
     - volume
     - macd
     - boll_ub
     - boll_lb
     - rsi_30
     - cci_30
     - dx_30
     - close_30_sma
     - close_60_sma
     - vix
     - turbulence
     - llm_sentiment
     - llm_risk
   * - 2019-01-02
     - AAPL
     - 37.71
     - 39.71
     - 38.56
     - 38.72
     - 148158800.0
     - -1.98
     - 43.58
     - 34.71
     - 37.87
     - -91.62
     - 42.25
     - 40.37
     - 45.53
     - 23.22
     - 91.01
     - 
     - 
   * - 2019-01-02
     - ADBE
     - 224.57
     - 226.17
     - 219.00
     - 219.91
     - 2784100.0
     - -5.28
     - 259.82
     - 203.15
     - 45.78
     - -54.09
     - 18.28
     - 232.43
     - 239.19
     - 23.22
     - 91.01
     - 5.0
     - 2.0
   * - 2019-01-02
     - ADI
     - 76.83
     - 86.43
     - 83.96
     - 84.03
     - 2523900.0
     - -0.73
     - 83.01
     - 71.99
     - 48.22
     - -65.52
     - 12.80
     - 78.22
     - 76.79
     - 23.22
     - 91.01
     - 3.0
     - 3.0

We include a processed `subset of this dataset <https://huggingface.co/datasets/benstaf/nasdaq_2013_2023>`_ as an example for participants to explore and build their solutions.

Participants can also incorporate additional public data sources, such as Twitter, or develop their own scraping and API-based agents. Some teams may focus on improving the dataset and news processing pipeline, while others may concentrate on designing better trading agents.


Starter Kit: Training & Environments
-------------------------------------
This `starter kit <https://github.com/Open-Finance-Lab/FinRL_Contest_2025/tree/main/Task_1_FinRL_DeepSeek_Stock>`_ includes training scripts and environment files for PPO-based and CPPO-based stock trading agents, including LLM-enhanced versions. Follow the instructions below to get started.

Training Commands
~~~~~~~~~~~~~~~~~~~~~~~
You can train different types of trading agents using the following scripts. Each script corresponds to a specific reinforcement learning setup, ranging from standard PPO to LLM-enhanced and risk-sensitive configurations.

To train the various models, follow the instructions below:

.. list-table:: Model Types and Training Scripts
   :header-rows: 1
   :widths: 20 40 40

   * - Model Type
     - Description
     - Training Script / Command
   * - PPO
     - Standard Proximal Policy Optimization (no LLM)
     - ``nohup mpirun --allow-run-as-root -np 8 python train_ppo.py > output_ppo.log 2>&1 &``
   * - CPPO
     - Conditional Value-at-Risk PPO (risk-sensitive)
     - ``python train_cppo.py``
   * - PPO-DeepSeek
     - PPO with LLM-enhanced trading signals
     - ``python train_ppo_llm.py``
   * - CPPO-DeepSeek
     - CPPO with LLM-based recommendations and risk
     - ``python train_cppo_llm_risk.py``

Environment Files
~~~~~~~~~~~~~~~~~~
Each training script corresponds to a specific environment implementation:

.. list-table:: Environment Files and Usage
   :header-rows: 1
   :widths: 40 60

   * - Environment File
     - Used In
   * - ``env_stocktrading.py``
     - PPO, CPPO (standard FinRL environment)
   * - ``env_stocktrading_llm.py`` or ``env_stocktrading_llm_01.py``
     - PPO-DeepSeek (LLM-modulated trading)
   * - ``env_stocktrading_llm_risk.py`` or ``env_stocktrading_llm_risk_01.py``
     - CPPO-DeepSeek (LLM + risk scoring)

Monitoring Training
--------------------
Each training script outputs logs (e.g., output_ppo.log). Key metrics to monitor include:

- AverageEpRet: Average episode return
- KL: KL divergence for policy update stability
- ClipFrac: Fraction of clipped policy updates

Use these metrics to track learning progress and tune hyperparameters accordingly.

Evaluation (Example)
----------------------
We provide an example evaluation workflow for the trading period 2019–2023, implemented in the `FinRL_DeepSeek_backtest.ipynb (Colab notebook) <https://colab.research.google.com/github/benstaf/FinRL_DeepSeek/blob/main/FinRL_DeepSeek_backtesting.ipynb#scrollTo=7r6aAYR1jOdN>`_. This example uses the following metrics to assess both return and risk-adjusted performance:

- Information Ratio – Measures risk-adjusted return relative to a benchmark
- Conditional Value at Risk (CVaR) – Quantifies expected losses in extreme downside scenarios
- Rachev Ratio – Captures the balance between upside potential and downside risk
- Outperformance Frequency – Measures how often the strategy outperforms the benchmark in rolling windows

Note: This evaluation setup is provided as a reference. Participants are encouraged to explore additional metrics or adapt the backtesting pipeline to fit their model's characteristics.


**References**

.. [1] Mostapha Benhenda. 2025. FinRL-DeepSeek: LLM-Infused Risk-Sensitive Reinforcement Learning for Trading Agents. arXiv preprint arXiv:2502.07393 (2025).




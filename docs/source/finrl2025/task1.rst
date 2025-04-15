=========================================
Task 1 FinRL-DeepSeek for Stock Trading
=========================================

Welcome to Task 1: FinRL-DeepSeek for Stock Trading, a competition that challenges you to build next-generation trading agents by combining FinRL and LLMs. Your mission: train agents that can trade stocks using both market data and financial news while effectively managing risk. This task builds on the FinRL-DeepSeek [1]_ — a novel integration of LLMs and  RL algorithms designed for real-world stock trading scenarios.

Task Overview
----------------
In this task, participants are invited to develop stock trading agents that integrate LLM-generated signals in FinRL using both market and news data. Building on the `FinRL-DeepSeek <https://github.com/benstaf/FinRL_DeepSeek>`_, participants can explore a variety of directions, including but not limited to:

- Designing new LLM prompting strategies to extract trading or risk signals from financial news
- Exploring novel ways to inject LLM-generated signals into the RL pipeline (e.g., into policy networks, reward shaping, or environment dynamics)
- Applying alternative RL algorithms, such as Generalized Reward Policy Optimization (GRPO).
- Investigating computationally intensive approaches, including adapting instruction-tuned variants of the DeepSeek R1 training methodology to this stock trading task

Participants are encouraged to propose creative improvements and extensions that further advance the hybrid LLM-RL paradigm in financial decision-making.

Why This Matters
----------------
Modern RL trading agents often rely on structured price data and overlook unstructured signals like news. They also tend to optimize returns without controlling risk — leading to poor performance in volatile markets.

FinRL-DeepSeek introduces a hybrid solution with:

- Dual LLM Guidance: One LLM for trading recommendations, another for risk scoring
- CVaR-PPO Integration: Risk-constrained learning to cap losses while pursuing gains
- Action Modulation: Agent actions are scaled by LLM recommendations
- Reward Adjustment: High-risk actions are penalized using LLM-derived risk scores

This is the first time CVaR-PPO has been adapted to stock trading with LLMs. We invite you to explore, improve, and extend it.


Datasets
--------

The `Financial News and Stock Price Integration Dataset (FNSPID) <https://huggingface.co/datasets/Zihan1004/FNSPID>`_ contains historical stock prices and over 15 million time-aligned financial news articles related to Nasdaq-listed companies, spanning from 1999 to 2023. We include a processed `subset of this dataset <https://huggingface.co/datasets/benstaf/nasdaq_2013_2023>`_ as an example for participants to explore and build their solutions.

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




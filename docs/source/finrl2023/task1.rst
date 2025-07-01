=========================================
Task 1: Data Centric Stock Trading
=========================================


This task emphasizes data engineering in building FinRL agents. The contestants are encouraged to devise novel data-centric strategies to clean, transform, and aggregate stock data to improve trading performance, with the model being fixed.

A dataset containing OHLCV data for stocks is provided. Contestants are free to design data processing strategies and perform feature engineering, such as constructing new indicators based on existing and/or external market data. Then the contestants are required to:

- Specify the state space, action space, and reward functions in the environment.
- Ensure that your data pipeline is reproducible with unseen, new data.
- Use the same model design without modification for a fair comparison. Specifically, teams are asked to use the PPO algorithm in the FinRL library with tunable hyperparameters.

Starter Kit
--------------
`Task 1 Starter Kit <https://github.com/Open-Finance-Lab/FinRL_Contest_2023?tab=readme-ov-file#task-1-data-centric-stock-trading-starter-kit>`_

Data
====

We provide the OHLCV data for 29 stocks from Jul 1, 2010 to Oct 24, 2023, for a total of 97208 pieces of data.

The OHLCV data corresponds to Open, High, Low, Close, and Volume data, which contain most of numerical information of a stock in time series and can help traders get further judgement and predictions such as the momentum, people's interest, market trends, etc.

Evaluation
==========

**Quantitative assessment** is the geometric mean of the rankings of the following metrics:

- **Portfolio cumulative return**. It measures the excess returns.
- **Sharpe ratio**. It takes into account both the returns of the portfolio and the level of risk.
- **Max drawdown**. It is the portfolio’s largest percentage drop from a peak to a trough in a certain time period, which provides a measure of downside risk.

**Quantitative assessment** of the report:

The assessment of the reports will be conducted by invited experts and professionals. The judges will independently rate the data and model analysis, results and discussion, robustness and generalizability, innovation and creativity, organization and readability, each accounting for 20% of the qualitative assessment.

The final ranking will be determined by the combination of 60% quantitative and 40% qualitative assessment.

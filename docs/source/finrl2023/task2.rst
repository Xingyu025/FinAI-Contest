=========================================
Task 2: Real Time Order Execution 
=========================================

This task focuses on building lightweight algorithmic trading systems in a fair environment. A template will be provided for contestants, and each team needs to write their functions in this template to perform order execution. We will provide an exchange of limit order book data to interact with contestants. After the contest, we will conduct real-time trading for all teams’ submissions at the same time. We would advise you to keep your algorithms lightweight.

Starter Kit
--------------
`Task 2 Starter Kit <https://github.com/Open-Finance-Lab/FinRL_Contest_2023?tab=readme-ov-file#task-2-real-time-order-execution-starter-kit>`_

Functions
=========

Contestants need to complete the `Strategy` class to implement order execution strategy and interact with our exchange. The functions in this class are explained below. Contestants are free to add new functions to the class but should not change the signatures of the provided functions.

- `place_market_order` allows you to place orders for the exchange at a given price/quantity, and you can call this in any function (including `init`).
- `on_orderbook_update` is called when a new order is placed by another algorithm (BUY or SELL).
- `on_trade_update` is called when two orders match (one BUY, one SELL). This could be your order or two other orders.
- `on_account_update` is called when one of your orders matches with another order.

The initial capital is **$100,000**.  
The libraries that can be used include:

- numpy  
- pandas  
- scipy  
- polars  
- scikit-learn  

The exchange allows **three tickers to trade**. The tickers will be randomly picked up during the evaluation period.

Exchange
========

- The exchange allows three tickers to trade. The tickers will be randomly picked up during the evaluation period.
- The exchange has a bot to add liquidity. It will be difficult to exhaust the liquidity in the market.
- During the final evaluation stage, all contestants' algorithms will compete together in our exchange at the same time. Therefore, one team's performance will be influenced by other teams.

Evaluation
==========

**Quantitative assessment** is pure **PnL**. Strategies that fail to adhere to the template will be disqualified, as well as any strategies that fail to pass linting.

**Qualitative assessment** of the report is the same as Task I:

The assessment of the reports will be conducted by invited experts and professionals. The judges will independently rate the data and model analysis, results and discussion, robustness and generalizability, innovation and creativity, organization and readability, each accounting for 20% of the qualitative assessment.

The final ranking will be determined by the combination of **60% quantitative** and **40% qualitative** assessment.


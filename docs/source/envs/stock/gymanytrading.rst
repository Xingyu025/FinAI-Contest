StockTradingEnv_gym_anytrading
==============================
Action Space
------------
The action space is a ``Discrete Space`` with two actions 0 and 1.
- 0 : Sell 
- 1 : Buy

State Space
-----------

The state space consists of the following components:

- **Account balance**: 1 element
- **Stock prices**: ``stock_dim`` elements
- **Holding positions**: ``stock_dim`` elements
- **Feature vector**: ``I × stock_dim`` elements (``I`` features per asset)

Thus, the total state dimension is ``1 + (I + 2) × stock_dim``.

Transition Dynamics
-------------------
Reward Design
-------------
The goal is to maximize the gain in asset value, thus the reward is defined as the change in asset value.

Initial State
-------------
At the start of each episode:

- **Account balance** is initialized to ``1,000,000``.
- **Holding positions** are initialized to ``0`` for all assets.
- **Stock prices** are set to the prices from the first trading day.
- **Feature vector** is initialized using the feature values for the first day.

Ending Condition
----------------
The episode terminates when the trading day would exceed the available data range.

Documentation
-------------
.. currentmodule:: finai_contest.env_stock_trading.env_stock_trading_gym_anytrading

.. autoclass:: StockTradingEnv_gym_anytrading
   :members:
   :inherited-members:
   :show-inheritance:
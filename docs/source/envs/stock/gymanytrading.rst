StockTradingEnv_gym_anytrading
==============================
Action Space
------------
The action space is discrete, with two actions:

- 0 : Sell 
- 1 : Buy

Observation Space
-----------------

The observation space consists of the following components:

- **Stock price**: 1 element
- **Price difference**: 1 element

Each observation contains a rolling history of these elements over the most recent ``window_size`` time steps.

Thus, the total observation space dimension is ``2*window_size``.

Transition Dynamics
-------------------

At each time step, the environment performs the following updates:


Reward Design
-------------
The goal is to maximize the gain in asset value, thus the reward is defined as the change in asset value.

Initial State
-------------
At the start of each episode:

- **Account balance** is initialized to ``1,000,000``.
- **Holding position** is initialized to ``0``.
- **Stock price** is set to the price from the first trading day.
- **Price difference** is initialized to 0.

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
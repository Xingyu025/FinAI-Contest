StockTradingEnv_FinRLMeta
=========================

Action Space
------------
The action space is a ``Box(-1,1,(stock_dim,),float32)``. 
Each element *a* is in the range [-1,1], representing the normalized number
of shares to buy or sell for the corresponding stock. 

- ``a > 0``: Buy ``a*hmax`` shares
- ``a = 0``: Hold
- ``a < 0``: Sell ``a*hmax`` shares

State Space
-----------
The state space consists of the following components:

- **Account balance** (1 element): current account balance
- **Stock prices** (``stock_dim`` elements): current price for each stock
- **Holding positions** (``stock_dim`` elements): number of shares held for each stock
- **Feature vector** (``I × stock_dim`` elements): technical indicators and turbulence index, ``I`` elements for each stock

Thus, the total state dimension is ``1 + (I + 2) × stock_dim``.

Transition Dynamics
-------------------

At each time step, the environment performs the following updates:

1. **Scale and Discretize Actions**

   The agent outputs a continuous action vector ``a_t ∈ [-1, 1]^{stock_dim}``.
   Actions are scaled to integer share counts:

   .. code-block:: python

      actions = (actions * self.hmax).astype(int)

2. **Risk Gate (Turbulence Check)**

   If ``turbulence_threshold`` is set and exceeded, all actions are overridden
   to force full liquidation:

   .. code-block:: python

      if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            actions = np.array([-self.hmax] * self.stock_dim)

3. **Execute Trades**

   Trades are executed in two passes to avoid interference between sells and buys:

   - **Sells first** (most negative actions first):
     
     For each asset ``i``:

     - If tradable and holdings > 0:
       
       .. math::

          \text{sell_qty} = \min(|a^{(i)}|, \text{holding}^{(i)})

          \text{cash} \leftarrow \text{cash} + \text{price}^{(i)} \text{sell_qty} (1 - \text{cost}^{\text{sell}}_i)

          \text{holding}^{(i)} \leftarrow \text{holding}^{(i)} - \text{sell_qty}

     - Otherwise, ``sell_qty = 0``.

   - **Buys second** (largest positive actions first, skipped if turbulence exceeds threshold):
     
     For each asset ``i``:

     - If turbulence < turbulence_threshold:

      .. math::

         \text{buy_qty}_{\max} = \left\lfloor \frac{\text{cash}}{\text{price}^{(i)} (1 + \text{cost}^{\text{buy}}_i)} \right\rfloor

         \text{buy_qty} = \min(\text{buy_qty}_{\max}, a^{(i)})

         \text{cash} \leftarrow \text{cash} - \text{price}^{(i)} \text{buy_qty} (1 + c^{\text{buy}}_i)

         \text{holding}^{(i)} \leftarrow \text{holding}^{(i)} + \text{buy_qty}

     - Otherwise, ``buy_qty = 0``.
4. **Advance Time and Update State**

   The environment advances to the next trading day, loads the new market data
   (prices, indicators), recomputes turbulence index, and reconstructs the state vector
   with updated cash, holdings, and features.

**Constraints**

- We set the transaction cost to 0.1% for buy and sell.



Reward Design
-------------
The goal is to maximize the gain in asset value, thus the reward is defined as the change in asset value, multiplied by a scaling multiplier.
It is calculated by ``(end_total_asset - begin_total_asset)*reward_scaling``, where ``reward_scaling`` is set to 0.0001.

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
.. currentmodule:: finai_contest.env_stock_trading.env_stock_trading_finrlmeta

.. autoclass:: StockTradingEnv_FinRLMeta
   :members:
   :inherited-members:
   :show-inheritance:
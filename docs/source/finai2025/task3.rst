=====================================================
Task 3 FinRL-DeFi
=====================================================

Task Overview
=================
This task focuses on training reinforcement learning (RL) agents to manage liquidity provisioning on Uniswap v3. The goal is to develop agents that can learn to optimize fee earnings while minimizing impermanent loss and gas costs through dynamic range adjustments. The task is based on the paper **Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning**, available at https://arxiv.org/abs/2501.07508. The paper also provides the baseline implementation.

Why This Matters
=================

Liquidity providers (LPs) on AMMs like Uniswap v3 face complex decisions that influence both profitability and risk. Traditionally, LPs rely on static or heuristic-based rebalancing strategies, which are suboptimal and often lead to losses when facing more sophisticated actors such as arbitrageurs. These actors can exploit price discrepancies between centralized and decentralized exchanges, systematically profiting at the expense of less sophisticated LPs.

This task explores the use of RL to automate liquidity management, allowing agents to learn directly from price dynamics. By doing so, it aims to improve the efficiency of liquidity provision and broaden access to DeFi participation, especially for non-expert users who currently face significant barriers to entry.

Starter Kit
=================

Data
----------------
The environment uses a single hourly price time series: `data_price_uni_h_time.csv`, which contains ETH/USDC price data from May 5, 2021 to April 29, 2024. All state variables are derived from this series. No additional data (e.g., volumes, centralized order book quotes or prices) is used in the baseline.

Participants may optionally expand the state with external data, provided no future information is used and environment logic remains unaltered.

Methodology
-------------------------------------

Data Processing and Feature Construction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The ETH/USDC hourly price series is preprocessed to construct the following state features. This calculation occurs within the custom Gymnasium environment to ensure consistency and stability during training. These core features are preserved across runs, although users are encouraged to augment them with additional indicators if desired.


.. list-table:: State Feature Descriptions
   :header-rows: 1

   * - Feature
     - Description
     - Method / Library Used

   * - ``price``
     - Raw hourly price of ETH/USDC
     - CSV import

   * - ``tick_index``
     - Log-scaled tick index computed from price 
     - Computed in custom environment according [1]_

   * - ``volatility``
     - Exponentially weighted std. dev. of log returns
     - ``pandas.DataFrame.ewm(...).std()``

   * - ``ma24``
     - 24-hour moving average
     - ``pandas.DataFrame.rolling(...).mean()``

   * - ``ma168``
     - 168-hour moving average (7 days)
     - ``pandas.DataFrame.rolling(...).mean()``

   * - ``bb_upper``
     - Upper Bollinger Band (T3 smoothing)
     - ``talib.BBANDS(..., matype=MA_Type.T3)``

   * - ``bb_middle``
     - Middle Bollinger Band
     - ``talib.BBANDS(..., matype=MA_Type.T3)``

   * - ``bb_lower``
     - Lower Bollinger Band
     - ``talib.BBANDS(..., matype=MA_Type.T3)``

   * - ``ADXR``
     - Average Directional Movement Index Rating
     - ``talib.ADX(...)``

   * - ``BOP``
     - Balance of Power
     - ``talib.BOP(...)``

   * - ``DX``
     - Directional Movement Index
     - ``talib.DX(...)``


The feature vector is dynamically constructed at each timestep using the current price and associated indicators. Feature windows are trimmed appropriately to remove NaNs (e.g., after MA or volatility computations).

MDP Formulation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The task is modeled as a discrete-time Markov Decision Process (MDP):

- **State**:
  A continuous vector combining price, tick index, interval width, liquidity, volatility, and technical indicators as listed above.

- **Action**:
  Discrete selection of liquidity width. Each action represents a symmetric range around the current price tick, for example: ``[0, 20, 50]``.

- **Reward**:  
  Defined as: ``reward_t = fee_t - lvr_t - gas_fee_t``  
  where:

  - ``fee_t``: Fees are collected only when the price remains within the selected range.
  - ``lvr_t``: Liquidity Value at Risk, penalizing capital inefficiency, scaled by volatility [3]_.
  - ``gas_fee_t``: A fixed gas fee (default $5) incurred at each repositioning action.


- **Environment**:
  A Gymnasium-compatible environment simulates interaction with a Uniswap v3-like AMM. It uses discrete hourly steps, assumes full observability, and prohibits lookahead. The agent repositions liquidity symmetrically around the current tick, with customizable tick spacing. Rewards are computed in real time using Uniswap v3 pricing formulas.

For more background on how Uniswap v3 and the mechanics of liquidity provisioning work, refer to [2]_.


Additional Mechanisms
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Transaction Costs**: Each repositioning action (i.e., nonzero ``action``) incurs a fixed gas fee (default \$5), specified by the ``gas_fee`` parameter.
- **Initial Capital**: The LP is initialized with a fixed quantity of asset X (default ``x = 2``), and the initial liquidity position is computed accordingly.
- **Discrete Action Range**: The environment maps discrete actions to symmetric tick intervals around the current tick. The set of allowable actions is defined in ``action_values``, e.g., ``[0, 45, 50, 55]``.
- **Forced Repositioning**: If the LP’s liquidity is entirely converted into a single asset (i.e., ``x = 0`` or ``y = 0``), the environment forces a reposition even if ``action = 0``, ensuring continuous participation in the pool.


These mechanisms are implemented in the Python script ``custom_env.py``, and their behavior can be configured via the associated YAML file ``config.yaml``.


RL Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We include a baseline agent trained using the **Proximal Policy Optimization (PPO)** algorithm, implemented via `stable-baselines3 <https://github.com/DLR-RM/stable-baselines3>`_.

PPO is a widely adopted reinforcement learning algorithm known for its stability and sample efficiency in both discrete and continuous control problems. We selected PPO as it is considered a state-of-the-art method in modern reinforcement learning pipelines, including recent advances in Reinforcement Learning from Human Feedback (RLHF) for language model fine-tuning [4]_. The original formulation and theoretical foundations of PPO are introduced in [5]_.

All PPO hyperparameters, architectural choices, and reward weights can be adjusted through the ``uniswap_rl_param_1108.yaml`` configuration file provided in the repository.


Evaluation
----------------
Agents are evaluated on an out-of-sample test period from January 29 to April 29, 2024.

Performance is evaluated based on the **cumulative reward** obtained over the test window defined above. The cumulative reward serves as a proxy for the **risk-adjusted PnL** of the liquidity provider's position.

This reward function aggregates three key components:

- **Fees earned** from providing liquidity within the chosen range;
- **Gas costs** incurred from repositioning;
- **Loss-versus-rebalancing (LVR)**, which penalizes adverse price movements when liquidity is not actively managed.

LVR accounts for both **impermanent loss** and the **opportunity cost** of not passively holding the assets (i.e., as on a centralized exchange). For further details on the concept and implications of LVR, refer to [3]_.

Baseline Performance
----------------------------

The baseline is a passive liquidity provider that rebalances every 500 timesteps using a fixed tick-width of 50. It alternates between two actions: ``0`` (hold position) and ``1`` (rebalance), simulating a non-adaptive strategy.

Performance metrics for the baseline are computed using the ``uniswap_test_bm.ipynb`` notebook, which evaluates the strategy on a rolling test window. 

Participants are expected to outperform this benchmark by training RL agents using the ``uniswap_test.py`` script, which relies on the custom environment implemented in the ``custom_env_folder``.


Citation
----------------------------
Please cite the original paper:

@article{xu2025improving,
  title={Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning},
  author={Xu, Haonan and Brini, Alessio},
  journal={arXiv preprint arXiv:2501.07508},
  year={2025}
}

References
----------------------------

.. [1] H. Adams, N. Zinsmeister, M. Salem, R. Keefer, and D. Robinson. *Uniswap v3 Core*. Tech. rep., Uniswap, 2021.

.. [2] M. Ottina, P. J. Steffensen, and J. Kristensen. *Automated Market Makers: A Practical Guide to Decentralized Exchanges and Cryptocurrency Trading*. Springer, 2023.

.. [3] Jason Milionis, Ciamac C. Moallemi, Tim Roughgarden, and Anthony Lee Zhang. *Automated Market Making and Loss-Versus-Rebalancing*, arXiv preprint arXiv:2208.06046, 2022.

.. [4] L. Ouyang et al., "Training language models to follow instructions with human feedback", *Advances in Neural Information Processing Systems*, vol. 35, pp. 27730–27744, 2022.

.. [5] J. Schulman et al., "Proximal Policy Optimization Algorithms", arXiv preprint arXiv:1707.063

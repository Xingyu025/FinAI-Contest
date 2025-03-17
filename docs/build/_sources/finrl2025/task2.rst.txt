==========================================
Task 2: FinRL-AlphaSeek for Crypto Trading
==========================================

Overview
========

This task focuses on developing robust and effective trading agents for cryptocurrencies through factor mining and ensemble learning. Participants will explore useful factors and ensemble methods specifically tailored for crypto trading. For this year's competition, the factor mining stage has been expanded, allowing participants to design their own factor mining models to generate powerful trading signals.

Participants are encouraged to apply various techniques to factor engineering, design component models, and use innovative methods to increase the diversity of component models in the ensemble. Additionally, participants need to specify the state space, action space, and reward function in their trading environment. The final model should seamlessly interact with the provided trading environment.

For reference, an example ensemble method using the majority voting approach is provided in the tutorial: `Task 1 Crypto Trading Ensemble <https://github.com/Open-Finance-Lab/FinRL_Contest_2024/tree/main/Tutorials/Task_1_tutorial>`_.

Detailed Description
====================

First Stage: Factor Mining and Alpha 101
----------------------------------------

**What is Alpha 101?**
  Alpha 101 refers to a set of formulaic alphas used to generate trading signals from financial data. These alphas leverage features derived from historical price and volume data to identify profitable trading opportunities.
  
  In this task, factor mining involves deriving meaningful financial indicators from limit order book (LOB) data and using techniques like recurrent neural networks (RNNs) to generate robust predictive signals.

**Architecture**
  The architecture consists of an ensemble of reinforcement learning agents trained using massively parallel simulations. The system integrates deep reinforcement learning (DRL) with a supervised learning stage where Alpha101 features are extracted and processed using RNN-based networks.
  
  The pipeline includes:
  
  * **Factor Mining**: Weak Alpha101 signals processed through LSTM+GRU networks to generate strong factors.
  * **Agent Training**: DRL agents trained in a parallel market environment to optimize trading decisions.
  * **Ensemble Method**: A combination of agents using majority voting or weighted action averaging to enhance robustness.

**Design Rationale**
  This design aims to mitigate policy instability and sampling bottlenecks, two major challenges in financial reinforcement learning.
  
  Factor mining provides stronger signals to RL agents, improving decision quality.
  
  Running thousands of parallel simulations on GPUs speeds up training, allowing agents to adapt to volatile markets.

Second Stage: Reinforcement Learning for Crypto Trading
-------------------------------------------------------

**RL Setting for Crypto Trading**
  The crypto trading task is modeled as a Markov Decision Process (MDP), where:
  
  * **State space** includes order book features, price data, and technical indicators.
  * **Action space** consists of discrete trade actions (buy, sell, hold) determined by DQN-based models.
  * **Reward function** is based on profit and loss calculations, emphasizing risk-adjusted returns.
  
  Reinforcement learning agents learn to optimize trading strategies by interacting with a simulated market environment.

**Parallel Environment**
  The parallel environment consists of thousands of simulated trading environments running concurrently on GPUs.
  
  This approach enhances sample efficiency and speeds up convergence, making it feasible to train complex trading agents in a short period.
  
  A market replay simulator is used to generate training samples from historical data, ensuring the agent learns from realistic market conditions.

**Agent Selection Rationale**
  Different agents have varying strengths:
  
  * **PPO (Proximal Policy Optimization)**: Stability in training, suitable for continuous action spaces.
  * **SAC (Soft Actor-Critic)**: Efficient exploration, performs well in complex environments.
  * **DDPG (Deep Deterministic Policy Gradient)**: Works well with continuous control tasks.
  * **DQN and Variants (Double DQN, Dueling DQN)**: Well-suited for discrete action spaces, effective in high-frequency trading.
  
  The diversity of agents allows the ensemble method to leverage multiple perspectives on market conditions, improving robustness.

**Increasing Agent Diversity**
  * Training different models on varied market scenarios
  * Using different subsets of historical data to train agents on different market regimes
  * Applying KL divergence penalties in the loss function to encourage diverse policy behaviors
  * Experimenting with different hyperparameters to ensure agents do not converge to similar strategies

Dataset
=======

A dataset containing second-level Limit Order Book (LOB) data for Bitcoin is provided. Please download it from `here <https://drive.google.com/drive/folders/1ExVPS1d77oPOHXMRYdtKpdEC0PycthKW?usp=sharing>`_. This dataset is essential for training both supervised learning models and reinforcement learning agents.

Submission Guidelines
=====================

Participants should submit:

1. Trained models and the scripts to load and test them
2. A README explaining their submission and how it should be evaluated
3. Code for factor mining, ensemble methods, and any modifications to the trading environment

Evaluation Criteria
===================

Models will be evaluated based on:

* Cumulative Return
* Sharpe Ratio
* Win/Loss Ratio

By combining factor mining, reinforcement learning, and ensemble methods, this task aims to advance state-of-the-art crypto trading strategies through AI-driven approaches.
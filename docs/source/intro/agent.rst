=============================
Agent
=============================

In FinRL, the agent is a deep reinforcement learning (DRL) model that learns to take trading actions based on the current state of the market. The agent interacts with the environment, which represents the financial market, and takes trading actions to maximize its cumulative reward over time. In this section, we summarize supported RL libraries and different algorithms used in the FinAI contests.

Supported Libraries
---------------------------
- Isaac Gym for massively parallel simulations,
- OpenAI Gym, MuJoCo, PyBullet, ElegantRL for benchmarking.

Algorithms
-------------
FinRL implements the following model-free deep reinforcement learning (DRL) algorithms:

- DDPG, TD3, SAC, PPO, REDQ for continuous actions in single-agent environment,
- DQN, Double DQN, D3QN for discrete actions in single-agent environment,




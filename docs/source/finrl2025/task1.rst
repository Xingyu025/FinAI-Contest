=========================================
Task 1 FinRL-DeepSeek for Stock Trading
=========================================

This task is about developing automated stock trading agents trained on stock prices and financial news data by combining reinforcement learning and large language models (LLMs). Participants can build upon the `FinRL-DeepSeek project <https://github.com/benstaf/FinRL_DeepSeek>`_ (e.g., with new prompts, new ways to inject LLM-processed news signals into the RL agent, new RL algorithms like GRPO) or explore more computationally intensive directions, such as adapting variants of the DeepSeek R1 training method to this stock trading task.

Datasets
--------

The `Financial News and Stock Price Integration Dataset (FNSPID) <https://huggingface.co/datasets/Zihan1004/FNSPID>`_ comprises stock prices and 15 million time-aligned financial news records for Nasdaq companies, covering the period from 1999 to 2023. The processed training dataset based on the FNSPID `are also provided <https://huggingface.co/datasets/benstaf/nasdaq_2013_2023>`_.

Participants are also encouraged to utilize publicly available data, such as Twitter, or develop scraping/API AI agents for this purpose. Some teams can choose to focus their submission on improving the dataset, while others may focus on enhancing trading agents.

Training and Environments
=========================

To train the various models, follow the instructions below:

PPO Training
------------

To train the PPO model, run the following command:

.. code-block:: bash

    nohup mpirun --allow-run-as-root -np 8 python train_ppo.py > output_ppo.log 2>&1 &

CPPO Training
-------------

To train the CPPO model, use the script ``train_cppo.py``

PPO-DeepSeek Training
---------------------

For training PPO-DeepSeek, use the script ``train_ppo_llm.py``

CPPO-DeepSeek Training
----------------------

For training CPPO-DeepSeek, use the script ``train_cppo_llm_risk.py``

Environment Files
-----------------

The environment files for each model are as follows:

- ``env_stocktrading.py``: Used for PPO and CPPO, same as in the original FinRL.
- ``env_stocktrading_llm.py`` or ``env_stocktrading_llm_01.py``: Used for PPO-DeepSeek, depending on the desired LLM influence.
- ``env_stocktrading_llm_risk.py`` or ``env_stocktrading_llm_risk_01.py``: Used for CPPO-DeepSeek.

Log Files
---------

Log files such as ``output_ppo.log`` should be monitored during training. Key metrics to observe include:

- ``AverageEpRet``
- ``KL``
- ``ClipFrac``

Evaluation
==========

Evaluation in the trading phase (2019-2023) is conducted in the ``FinRL_DeepSeek_backtest.ipynb`` Colab notebook. The metrics used are:

- Information Ratio
- CVaR
- Rachev Ratio

Adding ``Outperformance frequency`` would be beneficial.

Submission
==========

For evaluation, please submit the following links along with your paper:

- GitHub repository link
- Hugging Face link
- Colab notebook link




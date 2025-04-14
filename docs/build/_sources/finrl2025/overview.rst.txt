=============================
Overview
=============================

Financial reinforcement learning (FinRL) [1]_ [2]_ is an interdisciplinary field that applies reinforcement learning to financial tasks such as portfolio management, algorithmic trading, and option pricing [3]_ [4]_ [5]_. Recent breakthroughs in large language models (LLMs) are driving open finance, which provides affordable and scalable solutions for customers to make intelligent decisions, enabling personalized financial search and robo-advisors.

Features
---------------

The **FinRL Contest 2025** explores and evaluates the capability of machine learning methods in finance, with the following featured tracks:

1. **FinRL-DeepSeek**. 
   DeepSeek’s groundbreaking open models have demonstrated strong capabilities on par with proprietary models. FinRL-DeepSeek [6]_ integrates LLM-generated signals in FinRL to enhance stock trading strategies, inlcuding sentiment scores and risk levels extracted from financial news. Participants are encouraged to explore and innovate with FinRL-DeepSeek.

2. **FinRL-AlphaSeek**.  
   Factors are crucial in driving trading decisions and in enabling traders to design efficient, data-driven strategies. This task involves two stages: factor engineering and ensemble learning. Participants are encouraged to perform factor engineering and utilize ensemble methods to develop robust trading agents.

3. **Open FinLLM Leaderboard**.  
   The `Open FinLLM Leaderboard <https://huggingface.co/spaces/finosfoundation/Open-Financial-LLM-Leaderboard>`_ [7]_ is an open platform for evaluating LLMs on a wide range of financial tasks, emphasizing transparency and reproducibility. In addition to general financial benchmarks, we have extended the leaderboard with datasets focused on digital regulatory reporting, originally developed for the `Regulations Challenge at COLING 2025 <https://coling2025regulations.thefin.ai/home>`_ [8]_. These include:

   - **Common Domain Model (CDM)**
   - **Model Openness Framework (MOF)**
   - **eXtensible Business Reporting Language (XBRL)**

   Participants are invited to contribute to these benchmarks and also explore reinforcement learning approaches for improving LLMs' ability in various financial tasks.

Tasks
---------------
We organize four tasks to promote open finance:

1. **FinRL-DeepSeek for Stock Trading**.
    Develop stock trading agents by integrating LLM-generated signals in FinRL, using financial news and market data.
2. **FinRL-AlphaSeek for Crypto Trading**.
    Develop crypto trading agents using factor mining and ensemble learning to improve trading performance.
3. **Open FinLLM Leaderboard – Models with Reinforcement Fine-Tuning (ReFT)**.
    Train and fine-tune financial LLMs with ReFT and compete in the Open FinLLM Leaderboard.  
4. **Open FinLLM Leaderboard – Digital Regulatory Reporting (DRR)**.
    Train and fine-tune LLMs to improve their ability in digital regulatory reporting: CDM, MOF, and XBRL. 

For Task 1 and Task 2, we provide **parallel market environments** [9]_ that incorporate LLM-generated signals and support massivley parallel simulation.


References
==========

.. [1] X.-Y. Liu, Z. Xia, H. Yang, J. Gao, D. Zha, M. Zhu, Christina D. Wang*, Zhaoran Wang, and Jian Guo. Dynamic datasets and market environments for financial reinforcement learning. *Machine Learning Journal, Springer Nature*, 2023.

.. [2] X.-Y. Liu, Z. Xia, J. Rui, J. Gao, H. Yang, M. Zhu, C. Wang, Z. Wang, J. Guo. FinRL-Meta: Market environments and benchmarks for data-driven financial reinforcement learning. *NeurIPS, Special Track on Datasets and Benchmarks*, 2022.

.. [3] Ben Hambly, Renyuan Xu, and Huining Yang. Recent advances in reinforcement learning in finance. *Mathematical Finance*, vol. 33, no. 3, pp. 437–503, 2023.

.. [4] Shuo Sun, Rundong Wang, and Bo An. Reinforcement learning for quantitative trading. *ACM Transactions on Intelligent Systems and Technology*, vol. 14, no. 3, pp. 1–29, 2023.

.. [5] Yahui Bai, Yuhe Gao, Runzhe Wan, Sheng Zhang, and Rui Song. A review of reinforcement learning in financial applications. *Annual Review of Statistics and Its Application*, vol. 12, no. 1, pp. 209–232, 2025.

.. [6] Mostapha Benhenda. 2025. FinRL-DeepSeek: LLM-Infused Risk-Sensitive Reinforcement Learning for Trading Agents. arXiv preprint arXiv:2502.07393 (2025).

.. [7] Shengyuan Colin Lin, Felix Tian, Keyi Wang, Xingjian Zhao, Jimin Huang, Qian-qian Xie, Luca Borella, Christina Dan Wang, Matt White, Kairong Xiao, Xiao-Yang Liu Yanglet, and Li Deng. 2024. Open FinLLM Leaderboard: Towards Financial AI Readiness. *International Workshop on Multimodal Financial Foundation Models (MFFMs), ACM International Conference on AI in Finance*, 2024.

.. [8] Keyi Wang, Jaisal Patel, Charlie Shen, Daniel Kim, Andy Zhu, Alex Lin, Luca Borella, Cailean Osborne, Matt White, Steve Yang, Kairong Xiao, and Xiao-Yang Liu Yanglet. 2025. A Report on Financial Regulations Challenge at COLING 2025. *Proceedings of the Joint Workshop of FinNLP, FNP, and LLMFinLegal*, 2025.

.. [9] Keyi Wang, Kairong Xiao, and Xiao-Yang Liu. Parallel Market Environments for FinRL Contests. arXiv preprint arXiv:2504.02281 (2025).
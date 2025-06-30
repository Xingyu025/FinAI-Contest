=============================
Overview
=============================

`Contest Website <https://open-finance-lab.github.io/FinAI_Contest_2025/>`_; `Starter Kit <https://github.com/Open-Finance-Lab/FinAI_Contest_2025>`_


As AI continues to advance at a fast pace, more FinAI agents are being developed for the finance sector, such as `FinRL trading agents <https://berylventures.com/spotlights>`_ [1,2,3]_, FinGPT agents [4,5]_ with multimodal capabilities [6]_, and regulatory reporting agents [7]_. The FinAI Contest 2025 encourages the development of open FinAgents based on the frameworks FinRL [2,3]_ and FinGPT [4]_.

The FinAI Contest 2025 explores and evaluates the capability of machine learning methods in finance, with the following features:

1. **FinRL-DeepSeek**. In the volatile cryptocurrency markets, timely interpretation of market sentiment is critical. Cryptocurrency markets are highly sensitive to news headlines, tweets, regulatory shifts, and viral narratives. However, the massive amount of news, social media posts, and documents can overwhelm individual traders. Is it possible for an individual trader to ask a sentiment analyst to interpret market news and consult a risk manager to detect red flags in blockchain activity — all timely and on demand? AI agents are making this happen. These FinGPT-powered agents, such as a sentiment analysis agent and a risk management agent, form a professional investment team to extract actionable signals from financial news, tweets, and filings. In this task, we encourage participants to explore FinGPT-engineered signals and integrate them into a FinRL trading agent for crypto trading.

2. **FinGPT Agents in Real Life**. AI agents have seen rapid development and have been applied to various financial tasks recently. They have been applied to `financial analysis and accounting <https://openai.com/solutions/ai-for-finance/>`_ and are capable of `analyzing SEC filings <https://fintool.com/press/fintool-outperforms-analysts-sec-filings>`_. Researchers also show that [large language models (LLMs) can `pass CFA Level I and II exams <https://aclanthology.org/2024.emnlp-industry.80/>`_, achieving performance above the human average. While BloombergGPT is the first financial LLM pre-trained on large-scale financial data, it is no longer unmatched. Many open FinLLMs, such as FinGPT [4]_, have outperformed BloombergGPT on public benchmarks. It is not hard to build your own FinGPT agent that rivals or surpasses BloombergGPT and serves as professional financial assistant. This task encourages participants to fine-tune open LLMs and develop FinAgents for financial analytics, including the CFA exam, BloombergGPT's public benchmark tasks, and XBRL tasks.

3. **FinRL-DeFi**. Decentralized Finance (DeFi) is reshaping the crypto economy by enabling peer-to-peer trading, lending, and liquidity provision without banks, brokers, or intermediaries. As a core component of DeFi, the automated market makers (AMMs) act as liquidity providers (LPs) and replace order books with liquidity pools. However, liquidity provision is complex and risky. For example, impermanent loss can occur for LPs when the price of assets in a liquidity pool diverges from their initial value. LPs must actively manage price ranges, balance transaction fees, and mitigate impermanent loss. How can we develop an intelligent LP that adapts to market dynamics in DeFi? In this contest, we challenge participants to develop reinforcement learning agents that act as LPs [8]_, dynamically adjusting their liquidity positions in response to market conditions. 

Tasks
---------------

The **FinAI Contest 2025** has the following three tasks:

1. **FinRL-DeepSeek for Crypto Trading**: Develop crypto trading agents by integrating LLM-generated signals in FinRL, using financial news and market data.

2. **FinGPT Agents in Real Life**: Fine-tune LLMs and develop financial agents for financial analytics.
        
        - **CFA exam**: The CFA (Chartered Financial Analyst) exam is a globally recognized exam for financial analysts. It covers a wide range of financial topics, such as investment, economics, and quantitative methods.
        - **BloombergGPT** [2]_: Compare the performance of your model with BloombergGPT on its public financial benchmarks.
        - **XBRL**: XBRL (eXtensible Business Reporting Language) is a standard language for electronic communication of business and financial data. It has been widely used in regulatory filings, such as SEC filings.

3. **FinRL-DeFi**: Develop reinforcement learning agents that act as liquidity providers (LPs) in DeFi for liquidity provisioning.


.. [1] Keyi Wang, Nikolaus Holzer, Ziyi Xia, Yupeng Cao, Jiechao Gao, Anwar Walid, Kairong Xiao, and  Xiao-Yang Liu Yanglet. FinRL Contests: Benchmarking Data-driven Financial Reinforcement Learning Agents. arXiv preprint arxiv.org/abs/2504.02281, 2025.

.. [2] Xiao-Yang Liu, Ziyi Xia, Jingyang Rui, Jiechao Gao, Hongyang Yang, Ming Zhu, Christina Wang, Zhaoran Wang, and Jian Guo. FinRL-Meta: Market environments and benchmarks for data-driven financial reinforcement learning. Advances in Neural Information Processing Systems 35, 1835-1849, 2022.

.. [3] Xiao-Yang Liu, Hongyang Yang, Qian Chen, Runjia Zhang, Liuqing Yang, Bowen Xiao, and Christina Dan Wang. FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance. Deep Reinforcement Learning Workshop, NeurIPS. 2020.

.. [4] Xiao-Yang Liu, Guoxuan Wang, Hongyang Yang, and Daochen Zha. FinGPT: Democratizing internet-scale data for financial large language models. Workshop on Instruction Tuning and Instruction Following, NeurIPS 2023.

.. [5] Felix Tian, Ajay Byadgi, Daniel Kim, Daochen Zha, Matt White, Kairong Xiao, Xiao-Yang Liu. Customized FinGPT Search Agents Using Foundation Models. ACM International Conference on AI in Finance, 2024.

.. [6] Xiao-Yang Liu Yanglet, Yupeng Cao, and Li Deng. Multimodal financial foundation models (MFFMs): Progress, prospects, and challenges.  arXiv preprint arxiv.org/abs/2506.01973, 2025.

.. [7] Shijie Han, Haoqiang Kang, Bo Jin, Xiao-Yang Liu, Steve Yang. XBRL Agent: Leveraging Large Language Models for Financial Report Analysis. ACM International Conference on AI in Finance, 2024.

.. [8] Haonan Xu and Alessio Brini. Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning. arXiv: 2501.07508, 2025.

.. [9] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, Gideon Mann. BloombergGPT: A Large Language Model for Finance. arXiv: 2303.17564, 2023.

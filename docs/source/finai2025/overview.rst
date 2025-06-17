=============================
Overview
=============================

As artificial intelligence (AI) continues to advance rapidly, more and more AI agents are being developed and applied to various financial tasks, such as trading agents, search agents, and regulatory reporting agents. The FinAI Contest 2025 aims to encourage the development of advanced financial agents and benchmark their performance across different financial tasks.

The FinRL Contest 2025 explores and evaluates the capability of machine learning methods in finance, with the following features:

1. **FinRL-DeepSeek**. Generating alpha signals is crucial for making informed trading decisions. As individual investors without resources, what if we can ask Warren Buffett for value-investing advice, consult a risk manager to identify red flags in SEC filings, or engage a sentiment analyst to interpret the tone of market news — all timely and on demand. AI agents make this possible. These LLM-powered agents, such as a Warren Buffett agent, a sentiment analysis agent, and a risk management agent, form a professional investment team to extract actionable signals from financial documents. In this contest, we encourage participants to explore LLM-generated signals and integrate them into FinRL for crypto trading.

2. **FinAI Agents**. AI agents have seen rapid development and have been applied to various financial tasks recently. However, can they truly serve as professional financial assistants in real life? Imagine an AI agent that can help us prepare for CFA exams, analyze the SEC filings, or navigate financial contracts. Despite this promise, there still remain `doubts that AI cannot perform financial analytics <https://www.cnbc.com/2023/12/19/gpt-and-other-ai-models-cant-analyze-an-sec-filing-researchers-find.html>`_. This task encourages participants to take on that challenge: fine-tune LLMs and develop financial agents for financial analytics, including the CFA exam, BloombergGPT's public benchmarks, and XBRL.

3. **Decentralized Finance (DeFi)**. DeFi is an emerging blockchain-based peer-to-peer financial ecosystem and has transformed the crypto market. It enables users to trade, lend, and borrow assets without intermediaries like brokers. Instead, smart contracts automate these processes in a trustless and permissionless manner. Unlike traditional centralized crypto exchanges, DeFi protocols (e.g., Uniswap v3) employ automated market maker (AMM) models, which replace order books with liquidity pools. The introduction of concentrated liquidity in Uniswap v3 significantly improved capital efficiency but made liquidity provision more complex and risky. LPs must actively manage price ranges, balance transaction fees, and mitigate impermanent loss. In this contest, we encourage participants to develop reinforcement learning agents that act as LPs [2], dynamically adjusting liquidity positions in response to market conditions.

Tasks
---------------

The **FinAI Contest 2025** has the following three tasks:

1. **FinRL-DeepSeek for Crypto Trading**: Develop crypto trading agents by integrating LLM-generated signals in FinRL, using financial news and market data.

2. **FinAgents in Real Life**: Fine-tune LLMs and develop financial agents for financial analytics.
        
        - **CFA exam**: The CFA (Chartered Financial Analyst) exam is a globally recognized exam for financial analysts. It covers a wide range of financial topics, such as investment, economics, and quantitative methods.
        - **BloombergGPT** [1]: Compare the performance of your model with BloombergGPT on its public financial benchmarks.
        - **XBRL**: XBRL (eXtensible Business Reporting Language) is a standard language for electronic communication of business and financial data. It has been widely used in regulatory filings, such as SEC filings.

3. **DeFi Liquidity Provisioning**: Develop reinforcement learning agents that act as liquidity providers (LPs) in DeFi for liquidity provisioning.


.. [1] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, Gideon Mann. BloombergGPT: A Large Language Model for Finance. arXiv: 2303.17564 (2023).

.. [2] Haonan Xu and Alessio Brini. Improving DeFi Accessibility through Efficient Liquidity Provisioning with Deep Reinforcement Learning. arXiv: 2501.07508 (2025).
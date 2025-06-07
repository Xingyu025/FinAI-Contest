=============================
Overview
=============================

As artificial intelligence (AI) continues to advance rapidly, more and more AI agents are being developed and applied to various financial tasks, such as trading agents, search agents, and regulatory reporting agents. The FinAI Contest 2025 aims to encourage the developemnt of advanced financial agents and benchmark their performance across different financial tasks.

Tasks
---------------

The **FinAI Contest 2025** has the following featured tasks:

1. **FinRL-DeepSeek for Crypto Trading**: Develop crypto trading agents by integrating LLM-generated signals in FinRL, using financial news and market data.

    - Cryptocurrency markets are highly volatile due to drastic price fluctuations and market sentiment shifts. Reinforcement learning has been applied to cryptocurrency trading in practice, which can be well adapted to the dynamic markets. Financial documents such as news, tweets, and regulatory updates provide valuable insights into market sentiment. LLMs have shown strong capabilities in processing financial documents adn extracting actionable signals from them. 
    - This task encourages participants to explore and innovate with FinRL-DeepSeek, leveraging LLM-generated signals and FinRL to develop cryptocurrency trading agents. 

2. **FinAgent using Fine-Tuning**: Fine-tune LLMs and develop financial agents for professional financial language.

    - AI agent has seen rapid development recently. For example, The AI agent "Marc Andreessen" was trained on Marc's blogs and writings, which emulates Marc Andreessen's investment strategies. It has contributed to the ai16z DAO which once reached a market capitalization of nearly $2.2 billion. Another example is `AI-powered hedge fund <https://github.com/virattt/ai-hedge-fund>`_, which aims to explore the use of AI to make trading decisions.
    - Fine-tuning an LLM is a much simpler and more efficient way to develop an AI agent than training from scratch. This task encourages participants to fine-tune LLMs and develop financial agents for professional financial language, including CFA exam, XBRL, and CDM.

        - **CFA exam**: The CFA (Chartered Financial Analyst) exam is a globally recognized exam for financial analysts. It covers a wide range of financial topics, such as investment, economics, and quantitative methods.
        - **XBRL**: XBRL (eXtensible Business Reporting Language) is a standard language for electronic communication of business and financial data. It has been widely used in regulatory filings, such as SEC filings.
        - **CDM**: CDM (Common Domain Model) is a machine-oriented model for managing the lifecycle of financial products and transactions.
    - This task encourages participants to fine-tune LLMs and develop financial agents for professional financial language.

Timeline (Tentative)
----------------------
06/01 - 07/25: Contest preparation.

    - Task definition (what participants need to do, evaluaiton metrics)
    - Datasets.

        - Task 1: BTC hour/minute data and BTC news 
            - find existing datasets (e.g. Kaggle or past competitions) or download via API (coinbase, Binance with `sylvanus <https://www.sylvanus.io/>`_)
            - split into train/validation/test sets
        - Task 2: (discuss with **Jaisal**) 
            - identify data sources for fine-tuning (collect links) 
            - benchmark datasets 

                - CFA exam: focusing only on regulations?
                - XBRL: select some from existing datasets.
                - CDM: collaborate with **Yan Wang** to develop the datasets for tagging tasks.

    - Code and documentation 

        - Task 1: FinRL-DeepSeek
            
            - Generate sentiment scores and risk levels from LLMs
            - FinRL with LLM-generated signals for crypto trading using massively parallel simulation on GPUs.

        - Task 2: FinLoRA: discuss with **Jaisal**.

    - Baseline performance 

        - Task 1: market index, buy-and-hold, etc.
        - Task 2: GPT-4o, Llama, Mistral, DeepSeek-V3 on CDM, XBRL, CFA exam. (need to identify similar model size with participants' models)

    - Tutorials and resources (code demo, docs, repos, papers, blogs, etc).
    - Website and promotion materials. (**Jin Bo**?)

07/25: Team Registration Opens. 

08/01: Starter Kit Release. The starter kit will be released, including datasets, baseline code, and documentation.

10/10: Model submission deadline. 

10/17: Paper submission deadline.

10/20: Paper Notification

10/31 :Paper Cemera Ready

10/31: Leaderboard Announcement.

Nov: conference.

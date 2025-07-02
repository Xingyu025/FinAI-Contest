====================================
Task 2 FinGPT Agents in Real Life
====================================

Task Overview
====================================
This task is to fine-tune LLMs and develop financial agents for financial analytics, including the CFA exam, BloombergGPT's public benchmark tasks, and XBRL tasks. We encourage participants to use LoRA and reinforcement fine-tuning.

- CFA exam: The CFA (Chartered Financial Analyst) exam is a globally recognized exam for financial analysts. It covers a wide range of financial topics, such as investment, economics, and quantitative methods.
- BloombergGPT: Compare the performance of your model with BloombergGPT on its public financial benchmarks.
- XBRL: XBRL (eXtensible Business Reporting Language) is a standard language for electronic communication of business and financial data. It has been widely used in regulatory filings, such as SEC filings.

Why This Matters
====================================
AI agents have seen rapid development and have been applied to various financial tasks recently. They have been applied to `financial analysis and accounting <https://openai.com/solutions/ai-for-finance/>`_ and are capable of `analyzing SEC filings <https://fintool.com/press/fintool-outperforms-analysts-sec-filings>`_. Researchers also show that large language models (LLMs) can `pass CFA Level I and II exams <https://aclanthology.org/2024.emnlp-industry.80/>`_, achieving performance above the human average. While BloombergGPT is the first financial LLM pre-trained on large-scale financial data, it is no longer unmatched. Many open FinLLMs, such as FinGPT, have outperformed BloombergGPT on public benchmarks. It is not hard to build your own FinGPT agent that rivals or surpasses BloombergGPT and serves as professional financial assistant. This task encourages participants to fine-tune open LLMs and develop FinAgents for financial analytics, including the CFA exam, BloombergGPT's public benchmark tasks, and XBRL tasks.

Question Sets
====================================
These question sets contain question-answer pairs collected and organized for evaluating model capabilities across CFA exams, BloombergGPT benchmark tasks, and XBRL tasks. These question sets are sampled from the test split of the datasets, which are used to benchmark your agent's performance. You **SHOULD NOT** use it or the entire test split for fine-tuning or training.

CFA Exams
------------
The Chartered Financial Analyst (CFA) exams cover a wide range of practice scenarios in finance and accounting, designed to test candidates' understanding of core concepts in finance, investment tools, and ethical and professional standards. CFA has three levels of exams, each with a different focus and complexity. Here we provide the question sets for Level I and Level II exams, some of which include multiple-choice questions with contextual scenarios.

Question Set Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: CFA Exam Question Sets Statistics
   :header-rows: 1
   :widths: 15 10 15 10 50

   * - **Exam Level**
     - **# Exams**
     - **Questions/Exam**
     - **Total**
     - **Description**
   * - Level I
     - 3
     - 180
     - 540
     - Level I has multiple-choice questions to test a candidate's understanding of core concepts in finance, investment tools, and ethical and professional standards.
   * - Level II
     - 2
     - 88
     - 176
     - Level II has multiple-choice questions with contextual scenario, emphasizing the application of concepts and analytical skills.

.. list-table:: CFA Exam Question Sets Statistics by Topics
   :header-rows: 1
   :widths: 30 10 10 10

   * - **Topic**
     - **Level I**
     - **Level II**
     - **Total**
   * - Ethics
     - 81
     - 24
     - 105
   * - Quantitative Methods
     - 43
     - 12
     - 55
   * - Economics
     - 41
     - 12
     - 53
   * - Financial Reporting
     - 66
     - 24
     - 90
   * - Corporate Issuers
     - 41
     - 12
     - 53
   * - Equity Investments
     - 67
     - 24
     - 91
   * - Fixed Income
     - 65
     - 24
     - 89
   * - Derivatives
     - 43
     - 12
     - 55
   * - Alternative Investments
     - 36
     - 12
     - 48
   * - Portfolio Management
     - 57
     - 20
     - 77
   * - **Total**
     - **540**
     - **176**
     - **716**

Question Set Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
**CFA Exam Level I Example**

.. list-table:: CFA Exam Level I Example
   :header-rows: 1
   :widths: 60 10 10 10 10

   * - **Question**
     - **Choice 1**
     - **Choice 2**
     - **Choice 3**
     - **Answer**
   * - Sammy Sneadle, CFA, is the founder and portfolio manager of the Everglades Fund. In its first year, the fund generated a return of 30 percent. Building on the fund's performance, Sneadle created new marketing materials that showed the fund's gross 1-year return, as well as the 3 and 5-year returns, which he calculated by using back-testedperformance information. As the marketing material is used only for presentations to institutional clients, Sneadle does not mention the inclusion of back-tested data. According to the Standards of Practice Handbook, how did Sneadle violate CFA Institute Standards of Professional Conduct?
     - A. He did not disclose the use of back-tested data.
     - B. He failed to deduct all fees and expenses before calculating the fund's track record.
     - C. The marketing materials only include the Everglades Fund's performance and are not a weighted composite of similar portfolios.
     - A. He did not disclose the use of back-tested data.

**CFA Exam Level II Example**

.. list-table:: CFA Level II Example
   :header-rows: 1
   :widths: 10 30 50 15 15 15 15

   * - **Question Number**
     - **Context**
     - **Question**
     - **Choice 1**
     - **Choice 2**
     - **Choice 3**
     - **Answer**
   * - 1
     - Maria Harris is a CFA Level 3 candidate and portfolio manager for Islandwide Hedge Fund. Harris is commonly involved in complex trading strategies on behalf of...
     - Which action by Park violated Standard III(B) Duties to Clients: Fair Dealing?
     - A. Increasing allocation to the problem client.
     - B. Decreased allocation to the brother-in-law and other firm clients.
     - C. Both actions are violations.
     - C. Both actions are violations.

Please view `FinLoRA Documentation Financial Certification Tasks page <https://finlora-docs.readthedocs.io/en/latest/tasks/certification_tasks.html>`__ for more examples and details.


BloombergGPT [2]_ Public Benchmark Datasets
------------------------------------------------------------

BloombergGPT has released a set of public benchmark datasets for financial tasks, which we will use to evaluate the performance of your FinGPT agents. These datasets cover various financial tasks, including sentiment analysis, named entity recognition, math calculation, and so on.

Question Set Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 10 10 10 30 50

   * - **Dataset**
     - **Size**
     - **Metrics**
     - **# of shots**
     - **Dataset Link**
     - **Description**
   * - Financial Phrasebank Dataset (FPB) [3]
     - 150
     - F1
     - 5
     - `Link <https://huggingface.co/datasets/ChanceFocus/en-fpb>`__
     - The Financial Phrasebank Dataset includes a sentiment classification task on sentences from financial news. Any news that could benefit/hurt an investor is considered positive/negative and neutral otherwise.
   * - FiQA SA [4]
     - 150
     - F1
     - 5
     - `Link <https://huggingface.co/datasets/ChanceFocus/flare-fiqasa>`__
     - This is a sentiment analysis task to predict the aspect-specific sentiment in English financial news and microblog headlines, from the 2018 challenge on financial question answering and opinion mining.
   * - Headline [5]
     - 150
     - F1
     - 5
     - `Link <https://huggingface.co/datasets/FinGPT/fingpt-headline-cls>`__
     - This is a binary classification task on whether a gold commodity news headline includes specific information.
   * - NER [6]
     - 150
     - F1
     - 20
     - `Link <https://huggingface.co/datasets/FinGPT/fingpt-ner-cls>`__
     - Named entity recognition task on financial agreements filed with the SEC for credit risk assessment.
   * - ConvFinQA [7]
     - 150
     - Match Accuracy
     - /
     - `Link <https://huggingface.co/datasets/FinGPT/fingpt-convfinqa>`__
     - Given S&P 500 earnings reports with tables and text, answer questions requiring numerical reasoning.

We will sample 150 questions from the test split for our evaluation.

Question Set Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**FPB Examples**

.. list-table:: Financial Phrasebank Sentiment Examples
   :header-rows: 1
   :widths: 5 20 20 50 10 10

   * - **ID**
     - **Query**
     - **Text**
     - **Choices**
     - **Answer**
     - **Gold**
   * - fpb3876
     - Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.  
       Text: The new agreement , which expands a long-established cooperation between the companies , involves the transfer of certain engineering and documentation functions from Larox to Etteplan .  
       Answer:
     - The new agreement , which expands a long-established cooperation between the companies , involves the transfer of certain engineering and documentation functions from Larox to Etteplan .
     - ["positive", "neutral", "negative"]
     - positive
     - 0
   * - fpb3877
     - Analyze the sentiment of this statement extracted from a financial news article. Provide your answer as either negative, positive, or neutral.  
       Text: ( ADP News ) - Finnish handling systems provider Cargotec Oyj ( HEL : CGCBV ) announced on Friday it won orders worth EUR 10 million ( USD 13.2 m ) to deliver linkspans to Jordan , Morocco and Ireland .  
       Answer:
     - ( ADP News ) - Finnish handling systems provider Cargotec Oyj ( HEL : CGCBV ) announced on Friday it won orders worth EUR 10 million ( USD 13.2 m ) to deliver linkspans to Jordan , Morocco and Ireland .
     - ["positive", "neutral", "negative"]
     - positive
     - 0

**FiQA SA Sentiment Examples**

.. list-table:: FiQA SA Examples
   :header-rows: 1
   :widths: 10 40 30 40 10 10

   * - **ID**
     - **Query**
     - **Text**
     - **Choices**
     - **Answer**
     - **Gold**
   * - fiqasa938
     - What is the sentiment of the following financial post: Positive, Negative, or Neutral?  
       Text: $BBRY Actually lost .03c per share if U incl VZ as no debt and 3.1 in Cash.  
       Answer:
     - $BBRY Actually lost .03c per share if U incl VZ as no debt and 3.1 in Cash.
     - ["negative", "positive", "neutral"]
     - negative
     - 0
   * - fiqasa939
     - What is the sentiment of the following financial headline: Positive, Negative, or Neutral?  
       Text: Legal & General share price: Finance chief to step down  
       Answer:
     - Legal & General share price: Finance chief to step down
     - ["negative", "positive", "neutral"]
     - negative
     - 0

**Headline Examples**

.. list-table:: Headline Examples
   :header-rows: 1
   :widths: 40 20 40

   * - **Input**
     - **Output**
     - **Instruction**
   * - Trading partners unlikely to take India's gold import curbs lightly
     - No
     - Examine the news headline and decide if it includes price.  
       Options: No, Yes
   * - Trading partners unlikely to take India's gold import curbs lightly
     - No
     - Let me know if the news headline talks about price going up.  
       Options: Yes, No

**NER Examples**

.. list-table:: Named Entity Recognition (NER) Examples
   :header-rows: 1
   :widths: 40 10 40

   * - **Input**
     - **Output**
     - **Instruction**
   * - Subordinated Loan Agreement - Silicium de Provence SAS and Evergreen Solar Inc . 7 - December 2007 [ HERBERT SMITH LOGO ] ................................ 2007 SILICIUM DE PROVENCE SAS and EVERGREEN SOLAR , INC .
     - organization
     - What is the entity type of 'EVERGREEN SOLAR' in the input sentence.  
       Options: location, organization, person
   * - Subordinated Loan Agreement - Silicium de Provence SAS and Evergreen Solar Inc . 7 - December 2007 [ HERBERT SMITH LOGO ] ................................ 2007 SILICIUM DE PROVENCE SAS and EVERGREEN SOLAR , INC .
     - organization
     - With the input text as context, identify the entity type of 'EVERGREEN SOLAR'.  
       Options: person, organization, location

Please view `FinLoRA Documentation General Financial Tasks Page <https://finlora-docs.readthedocs.io/en/latest/tasks/general_financial_tasks.html>`__ for more examples and details.


XBRL Dataset
------------------------------------------------------------

XBRL (eXtensible Business Reporting Language) is a standard for electronic communication of business and financial data. It is widely used in regulatory filings, such as SEC filings. The XBRL dataset contains tasks related to XBRL tag extraction, value extraction, formula construction, calculation, numeric identification, and concept linking. These tasks are designed to evaluate the ability of FinGPT agents to understand and analyze XBRL data.

Question Set Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 10 15 40 60

   * - **Dataset**
     - **Size**
     - **Metrics**
     - **Dataset Link**
     - **Description**
   * - XBRL tag extraction
     - 100
     - Accuracy
     - `Link <https://huggingface.co/datasets/wangd12/XBRL_analysis>`__
     - Extract a specific XBRL tag from a raw XBRL text segment given a natural language description of the tag.
   * - XBRL value extraction
     - 100
     - Accuracy
     - `Link <https://huggingface.co/datasets/wangd12/XBRL_analysis>`__
     - Extract a numeric value from the raw XBRL text segment based on a natural language description.
   * - XBRL formula construction
     - 100
     - Accuracy
     - `Link <https://huggingface.co/datasets/wangd12/XBRL_analysis>`__
     - Identify relevant facts and their tags to construct a standard financial formula.
   * - XBRL formula calculation
     - 100
     - Accuracy
     - `Link <https://huggingface.co/datasets/wangd12/XBRL_analysis>`__
     - Substitute numeric values into a constructed formula and compute the result.
   * - Financial Numeric Identification (FinNI) [8]
     - 200
     - F1
     - `Link <https://github.com/The-FinAI/FinTagging/tree/main/subdata>`__
     - Identify financial values in documents and assign a coarse-grained data type: Fact and Type components of a {Fact, Type, Tag} triplet.
   * - Financial Concept Linking (FinCL) [8]
     - 500
     - Accuracy
     - `Link <https://github.com/The-FinAI/FinTagging/tree/main/subdata>`__
     - Link each identified value to a financial taxonomy concept: the Tag component of the {Fact, Type, Tag} triplet.

For all question sets, we sample some questions from the test split. For FinNI and FinCL tasks, participants can use the `training resource <https://github.com/The-FinAI/FinTagging/blob/main/annotation/TrainingSet_Annotation.json>`__.

Question Set Examples
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
For the first four tasks, please view `FinLoRA Documentation Financial Statement Analysis page <https://finlora-docs.readthedocs.io/en/latest/tasks/xbrl_analysis_tasks.html>`__ for examples and details.

**FinNI Example**

**FinCL Example**



FinLoRA
=====================================

FinLoRA [1]_ provides detailed `documentation <https://finlora-docs.readthedocs.io/en/latest/>`__ and reference implementations for fine-tuning open LLMs on financial tasks with Low-Rank Adaptation (LoRA). Participants may find its guidelines and codebase helpful for reproducing the strong baselines reported below.

Baseline Performance
=====================================

.. list-table:: Baseline CFA Exam Performance (accuracy %)
   :header-rows: 1
   :widths: 20 12 12

   * - **Model**
     - **Level I**
     - **Level II**
   * - GPT-4o
     - 88.1
     - 76.7
   * - GPT-4
     - 74.6
     - 61.4
   * - GPT-3.5 Turbo
     - 63.0
     - 47.6

.. list-table:: Baseline XBRL Task Performance (higher is better)
   :header-rows: 1
   :widths: 20 12 12 12 12

   * - **Model**
     - **Tag Extraction**
     - **Value Extraction**
     - **Formula Construction**
     - **Formula Calculation**
   * - GPT-4o
     - 81.60
     - 97.01
     - 79.76
     - 83.59
   * - DeepSeek V3
     - 85.03
     - 98.01
     - 22.75
     - 85.99
   * - Llama 3 70B
     - 69.64
     - 88.19
     - 59.28
     - 77.49
   * - Gemini 2.0 FL
     - 80.27
     - 98.02
     - 61.90
     - 53.57

References
=====================================

.. [1] Wang, D., Patel, J., Zha, D., Yang, S. Y., & Liu, X. Y. (2025). *FinLoRA: Benchmarking LoRA Methods for Fine-Tuning LLMs on Financial Datasets*. arXiv:2505.19819.
.. [2] Shijie Wu, Ozan Irsoy, Steven Lu, Vadim Dabravolski, Mark Dredze, Sebastian Gehrmann, Prabhanjan Kambadur, David Rosenberg, Gideon Mann. *BloombergGPT: A Large Language Model for Finance*. arXiv:2303.17564 (2023).
.. [3] Malo, P., Lu, H., Ahlgren, M., Rönnqvist, S., & Nyberg, P. (2014). *FinancialPhraseBank-v1.0*. SSRN 2512146.  https://ssrn.com/abstract=2512146
.. [4] Sinha, A., Joglekar, M., & Murphy, F. (2018). *FiQA: Financial Opinion Mining and Question Answering*. arXiv:1809.09431.
.. [5] FinGPT. (2023). *FinGPT Headline Classification* (HuggingFace Dataset). https://huggingface.co/datasets/FinGPT/fingpt-headline-cls
.. [6] FinGPT. (2023). *FinGPT Named Entity Recognition* (HuggingFace Dataset). https://huggingface.co/datasets/FinGPT/fingpt-ner-cls
.. [7] Chen, Z., Li, S., Smiley, C., Ma, Z., Shah, S., & Wang, W. Y. (2022). *ConvFinQA: Exploring the Chain of Numerical Reasoning in Conversational Finance Question Answering*. arXiv:2210.03849.
.. [8] Wang, Y., Ren, Y., Qian, L., Peng, X., Wang, K., Han, Y., ... & Xie, Q. (2025). *FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information*. arXiv:2505.20650.

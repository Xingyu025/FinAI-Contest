=====================================================
Task 1 FinRL-DeepSeek for Crypto Trading
=====================================================

Task Overview
=================
This task is to develop crypto trading agents by integrating LLM-engineered signals in FinRL, using financial news and market data. Participants will explore the design and use of LLM-engineered signals, such as sentiment scores and risk levels extracted from financial news, to enhance trading strategies in the FinRL framework. The task encourages participants to innovate with these signals to improve trading performance in the volatile Bitcoin markets.


Why This Matters
=================
In the volatile cryptocurrency markets, timely interpretation of market sentiment is critical. Cryptocurrency markets are highly sensitive to news headlines, tweets, regulatory shifts, and viral narratives. However, the massive amount of news, social media posts, and documents can overwhelm individual traders. LLMs have shown strong capabilities in understanding and generating financial text, making them valuable tools for extractiong actionable signals from news, social media posts, and regulatory filings. In this task, we encourage participants to explore FinGPT-engineered signals and integrate them into a FinRL trading agent for crypto trading.

Starter Kit
=================
In the starter kit, we provide a basic implementation of a FinRL Bitcoin trading agent that integrates LLM-generated signals. Participants can use this as a foundation to build their own trading strategies and enhance them with additional LLM-generated signals.

Data
----------------
We provide the second-level Limit Order Book (LOB) data and Bitcoin (BTC) nwes for BTC trading.

Limit Order Book (LOB) Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
We provide the second level LOB data of BTC, which is sourced form `Kaggle <https://www.kaggle.com/datasets/martinsn/high-frequency-crypto-limit-order-book-data?resource=download&select=BTC_1sec.csv>`_. Some of important columns represent:

.. list-table:: LOB Data Columns Description
   :header-rows: 1

   * - Column
     - Description

   * - ``system_time``
     - Timestamp of the order book snapshot in UTC.

   * - ``midpoint``
     - Mid-price between best bid and best ask: midpoint = (best bid + best ask)/2

   * - ``spread``
     - Bid-ask spread: spread = best ask - best bid}

   * - ``bids_distance_0`` to ``bids_distance_14``
     - Distance (in %) of bid level *x* from the midpoint:  

   * - ``asks_distance_0`` to ``asks_distance_14``
     - Distance (in %) of ask level *x* from the midpoint:  

   * - ``bids_notional_0`` to ``bids_notional_14``
     - Notional volumes of active bid orders at each depth level.

   * - ``asks_notional_0`` to ``asks_notional_14``
     - Notional volumes of active ask orders at each depth level.

   * - ``bids_market_notional_0`` to ``bids_market_notional_14``
     - Volume of market sell orders matched with bid level *x*.

   * - ``asks_market_notional_0`` to ``asks_market_notional_14``
     - Volume of market buy orders matched with ask level *x*.

   * - ``bids_limit_notional_0`` to ``bids_limit_notional_14``
     - Volume of new limit buy orders placed at bid level *x*.

   * - ``asks_limit_notional_0`` to ``asks_limit_notional_14``
     - Volume of new limit sell orders placed at ask level *x*.

News Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We also provide news data from BTC from multiple sources [`1 <https://huggingface.co/datasets/edaschau/bitcoin_news/viewer?views%5B%5D=train>`_, `2 <https://github.com/soheilrahsaz/cryptoNewsDataset>`_] and aggregated them into a single dataset (684 pieces news in total). The news data includes the following columns:

.. list-table:: News Dataset Column Definitions
   :header-rows: 1

   * - Column
     - Description

   * - ``date_time``
     - Timestamp when the news article was published.

   * - ``title``
     - Headline or title of the news article.

   * - ``url``
     - Full URL link to the original news article.

   * - ``source``
     - Name of the news outlet or publishing organization (e.g. Reuters, Bloomberg, CoinDesk).

   * - ``article_text``
     - Full body text of the article, typically extracted from the web page and cleaned for analysis.


Stage 1: LLM-generated signals
-------------------------------------
For each news, we use DeepSeek chat model to extract sentiment score, risk level and their correpsonding confidence level and one-sentence reasoning. 

We use the following prompt:
1. Sentiment scores. LLM assigns a sentiment score of 1 to 5 according to the news, with 1 for highly negative and 5 for highly positive.

.. raw:: html

   <div style="background-color:#f0f0f0; padding:12px; border-radius:6px; line-height:1.6;">
     <p><strong>Sentiment Score Prompt:</strong></p>

     <p>You are a financial news analyst. Your task is to analyze the sentiment of the following news article.</p>

     <p>You must provide your analysis in a structured JSON format. The JSON object must contain the following keys:</p>

     <ul>
       <li><code>"sentiment_score"</code>: An integer from 1 (extremely negative) to 5 (extremely positive), where 3 means neutral.</li>
       <li><code>"confidence_score_sentiment"</code>: A float between 0.0 and 1.0, representing your confidence in the sentiment analysis.</li>
       <li><code>"reasoning_sentiment"</code>: A concise, one-sentence explanation for your sentiment score.</li>
     </ul>

     <p>Here is a perfect example of the output format:</p>

     <div style="background-color:#f0f0f0; padding:0px; font-family:monospace; white-space:pre-wrap;">
        {
            "sentiment_score": 4,
            "confidence_score_sentiment": 0.95,
            "reasoning_sentiment": "The article reports a significant earnings beat and a positive future outlook, which are strong bullish signals."
        }
     </div>

     <p>Now, analyze the following news item and provide <strong>ONLY</strong> the JSON object as your response.</p>

     <p><strong>Title:</strong> {title}</p>
     <p><strong>Article Text:</strong> {text}</p>
   </div>

2. Risk levels. LLM assigns a risk level of 1 to 5 from the news, with 1 for very low risk and 5 very high risk.

.. raw:: html

   <div style="background-color:#f0f0f0; padding:12px; border-radius:6px;">
     <p><strong>Risk Level Prompt:</strong></p>
     
     <p>You are a professional cryptocurrency risk analyst. Your task is to analyze the following news article to identify potential risks related to Bitcoin (BTC) or the broader crypto market.</p>

     <p>You must provide your analysis in a structured JSON format. The JSON object must contain the following keys:</p>
     <ul>
       <li>"risk_score": An integer from 1 (extremely negative) to 5 (extremely positive), where 3 means neutral.</li>
       <li>"confidence_score_risk": A float between 0.0 and 1.0, representing your confidence in the risk analysis.</li>
       <li>"reasoning_risk": A concise, one-sentence explanation for your risk assessment.</li>
     </ul>
     <p>Here is a perfect example of the output format for a BTC-related article:</p>

      <div style="background-color:#f0f0f0; padding:0px; font-family:monospace; white-space:pre-wrap;">
        {
            "risk_score": 4,
            "confidence_score_risk": 0.85,
            "reasoning_risk": "The announcement of new government regulations..."
        }
     </div>

     <p>Now, analyze the following news item and provide <strong>ONLY</strong> the JSON object as your response.</p>

     <p><strong>Title:</strong> {title}<br>
     <strong>Article Text:</strong> {text}</p>
   </div>

After getting the signals, confidence scores and reasoning, we add them to the news dataset. The final news dataset looks like this:

.. list-table:: News Dataset with Sentiment and Risk Signals
   :header-rows: 1
   :widths: 12 20 15 10 30 5 10 20 5 10 20

   * - ``date_time``
     - ``title``
     - ``url``
     - ``source``
     - ``article_text``
     - ``sentiment_score``
     - ``confidence_score_sentiment``
     - ``reasoning_sentiment``
     - ``risk_score``
     - ``confidence_score_risk``
     - ``reasoning_risk``

   * - 2021-04-07 12:00:00+00:00
     - 8 Smart Ways to Analyze Crypto Token Before Investing in It
     - https://finance.yahoo.com/news/8-smart-ways-analyze-crypto-200000331.html
     - Entrepreneur
     - The world of cryptocurrencies is vast ...
     - 3
     - 0.85
     - The article provides a balanced view on analyzing crypto tokens, highlighting both opportunities and risks without leaning heavily towards positive or negative sentiment.
     - 2
     - 0.75
     - The article highlights the prevalence of scams and the challenge of identifying legitimate investments in the vast and unregulated crypto market, indicating a high risk for investors.

   * - 2021-04-07 12:00:00+00:00
     - Coinme Launches 300 Bitcoin-Enabled Coinstar Kiosks in Florida
     - https://finance.yahoo.com/news/coinme-launches-300-bitcoin-enabled-120000323.html
     - GlobeNewswire
     - Floridians can now conveniently buy bitcoin ...
     - 4
     - 0.90
     - The article highlights the expansion of accessible bitcoin purchasing options in Florida, indicating positive growth and adoption of cryptocurrency.
     - 4
     - 0.75
     - The expansion of bitcoin-enabled kiosks in Florida increases accessibility and convenience for purchasing bitcoin, potentially boosting adoption and positive market sentiment.

   * - 2021-04-07 12:43:12+00:00
     - Alcoa, Anheueser-Busch InBev, Rent-A-Center, Target and Walmart ...
     - https://finance.yahoo.com/news/alcoa-anheueser-busch-inbev-rent-124312449.html
     - Zacks
     - Chicago, IL – April 7, 2021 – Zacks Equity Research highlights ...
     - 4
     - 0.85
     - The article highlights strong bullish signals for several stocks, including Alcoa and Rent-A-Center, with positive earnings estimates and strategic moves, despite mentioning a bearish outlook for Anheueser-Busch InBev.
     - 3
     - 0.50
     - The article does not directly mention Bitcoin or the broader crypto market, making its impact on cryptocurrency risks neutral and uncertain.

After getting the sentiment and risk signals, we need to combine these signals with LOB data in a time-aligned manner. We should handle two issues:

    1. **Multiple news at the same time stamp**. For example, there are multiple news at 2021-04-07 12:00:00+00:00. We directly average the sentiment and risk scores of these news. A more advanced method is to use a weighted average based on the confidence scores.
    2. **LOB data is at a higher frequency than news data**. To align the two sources, we propagate each news-derived signal to all LOB timestamps after the news timestamp, up to (but not including) the timestamp of the next news event. This ensures that the sentiment and risk signals extracted from news articles are applied consistently to the corresponding trading intervals until new information becomes available.

.. code-block:: python

   import pandas as pd
   # Load the LOB data and news data
   lob_df = pd.read_csv('lob_data.csv')
   news_df = pd.read_csv('news_data.csv')

   # Convert the news timestamp column to timezone-aware datetime
   news_df['date_time'] = pd.to_datetime(news_df['date_time'], utc=True)

   # Extract only the relevant columns: timestamp and signals
   signals = news_df[['date_time', 'sentiment_score', 'risk_score']]

   # Aggregate signals for identical timestamps by taking the mean directly
   signals_agg_df = signals.groupby('date_time').mean().reset_index()

   # Sort both datasets by their respective time columns (required for merge_asof)
   lob_df = lob_df.sort_values('system_time')
   signals_agg_df = signals_agg_df.sort_values('date_time')

   # Merge two datasets
   # The 'direction="backward"' parameter applies the last known signal forward in time.
   merged_df = pd.merge_asof(
       left=price_df,
       right=signals_agg_df,
       left_on='system_time',
       right_on='date_time',
       direction='backward'
   )

   # Drop the redundant 'date_time' column (now aligned with 'system_time')
   merged_df = merged_df.drop(columns=['date_time'])

Stage 2: Factor Mining 
----------------------------------------

In this stage, we will use the supervised training of deep learning recurrent networks to extract strong factors from the LOB datasets. The goal is to derive strong predictive factors that can be used as features for reinforcement learning agents in the crypto trading task. First, we use the LOB data to derive 101 weak alpha signals. Second, we train a recurrent neural network (RNN) model to process these weak signals and generate strong factors. The strong factors will then be used as features for the reinforcement learning agents.


**What is Alpha 101?**
  Alpha 101 [1]_ refers to a set of 101 quantitative trading signals or features, known as alpha factors. These signals are designed to capture various predictive relationships between market data (e.g., price, volume) and future returns. These alphas leverage features derived from historical price and volume data to identify profitable trading opportunities.
  
  In this task, factor mining involves deriving meaningful financial indicators (i.e., Alpha 101) from limit order book (LOB) data and using techniques like recurrent neural networks (RNNs) to generate strong predictive signals.

**Architecture**

The model combines multiple components:


1. **Input Projection (MLP Encoder)**  
   The input is 101 weak alpha signals . These features are passed through a feedforward encoder — a multi-layer perceptron (MLP) — that maps the raw input into a hidden representation.

2. **Dual Recurrent Branches (LSTM + GRU)**  
   The encoded sequence is processed in parallel by two recurrent networks:
   
   - An **LSTM** (Long Short-Term Memory) with ``num_layers = 4`` layers.
   - A **GRU** (Gated Recurrent Unit) with the same configuration.
   
   Using both LSTM and GRU allows the model to capture diverse temporal patterns and long-term dependencies across the input sequence. The depth of 4 layers enables the model to learn complex hierarchical patterns in the data.

3. **Refinement MLPs**  
   The output of each recurrent branch ia passed through a separate MLP with GELU activations. These layers nonlinearly transform and refine the temporal features produced by the RNNs.

4. **Concatenation and Output Projection**  
   The refined outputs from the LSTM and GRU branches are concatenated along the feature dimension. This is passed through a final MLP that projects the combined features into a lower-dimensional output space (e.g., 8 dimensions), using a Tanh activation to produce bounded outputs.

5. **Output**  
   The model outputs a sequence of strong predictive factors that can be used as features in the state in FinRL.

The following table lists the key hyperparameters used in training this model, along with their default values and descriptions.

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Hyperparameter
     - Value
     - Description

   * - ``batch_size``
     - 256
     - Number of samples processed in each training batch.

   * - ``mid_dim``
     - 128
     - Dimensions of hidden layers in recurrent networks.

   * - ``num_layers``
     - 4
     - The number of layers in the recurrent network. The larger the value, the more content the recurrent network can remember.


   * - ``epoch``
     - 256
     - Total number of training epochs.

   * - ``wup_dim``
     - 64
     - The length of the sequence used for pre-warming of the recurrent network. The output loss will not be calculated during the pre-warming phase. The pre-warming phase is only used to obtain the hidden state of the recurrent network.

   * - ``weight_decay``
     - 1e-4
     - Weight decay is used to control the strength of the regularization term to prevent overfitting.

   * - ``learning_rate``
     - 1e-3
     - Learning rate, controls the step size of parameter update at each iteration.

   * - ``clip_grad_norm``
     - 2
     - Gradient clipping threshold, used to control the size of the gradient and prevent gradient explosion problems.


Stage 3: FinRL for Crypto Trading
-------------------------------------------------------

**RL Setting for Crypto Trading**
  The crypto trading task is modeled as a Markov Decision Process (MDP), where:
  
  * **State space** includes position (number of shares/contracts held), holding (how long you've held the current position), 8 strong facors (obtained from RNN model), 2 LLM-engineered signals (sentiment score and risk score).
  * **Action space** consists of discrete trade actions (buy, sell, hold) determined by DQN-based models.
  * **Reward function** is the change of asset value.
  
  FinRL agents learn to optimize trading strategies by interacting with a simulated market environment.

**Parallel Environment**
  The parallel environment consists of thousands of simulated trading environments running concurrently on GPUs.
  
  This approach addresses the sampling bottleneck and improves sampling speed, making it feasible to train multiple trading agents in a short period.
  
**FinRL Agents**
  We use DQN-based agents for the crypto trading task, including D3QN, Double DQN, and Twin D3QN. The DQN architecture is designed to handle discrete action spaces, which is well-suited for high-frequency trading tasks. The agents are trained using the parallel market environment, which allows them to learn from a large number of simulated trades and adapt to changing market conditions.
  
  We use the majority voting ensemble method to combine the actions of multiple agents. This approach improves robustness and reduces the risk of overfitting to specific market conditions. The ensemble method allows us to leverage the strengths of different agents and improve overall trading performance.


Evaluation
----------------
Models will be evaluated based on:

* **Cumulative return**. It is the total return generated by the trading strategy over a trading period.
* **Sharpe ratio**. It takes into account both the returns of the portfolio and the level of risk.
* **Max drawdown**. It is the portfolio’s largest percentage drop from a peak to a trough in a certain time period, which provides a measure of downside risk.




.. [1] Zura Kakushadze. 101 Formulaic Alphas. arXiv preprint arXiv:1601.00991 (2016).
=============================
Market Data
=============================
We provide a variety of market data for FinRL Contests, including:
    
    - **OHLCV data**: daily open, high, low, close, volume data for stocks.
    - **Limit Order Book (LOB) data**: second-level LOB data for Bitcoin.
    - **Financial news data**: financial news for different stocks.

OHLCV data
---------------------------
For stock trading tasks, we use daily OHLCV data, a rich source for learning financial market behaviors and trends. It’s a list of five most common types of data in financial analysis: 

    - **Open**: the price at which the stock opened for trading on a trading day.
    - **High**: the highest price at which the stock traded during a trading day.
    - **Low**: the lowest price at which the stock traded during a trading day.
    - **Close**: the price at which the stock closed at the end of a trading day.
    - **Volume**: the total number of shares traded during a trading day.

To download the OHLCV data, we can use `yfinance`. The data is stored in a CSV file, which can be easily read and processed.

.. code-block:: python

    import yfinance as yf
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    start_date = '2020-01-01'
    end_date = '2025-01-01'
    data = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        interval='1d',  # Daily data
    )
    data.to_csv('ohlcv_data.csv')


Limit Order Book (LOB) data
---------------------------




Financial news data - FNSPID [1]_
---------------------------
The `Financial News and Stock Price Integration Dataset (FNSPID) <https://huggingface.co/datasets/Zihan1004/FNSPID>`_ is a comprehensive financial news dataset. It contains historical stock prices and over 15 million time-aligned financial news articles related to Nasdaq-listed companies, spanning from 1999 to 2023. 

Here is a preview of the dataset:

.. list-table::
   :header-rows: 1
   :widths: 15 30 10 50 20 10 10 10 10 10 10

   * - Date
     - Article_title
     - Stock_symbol
     - Url
     - Publisher
     - Author
     - Article
     - Lsa_summary
     - Luhn_summary
     - Textrank_summary
     - Lexrank_summary
   * - 2020-06-05 06:30:54 UTC
     - Stocks That Hit 52-Week Highs On Friday
     - A
     - https://www.benzinga.com/news/20/06/16190091/stocks-that-hit-52-week-highs-on-friday
     - Benzinga Insights
     - null
     - null
     - null
     - null
     - null
     - null
   * - 2020-06-03 06:45:20 UTC
     - Stocks That Hit 52-Week Highs On Wednesday
     - A
     - https://www.benzinga.com/news/20/06/16170189/stocks-that-hit-52-week-highs-on-wednesday
     - Benzinga Insights
     - null
     - null
     - null
     - null
     - null
     - null
   * - 2020-05-26 00:30:07 UTC
     - 71 Biggest Movers From Friday
     - A
     - https://www.benzinga.com/news/20/05/16103463/71-biggest-movers-from-friday
     - Lisa Levin
     - null
     - null
     - null
     - null
     - null
     - null


Feature Engineering
---------------------------

.. _technical-indicators:

Feature Engineering
~~~~~~~~~~~~~~~~~~~~~
We also provide 10 market indicators, which are explained below.

.. list-table::
   :header-rows: 1
   :widths: 10 20 50

   * - Indicator
     - Name
     - Description
   * - macd
     - Moving Average Convergence Divergence
     - A trend-following momentum indicator that shows the relationship between two exponential moving averages (EMAs) of a stock’s price. Traders use the MACD to identify potential trend changes, divergence between price and momentum, and overbought or oversold conditions.
   * - boll_ub
     - Bollinger Bands Upper Band
     - Bollinger Bands are used to visualize the volatility and potential price levels of a stock. The upper band represents the upper volatility boundary, showing where the price might find resistance.
   * - boll_lb
     - Bollinger Bands Lower Band
     - The lower band represents the lower volatility boundary and shows where the price might find support.
   * - rsi_30
     - Relative Strength Index for 30 periods
     - A momentum oscillator that measures the speed and change of price movements. RSI oscillates between zero and 100.
   * - cci_30
     - Commodity Channel Index for 30 periods
     - A versatile indicator that can be used to identify a new trend or warn of extreme conditions. It measures the current price level relative to an average price level over a given period of time.
   * - dx_30
     - Directional Movement Index for 30 periods
     - An indicator that assesses the strength and direction of a trend by comparing highs and lows over time.
   * - close_30
     - 30-Period Simple Moving Average of Closing Prices
     - Represents the average closing price over the last 30 periods. This moving average smooths price data to help identify trends and potential support/resistance levels.
   * - close_60
     - 60-Period Simple Moving Average of Closing Prices
     - Represents the average closing price over the last 60 periods. Like the 30-period average, it helps visualize longer-term trends and support/resistance zones.
   * - vix
     - Volatility Index
     - Often referred to as the "fear index", it represents the market's expectation of 30-day forward-looking volatility. Calculated from prices of selected stock option contracts on the S&P 500 Index.
   * - turbulance
     - Turbulence
     - A risk control measure used in FinRL that quantifies extreme asset price fluctuations, useful for handling worst-case market scenarios like the 2007–2008 financial crisis.

Factor mining
~~~~~~~~~~~~~~~~~~~~~


LLM-Generated Signals
~~~~~~~~~~~~~~~~~~~~~
We can leverage LLMs to extract signals from the news articles. These signals can be used to enhance the trading strategy and improve the performance of the trading agent. The LLM-generated signals include sentiment scores, risk levels, and other relevant information extracted from the financial news.

1. Sentiment scores. LLM assigns a sentiment score of 1 to 5 according to the news, with 1 for negative and 5 for positive.

.. raw:: html

   <div style="background-color:#f0f0f0; padding:12px; border-radius:6px;">
     <strong>Prompt:</strong><br>
       You are a financial expert with sentiment analysis and stock recommendation experience. Based on a specific stock, score for range from 1 to 5, where 1 is negative, 2 is somewhat negative, 3 is neutral, 4 is somewhat positive, 5 is positive
   </div>

2. Risk levels. LLM assigns a risk level of 1 to 5 from the news, with 1 for very low risk and 5 very high risk.

.. raw:: html

   <div style="background-color:#f0f0f0; padding:12px; border-radius:6px;">
     <strong>Prompt:</strong><br>
       You are a financial expert specializing in risk assessment. Based on a specific stock, provide a risk score from 1 to 5, where: 1 indicates very low risk, 2 indicates low risk, 3 indicates moderate risk (default if the news lacks any clear indication of risk), 4 indicates high risk, and 5 indicates very high risk.
   </div>


**References**

.. [1] Zihan Dong, Xinyu Fan, and Zhiyuan Peng. 2024. FNSPID: A Comprehensive Financial News Dataset in Time Series. arXiv preprint arXiv:2402.06698 (2024).
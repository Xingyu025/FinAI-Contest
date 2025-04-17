=============================
Market Data
=============================
We provide a variety of market data for FinRL Contests, including:
    
    - **OHLCV data**: daily open, high, low, close, volume data for stocks.
    - **Limit Order Book (LOB) data**: second-level LOB data for Bitcoin.
    - **Financial news data**: financial news for different stocks.

OHLCV data
---------------------------
OHLCV
~~~~~~~~~~~~~~~~~
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

Limit Order Book (LOB) data
---------------------------


LOB data
~~~~~~~~~~~~~~~~~

Factor Mining
~~~~~~~~~~~~~~~~~


Financial news data
---------------------------
FNSPID [1]_
~~~~~~~~~~~~~~~~~

LLM-Generated Signals
~~~~~~~~~~~~~~~~~~~~~



**References**

.. [1] Zihan Dong, Xinyu Fan, and Zhiyuan Peng. 2024. FNSPID: A Comprehensive Financial News Dataset in Time Series. arXiv preprint arXiv:2402.06698 (2024).
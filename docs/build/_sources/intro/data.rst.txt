=============================
Market Data
=============================
We provide a variety of market data for FinRL Contests, including:
    
    - **OHLCV data**: daily open, high, low, close, volume data for stocks.
    - **Limit Order Book (LOB) data**: second-level LOB data for Bitcoin.
    - **Financial news data**: Financial news for different stocks.

OHLCV data
---------------------------
OHLCV
~~~~~~~~~~~~~~~~~
For stock trading tasks, we use daily OHLCV data, a rich source for learning financial market behaviors and trends. It’s a list of five most common types of data in financial analysis: 

    - **Open**: The price at which the stock opened for trading on a trading day.
    - **High**: The highest price at which the stock traded during a trading day.
    - **Low**: The lowest price at which the stock traded during a trading day.
    - **Close**: The price at which the stock closed at the end of a trading day.
    - **Volume**: The total number of shares traded during a trading day.

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


Feature Engineering
~~~~~~~~~~~~~~~~~~~~~


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
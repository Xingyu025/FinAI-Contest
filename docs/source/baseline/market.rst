=================================================
Market Index and Conventional Trading Strategies
=================================================

We use market indexes and conventional trading strategies as baselines in the FinAI contests. They are widely adopted in the financial industry.


Market Indexes
------------------

Stock markets:

- **DJIA**: The Dow Jones Industrial Average index is a price-weighted index for 30 blue-chip U.S. companies. It is calculated as the sum of constituent stock prices divided by the Dow Divisor. The divisor is a constant adjusted for stock splits and structural changes. The data can be downloaded using ``yfinance``. DJIA is one of the oldest and most recognized market indexes. It provides a real-world baseline to evaluate whether a FinRL agent outperforms a passive investment strategy.

- **S&P 500**: The Standard and Poor's 500 is a capitalization-weighted index tracking the stock performance of 500 leading companies. The data can be downloaded using ``yfinance``. Covering approximately 80% of the total market capitalization, it offers a broader and more diversified baseline.

- **Nasdaq-100**: The Nasdaq-100 is a stock market index consists of 100 of the largest non-financial companies listed on the Nasdaq stock exchange. It is a capitalization-weighted index. The NASDAQ-100 is heavily concentrated in the technology sector but also includes firms from industries such as consumer discretionary, healthcare, communication services, and industrials. Different from the S&P 500, the NASDAQ-100 excludes financial companies such as banks and insurance firms. Different from the DJIA, it offers broader exposure and follows a market cap–weighted methodology instead of price weighting.

Cryptocurrency markets:

- **CoinDesk Market Index (CMI)**

- **Nasdaq Crypto Index (NCI)**


Conventional Trading Strategies
---------------------------------

- **Mean-Variance Optimization**: Mean-variance optimization, as part of Modern Portfolio Theory (MPT) [Markowitz1952]_, constructs portfolios that maximize the expected return for a given level of risk. It uses expected asset returns and covariances to solve an optimization problem. We typically use the past one year's daily price data to calculate expected returns and the covariance matrix. We limit individual stock weights to a maximum of 5%. Mean-variance optimization is a foundational technique in portfolio management and can serve as a classical financial optimization strategy baseline.
- **Minimum-Variance Optimization**
- **Equally Weighted Portfolio**
- **Buy and Hold**





.. [Markowitz1952] Harry Markowitz. "Portfolio Selection," *Journal of Finance*, 1952.

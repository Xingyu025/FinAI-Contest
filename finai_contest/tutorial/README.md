# Standard Market Environments for Reinforcement Learning
In this project, we organize current open-source market environments, and evaluated them in standard evaluation pipelines. Currently, we are evaluating stock and crypto market environments. Here is a tutorial to get you started and familiar with reproducing and evaluating market environments!

## Stock Market Evaluation Pipeline Setting
### Data

For evaluation of stock market environments, we use yahooDownloader from FinRL, and achieve daily data from YahooFinance.
https://finrl.readthedocs.io/en/latest/finrl_meta/Data_layer.html

### Performance metrics

- Cumulative Reward: Percentage gain over the entire trading period
- Annualized Return: Average annual rate of return
- Annualized Volatility: How much return fluctuates annually
- Sharpe Ratio: Risk-adjusted return
- Maximum Drawdown: Maximum loss experienced during the trading period

Definition can be found in the FinRL documentation
https://finrl.readthedocs.io/en/latest/finrl_meta/Benchmark.html

### Baselines
- Dow Jones Industrial Average (DJIA) index: Stock market index for 30 prominent US companies on stock exchange
- Mean-variance strategy: Balances low risk and high reward
- Equal-weight strategy: Equal weight assigned to each asset.

Definition can be found in the FinRL documentation
https://finrl.readthedocs.io/en/latest/finrl_meta/Benchmark.html

### Steps to organize and evaluate an environment
#### 1. Run Orginal Code
run provided example tests in the project, plug in the same data from yahoofinance, and collect the same performance metrics
#### 2. Reproduce in our pipeline
plug in the environment to our pipeline, and collect results. Compare with 1. This serves to ensure the consistency of our reproduction with the original code. 
#### 3. Standardize the environment
Refactor the code for the environment to maintain the same state, action, reward design, but follows the standard format, including method design (name, arguments, returns) and parameter design and naming. The environment should follow the standard Gymnasium-Style. 
## Stock Market Environments

### Gym-anytrading
 https://github.com/AminHP/gym-anytrading

1. Install following installation instructions
2. 







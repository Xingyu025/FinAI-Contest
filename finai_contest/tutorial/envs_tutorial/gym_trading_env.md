# Gym-trading-env

Original Code: https://github.com/ClementPerroud/Gym-Trading-Env/blob/main/src/gym_trading_env/environments.py

To evaluate the stock market environment from gym-trading-env, follow the following steps.

**Table of contents**
- Installation
- Run original code (reference)
- Reproduce in our pipeline
  - `1_Data.ipynb` — data preparation
  - `2_Train.ipynb` — construct and register environment, training notes
  - `3_Backtest.ipynb` — backtesting, prediction and saving results
- Troubleshooting
- Example: list registered TradingEnv IDs at runtime
- References

## Installation
Gym Trading Env supports Python 3.9+ on Windows, Mac, and Linux. You can install it using pip:

```bash
pip install gym-trading-env
```
or using git:
```
git clone https://github.com/ClementPerroud/Gym-Trading-Env
```

Notes:
- Our notebooks use `gymnasium` (a modern fork of `gym`). The upstream project uses `gym`; the APIs are compatible in most places but registration and wrappers can differ.
- If you prefer to keep the original package separate, install upstream into a dedicated environment.

## Run Original Code

1. Initialize
```python
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
df = YahooDownloader(
   start_date="2023-01-01",
   end_date="2024-12-31",
   ticker_list=["^DJI"]
).fetch_data(auto_adjust=False)
```

and
```python
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
TRADE_START_DATE = '2023-01-01'
TRADE_END_DATE = '2024-12-31'
df_diji = YahooDownloader(
   start_date=TRADE_START_DATE, end_date=TRADE_END_DATE, ticker_list=["^DJI"]
).fetch_data()
df_diji
```

Use the same YahooFinance Data

Run 1_Data.ipynb and the data will save into ./data folder in the same directory.

2. Modify training pipeline

The current file runs the training in the new environment. It changes variables to make gym_trading_env scripts usable.

```python
env_used = "gym_trading_env"
```
Setup Environment for Training
```python
from gym_trading_env.environments import TradingEnv
import gymnasium as gym

def reward_function(history):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) #log (p_t / p_t-1 )


env_train = gym.make(
        "TradingEnv",
        name= "BTCUSD",
        df=train,
        windows = 5,
        positions = [ -1, -0.5, 0, 0.5, 1, 1.5, 2], # From -1 (=SHORT), to +1 (=LONG)
        initial_position = 'random', #Initial position
        trading_fees = 0.01/100, # 0.01% per stock buy / sell
        borrow_interest_rate = 0.0003/100, #per timestep (= 1h here)
        reward_function = reward_function,
        portfolio_initial_value = 1000, # in FIAT (here, USD)
        max_episode_duration = 500,
        disable_env_checker = True
    )
```

Modify the environment used in each stage to use the correct datasets.

3. Save training data and prepare for plotting




## Reproduce in our pipeline

We updated three notebooks to integrate the environment into this repo's flow.

### 1_Data.ipynb — Data preparation

The `TradingEnv` expects a time-indexed OHLCV-like DataFrame (or a DataFrame containing a `date`/`Time` column). From the FinRL CSVs we convert to the expected form. Example conversions used in the pipeline:

```python
import pandas as pd

train = pd.read_csv('./data/train_data.csv')
trade = pd.read_csv('./data/trade_data.csv')

# For single-asset demos (AAPL example)
train_aapl = train[train['tic'] == 'AAPL']
trade_aapl = trade[trade['tic'] == 'AAPL']

# Convert to the simplified format used by gym_trading_env
df_train = pd.DataFrame({
    'Time': pd.to_datetime(train_aapl['date']),
    'Price': train_aapl['close'].astype(float),
    'Volume': train_aapl['volume'].astype(float),
})

df_trade = pd.DataFrame({
    'Time': pd.to_datetime(trade_aapl['date']),
    'Price': trade_aapl['close'].astype(float),
    'Volume': trade_aapl['volume'].astype(float),
})
```

Adjust column names as needed (the env may use `Price`, `Ask_price`/`Bid_price`, or `Close` depending on config).

### 2_Train.ipynb — Construct & register environment

We used two patterns during development and training.

1) Direct construction (recommended for fast dev iteration):

```python
from gym_trading_env.environments import TradingEnv
import numpy as np

def reward_function(history):
    return np.log(history['portfolio_valuation', -1] / history['portfolio_valuation', -2])

env_train = TradingEnv(
    name='BTCUSD',
    df=df_train,
    windows=5,
    positions=[-1, -0.5, 0, 0.5, 1, 1.5, 2],
    initial_position='random',
    trading_fees=0.01/100,
    borrow_interest_rate=0.0003/100,
    reward_function=reward_function,
    portfolio_initial_value=1000,
    max_episode_duration=500,
    disable_env_checker=True,
)

# Convert to SB3-compatible VecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
env_train = DummyVecEnv([lambda: env_train])
```

2) Register & use `gym.make` (recommended if you want to `gym.make` by id):

```python
from gymnasium.envs.registration import register

register(
    id='TradingEnv-v0',
    entry_point='gym_trading_env.environments:TradingEnv',
    kwargs={'disable_env_checker': True},
)

env_train = gym.make('TradingEnv-v0', df=df_train, windows=5, ...)
```

Important:
- Do not register the same (unversioned) id multiple times. If `TradingEnv-v0` already exists in your environment (e.g., installed site-packages), registering an unversioned `TradingEnv` will raise a `RegistrationError`.

### 3_Backtest.ipynb — Backtesting, prediction & saving

We implemented helper methods in the gym-trading-env environment file. To adapt functionality in order to plug into our pipeline.

*environments.py*
```python
def get_sb_env(self):
        """
        Wrap this single env in a Stable-Baselines3 DummyVecEnv and return (env, obs).
        """
        from stable_baselines3.common.vec_env import DummyVecEnv
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

```

We also uncommented these methods in stablebaselines3's *model.py* DRL_prediction() helper function.

```python
        account_memory = test_env.env_method(method_name="save_asset_memory")
        actions_memory = test_env.env_method(method_name="save_action_memory")
```

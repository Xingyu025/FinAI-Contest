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
We chose to replicate and adjust the code at: https://github.com/ClementPerroud/Gym-Trading-Env/blob/main/examples/example_environnement.py

1. Import the trained data from YahooFinance

1_Data.ipynb saves three models at the same directory. To align the data, use the following to convert the data to the expected format.

```python
# Import your datas
df = pd.read_csv('../data/train_data.csv', parse_dates=["date"], index_col="date")
df.sort_index(inplace= True)
df.dropna(inplace= True)
df.drop_duplicates(inplace= True)

# Generating features
#WARNING: the column names need to contain keyword 'feature' !
df["feature_close"] = df["close"].pct_change()
df["feature_open"] = df["open"]/df["close"]
df["feature_high"] = df["high"]/df["close"]
df["feature_low"] = df["low"]/df["close"]
df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
df.dropna(inplace= True)
```

2. Create your own reward function

```python
def reward_function(history):
   return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2]) # log(p_t / p_t-1)
```

3. Modify training pipeline
The file runs the training and trading stage with the same dataset, which makes results unsuitable for meaningful comparison with our pipeline. As a solution, we create and adjust the environment created.

```python
env = gym.make(
   "TradingEnv",
   name="BTCUSD",
   df = df,
   windows = 5,
   positions = [ -1, -0.5, 0, 0.5, 1, 1.5, 2], # From -1 (=SHORT), to +1 (=LONG)
   initial_position = 'random', # Initial position
   trading_fees = 0.01/100, # 0.01% per stock buy / sell
   borrow_interest_rate = 0.0003 / 100, # per timestep (= 1h here)
   reward_function = reward_function,
   portfolio_initial_value = 1000, # in FIAT (here, USD)
   max_episode_duration = 500,
   disable_env_checker = True
)

env.add_metric('Position Changes', lambda history : np.sum(np.diff(history['position']) != 0))
env.add_metric('Episode Length', lambda history : len(history['position']))

done, truncated = False, False

observation, info = env.reset()
print(info)

while not done and not truncated:
   action = env.action_space.sample()
   observation, reward, done, truncated, info = env.step(action)
   print(observation)
```

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

First load FinRL + your envs + DRLAgent

```python
import numpy as np
import pandas as pd
from stable_baselines3.common.logger import configure
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.config import INDICATORS, TRAINED_MODEL_DIR, RESULTS_DIR
from finrl.main import check_and_make_directories

import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parents[1]))
from finai_contest.env_stock_trading.env_stock_trading_finrlmeta import StockTradingEnv_FinRLMeta
from finai_contest.env_stock_trading.env_stock_trading_gym_anytrading import StockTradingEnv_gym_anytrading

check_and_make_directories([TRAINED_MODEL_DIR])
```

For making the environment:


1) Add custom helper to gym_trading_env imported files on local machine (environments.py)

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


2) Direct construction (recommended for fast dev iteration):

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
```

We also use FinRL's DRLAgent, but only train PPO. So set *if_using_...* to false other than *if_using_ppo*.

```python

model_ppo = agent.get_model("ppo", model_kwargs=PPO_PARAMS)

if if_using_ppo:
    tmp_path = RESULTS_DIR + "/ppo"
    new_logger_ppo = configure(tmp_path, ["stdout", "csv", "tensorboard"])
    model_ppo.set_logger(new_logger_ppo)

trained_ppo = agent.train_model(
    model=model_ppo,
    tb_log_name="ppo",
    total_timesteps=100000,
) if if_using_ppo else None

```

So: **PPO** + ```gym_trading_env.TradingEnv``` is now the default training flow.

### 3_Backtest.ipynb — Backtesting, prediction & saving

Earlier, we implemented helper methods in the gym-trading-env environment file. To adapt functionality in order to plug into our pipeline.

We also uncommented these methods in stablebaselines3's *model.py* DRL_prediction() helper function to assist with methods used in some cells.
```python
        account_memory = test_env.env_method(method_name="save_asset_memory")
        actions_memory = test_env.env_method(method_name="save_action_memory")
```

1) Loaded the trained PPO model for the ```gym_trading_env``` variant

```python
env_used = "gym_trading_env"

trained_ppo = PPO.load(
    TRAINED_MODEL_DIR + "/agent_ppo_" + env_used,
    device="cpu"
) if if_using_ppo else None
```

2) Recreated the ```TradingEnv``` for the test data
```python
from gym_trading_env.environments import TradingEnv
import gymnasium as gym

def reward_function(history):
    return np.log(history["portfolio_valuation", -1] /
                  history["portfolio_valuation", -2])

e_trade_gym = gym.make(
    "TradingEnv",
    name="BTCUSD",
    df=trade_aapl,          # test / trade subset
    windows=5,
    positions=[-1, -0.5, 0, 0.5, 1, 1.5, 2],
    initial_position="random",
    trading_fees=0.01/100,
    borrow_interest_rate=0.0003/100,
    reward_function=reward_function,
    portfolio_initial_value=1000000,
    max_episode_duration=500,
    disable_env_checker=True,
)
```

3) Run PPO multiple times to estimate mean + std of the equity curve

```python
num_runs = 20

df_account_value_ppo_list = []
df_actions_ppo_list = []

for i in range(num_runs):
    df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
        model=trained_ppo,
        environment=e_trade_gym,
        deterministic=False,
    ) if if_using_ppo else (None, None)

    df_account_value_ppo_list.append(df_account_value_ppo)
    df_actions_ppo_list.append(df_actions_ppo)

df_account_value_all = pd.concat(
    [df.set_index("date")["account_value"] for df in df_account_value_ppo_list],
    axis=1
)
df_account_value_all.columns = [f"run_{i+1}" for i in range(len(df_account_value_ppo_list))]
df_account_value_all["mean"] = df_account_value_all.mean(axis=1)
df_account_value_all["std"] = df_account_value_all.std(axis=1)

df_account_value_ppo = df_account_value_all[["mean", "std"]].reset_index()
df_account_value_ppo.columns = ["date", "account_value", "std"]
```

4) Save the aggregated PPO account values

```python

import os
os.makedirs("./results_csv", exist_ok=True)
df_account_value_ppo.to_csv(
    "./results_csv/account_value_ppo_" + env_used + ".csv",
    index=False,
)
```

5) Compute Sharpe ratio and make a detailed metrics DataFrame

```python
def _calculate_sharpe_ratio(total_profits):
    total_profits = np.array(total_profits)
    daily_returns = np.diff(total_profits) / total_profits[:-1]
    if daily_returns.std() == 0 or len(daily_returns) < 2:
        return 0.0
    sharpe = (255 ** 0.5) * daily_returns.mean() / daily_returns.std()
    return sharpe

print("Sharpe Ratio:", _calculate_sharpe_ratio(df_account_value_ppo["account_value"]))
```

6) Compare results (still finishing)

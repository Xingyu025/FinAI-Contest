# Gym-trading-env
Original Code: https://github.com/ClementPerroud/Gym-Trading-Env/blob/main/src/gym_trading_env/environments.py

To evaluate the stock market environment from gym-trading-env, follow the following tutorial steps:

## Installation
Gym Trading Env supports Python 3.9+ on Windows, Mac, and Linux. You can install it using pip:
# Gym-trading-env

Original Code: https://github.com/ClementPerroud/Gym-Trading-Env/blob/main/src/gym_trading_env/environments.py

This document explains how we integrated and adapted the original `gym_trading_env` into this repository. It follows the same style used for `gym_anytrading.md` and documents the changes made to `1_Data.ipynb`, `2_Train.ipynb`, and `3_Backtest.ipynb`.

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

You can install the upstream project (optional) and the packages we use in the notebooks:

```bash
pip install git+https://github.com/ClementPerroud/Gym-Trading-Env
pip install gymnasium stable-baselines3 pandas numpy
```

Notes:
- Our notebooks use `gymnasium` (a modern fork of `gym`). The upstream project uses `gym`; the APIs are compatible in most places but registration and wrappers can differ.
- If you prefer to keep the original package separate, install upstream into a dedicated environment.

## Run Original Code

The upstream repo includes example scripts demonstrating the `TradingEnv`. We used the original as a reference and then adapted parts to fit FinRL-style training and backtesting.

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

The backtest notebook loads trained agents and runs predictions. `DRLAgent.DRL_prediction` expects an environment object exposing `get_sb_env()` which returns a (vectorized) SB3 env and the initial observation.

Common issues and solutions:

- `gym.make` may return a wrapped environment (e.g., `OrderEnforcing`) that does not expose `get_sb_env()`.
- The solution is to unwrap the wrapper and use the underlying env, or patch the wrapper to expose the method.

Unwrap helper used in the notebook:

```python
def unwrap_env(env):
    try:
        base = env.unwrapped
    except Exception:
        base = env
    while hasattr(base, 'env') and getattr(base, 'env') is not None:
        if hasattr(base, 'get_sb_env'):
            break
        base = base.env
    if not hasattr(base, 'get_sb_env'):
        raise AttributeError('Underlying env does not implement get_sb_env()')
    return base

# Before calling DRL_prediction
base_env = unwrap_env(e_trade_gym)
df_account_value_ppo, df_actions_ppo = DRLAgent.DRL_prediction(
    model=trained_ppo, environment=base_env, deterministic=False
)
```

Monkey-patch alternative (less invasive for quick fix):

```python
base = e_trade_gym.unwrapped
if hasattr(base, 'get_sb_env'):
    e_trade_gym.get_sb_env = base.get_sb_env
```

Saving episode outputs

The notebooks need to extract episode-level histories for plotting and CSV export. We added helper functions attached to the unwrapped env:

```python
def save_asset_memory(self):
    dates = self.df['Time'][self._start_tick:self._current_tick].dt.strftime('%Y-%m-%d')
    profits = self.history.get('total_profit', [])
    return pd.DataFrame({'date': dates, 'account_value': profits})

def save_action_memory(self):
    dates = self.df['Time'][self._start_tick:self._current_tick].dt.strftime('%Y-%m-%d')
    actions = self.history.get('position', [])
    return pd.DataFrame({'date': dates, 'action': actions})

uw = e_trade_gym.unwrapped
uw.save_asset_memory = save_asset_memory.__get__(uw)
uw.save_action_memory = save_action_memory.__get__(uw)
```

## Troubleshooting & Notes

- Registration conflicts: the environment may already be registered by the installed `gym_trading_env` package in `site-packages` (found registrations in `c:\\Users\\Jason\\miniconda3\\Lib\\site-packages\\gym_trading_env\\__init__.py`). If you need a different behavior, register a new versioned id in the notebook (e.g., `TradingEnv-v1`).

- Wrappers: `gym.make` returns wrappers. Use `env.unwrapped` or iterate `.env` to find the base env that contains custom helper methods.

- `DRLAgent.DRL_prediction` expects `get_sb_env()` on the environment; make sure the object you pass satisfies that expectation (either by unwrapping or monkey-patching).

## Example: list registered TradingEnv IDs at runtime

Run this snippet in a notebook cell to see which `TradingEnv*` IDs exist and which module provided the entry point:

```python
import importlib
try:
    import gymnasium as gym
except Exception:
    import gym

registry = getattr(getattr(gym, 'envs', gym), 'registry')
matches = [k for k in sorted(list(registry.keys())) if 'TradingEnv' in k]
print(matches)
for k in matches:
    spec = registry[k]
    print(k, spec.entry_point)
    if spec.entry_point and ':' in spec.entry_point:
        mod = spec.entry_point.split(':', 1)[0]
        try:
            m = importlib.import_module(mod)
            print('module file:', getattr(m, '__file__', 'n/a'))
        except Exception as e:
            print('cannot import', mod, '->', e)
```
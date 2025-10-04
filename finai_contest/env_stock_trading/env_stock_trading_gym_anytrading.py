# Adapted from gym-anytrading (https://github.com/AminHP/gym-anytrading)
# Original author: Amin HP (MIT License)
# Refactored by Chunlin Feng, 2025

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple
from collections import deque

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")


class StockTradingEnv_gym_anytrading(gym.Env):
    """Single-asset trading environment compatible with Gym/Gymnasium.

    The environment keeps cash, current price, position (shares held), and
    optional technical indicators. The observation is a **stack** of the last
    ``window_size`` flattened states to give temporal context.

    Parameters
    ----------
    df : pandas.DataFrame
        Price/feature dataframe indexed by time. Must include columns:
        ``close`` (price) and any names in ``tech_indicator_list``. If a
        ``date`` column exists, it will be used for logs.
    stock_dim : int
        Number of tradable assets. **Must be 1** in this environment.
    hmax : int or float
        Max shares to trade per step (used when converting discrete actions).
        Set to ``numpy.inf`` to allow "sell all"/"buy as much as possible".
    initial_amount : float
        Starting cash balance.
    num_stock_shares : list[int]
        Initial shares held per asset (length = ``stock_dim``, i.e., 1).
    buy_cost_pct : list[float]
        Proportional fee for buys per asset (length = ``stock_dim``).
    sell_cost_pct : list[float]
        Proportional fee for sells per asset (length = ``stock_dim``).
    reward_scaling : float
        Scalar multiplier applied to per-step reward.
    state_space : int
        Flattened state size used by your training pipeline. (Not enforced
        here; actual obs len = ``window_size * _base_obs_len``.)
    action_space : int
        Number of discrete action dimensions; here set via Gym space below.
    tech_indicator_list : list[str]
        Names of technical indicator columns to append to the state.
    turbulence_threshold : float, optional
        If set and current turbulence ≥ threshold, force flat/short and block
        new longs for the step.
    risk_indicator_col : str, default "turbulence"
        Column name used to read the risk/turbulence value from ``df``.
    make_plots : bool, default False
        If True, save a simple account-value plot at episode end.
    print_verbosity : int, default 10
        Print summary every N episodes (``episode % N == 0``).
    day : int, default 0
        Initial row index into ``df``.
    initial : bool, default True
        If True, start from ``initial_amount``/``num_stock_shares``; if False,
        resume from ``previous_state``.
    previous_state : list, optional
        Previous flattened state used when ``initial=False``.
    model_name : str, default ""
        Optional tag for saved logs.
    mode : str, default ""
        Optional tag for saved logs.
    iteration : str, default ""
        Optional tag for saved logs.
    window_size : int, default 30
        Number of past flat states to stack into each observation.

    Attributes
    ----------
    action_space : gymnasium.spaces.MultiDiscrete
        MultiDiscrete([2] * stock_dim), where 0=sell/close, 1=buy/open.
    observation_space : gymnasium.spaces.Box
        1D float vector of length ``window_size * _base_obs_len``.
    state : list[float]
        Current **flat** state (cash, price, position, indicators...).
    reward : float
        Last scaled reward.
    asset_memory : list[float]
        Portfolio value history across the episode.
    actions_memory : list[numpy.ndarray]
        Executed trade sizes per step (zeros when no trade).
    rewards_memory : list[float]
        Scaled per-step rewards.
    date_memory : list
        Logged dates/timestamps for plotting.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int | float,
        initial_amount: float,
        num_stock_shares: list[int],
        buy_cost_pct: list[float],
        sell_cost_pct: list[float],
        reward_scaling: float,
        state_space: int,
        action_space: int,
        tech_indicator_list: list[str],
        turbulence_threshold: float | None = None,
        risk_indicator_col: str = "turbulence",
        make_plots: bool = False,
        print_verbosity: int = 10,
        day: int = 0,
        initial: bool = True,
        previous_state: list[float] | None = None,
        model_name: str = "",
        mode: str = "",
        iteration: str = "",
        window_size: int = 30,
    ) -> None:
        """Initialize the trading environment."""
        assert stock_dim == 1, "This env is single-asset. Use stock_dim=1."
        assert hmax == np.inf, "No maximum cash restriction"
        assert window_size >= 1, "window_size must be >= 1"

        self.day = window_size
        self.df = df.reset_index(drop=True)
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = float(initial_amount)
        self.num_stock_shares = num_stock_shares
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space  # numeric; actual Gym space set below
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state or []
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        self.window_size = int(window_size)
        self._frames: deque[np.ndarray] = deque(maxlen=self.window_size)

        # Actions: 0 = sell/close long, 1 = buy/open long
        self.action_space = gym.spaces.MultiDiscrete([2] * stock_dim)

        # Base flat state = [cash] + [price] + [position] + [price_diff]
        self._base_obs_len = 2*self.stock_dim
        obs_len = self.window_size * self._base_obs_len
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,))
        self.position = [0] * self.stock_dim  # 0=flat/short, 1=long
        self.last_trade_price = [self.df.loc[self.day-1, "close"]] * self.stock_dim

        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.state = self._initiate_state()

        self.reward = 0.0
        self.turbulence = 0.0
        self.cost = 0.0
        self.trades = 0
        self.episode = 0

        initial_total_asset = self.initial_amount + np.sum(
            np.array(self.num_stock_shares) * np.array(self.state[1 : 1 + self.stock_dim])
        )
        self.asset_memory = [float(initial_total_asset)]
        self.rewards_memory: List[float] = []
        self.actions_memory: List[np.ndarray] = []
        self.state_memory: List[list] = []
        self.date_memory: List = [self._get_date()]

    # ---------------------------------------------------------------------
    # Trading helpers
    # ---------------------------------------------------------------------
    def _sell_stock(self, index: int, shares_to_sell: int) -> int:
        """Execute a sell that closes current long position.

        Parameters
        ----------
        index : int
            Asset index (always 0 in this single-asset env).
        shares_to_sell : int
            Requested number of shares to sell (ignored: sell all current).

        Returns
        -------
        int
            Executed shares sold (≥ 0).
        """
        price = float(self.state[1])
        position = float(self.state[1 + self.stock_dim])
        if price <= 0 or position <= 0:
            return 0
        actual = int(position)
        gross = price * actual
        fee = gross * self.sell_cost_pct[index]
        proceeds = gross - fee
        self.state[0] += proceeds
        self.state[1 + self.stock_dim] -= actual
        self.cost += fee
        self.trades += 1
        return actual

    def _buy_stock(self, index: int, shares_to_buy: int) -> int:
        """Execute a buy that opens/expands a long position.

        Parameters
        ----------
        index : int
            Asset index (always 0).
        shares_to_buy : int
            Requested number of shares to buy (ignored: buy as much as possible).

        Returns
        -------
        int
            Executed shares bought (≥ 0).
        """
        price = float(self.state[1])
        if price <= 0:
            return 0
        unit_cost = price * (1 + self.buy_cost_pct[index])
        affordable = int(self.state[0] // unit_cost)
        actual = int(affordable)
        if actual <= 0:
            return 0
        gross = price * actual
        fee = gross * self.buy_cost_pct[index]
        total_cost = gross + fee
        self.state[0] -= total_cost
        self.state[1 + self.stock_dim] += actual
        self.cost += fee
        self.trades += 1
        return actual

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Advance one timestep.

        Parameters
        ----------
        actions : numpy.ndarray
            Discrete per-asset actions (shape ``(stock_dim,)``) where
            ``0``=sell/close long, ``1``=buy/open long.

        Returns
        -------
        obs : numpy.ndarray
            Stacked observation of shape ``(window_size * _base_obs_len,)``.
        reward : float
            Scaled reward = ``reward_scaling * (Δ portfolio value)``.
        terminated : bool
            ``True`` if episode ended naturally (end of data).
        truncated : bool
            Always ``False`` in this implementation.
        info : dict
            Extra diagnostics (empty here).
        """
        self.terminal = self.day >= len(self.df.index) - 1
        if self.terminal:
            if self.make_plots:
                self._make_plot()

            end_total_asset = self._total_asset(self.state)
            df_total_value = pd.DataFrame({"account_value": self.asset_memory})
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)

            tot_reward = end_total_asset - self.asset_memory[0]
            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:.2f}")
                print(f"end_total_asset:   {end_total_asset:.2f}")
                print(f"total_reward:      {tot_reward:.2f}")
                print(f"total_cost:        {self.cost:.2f}")
                print(f"total_trades:      {self.trades}")
                std = df_total_value["daily_return"].std()
                if std not in (None, 0):
                    sharpe = (252 ** 0.5) * df_total_value["daily_return"].mean() / std
                    print(f"Sharpe: {sharpe:.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                self.save_action_memory().to_csv(
                    f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                df_total_value.to_csv(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                pd.DataFrame(
                    {"date": self.date_memory[:-1], "account_rewards": self.rewards_memory}
                ).to_csv(
                    f"results/account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.png")
                plt.close()

            return self._get_obs(), self.reward, self.terminal, False, {}

        acts = np.asarray(actions, dtype=int).reshape(-1)
        pos = np.asarray(self.position, dtype=int).reshape(-1)

        # Optional turbulence override: force flat
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            acts = np.zeros_like(acts)

        begin_total_asset = self._total_asset(self.state)

        sell_hits = np.where((acts == 0) & (pos == 1))[0]  # close longs
        buy_hits = np.where((acts == 1) & (pos == 0))[0]   # open longs

        executed = np.zeros_like(acts, dtype=int)
        
        price_diff = [0] * self.stock_dim
        current_trade_price = [0] * self.stock_dim

        # BUY (open longs)
        for i in buy_hits:
            qty = self.hmax
            traded = self._buy_stock(i, qty)
            self.last_trade_price[i] = self.state[1+i]
            executed[i] = traded
            self.position[i] = 1

        # SELL (close longs)
        for i in sell_hits:
            qty = min(self.hmax, self.state[1 + self.stock_dim])
            traded = self._sell_stock(i, qty)
            current_trade_price[i] = self.state[1+i]
            price_diff[i] = current_trade_price[i] - self.last_trade_price[i]
            executed[i] = -traded
            self.position[i] = 0

        # s -> s+1
        self.day += 1
        self.data = self.df.loc[self.day, :]
        if self.turbulence_threshold is not None and (self.risk_indicator_col in self.data):
            self.turbulence = float(self.data[self.risk_indicator_col])

        self.state = self._update_state()

        end_total_asset = self._total_asset(self.state)
        reward = sum(price_diff)

        self.actions_memory.append(executed.copy())
        self.asset_memory.append(self._total_asset(self.state))
        self.date_memory.append(self._get_date())
        self.rewards_memory.append(reward)
        self.state_memory.append(self.state)
        self._frames.append(np.array([self.state[1], self.state[3]], dtype=np.float32))

        self.reward = reward
        return self._get_obs(), self.reward, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment.

        Parameters
        ----------
        seed : int, optional
            RNG seed.
        options : dict, optional
            Unused; kept for Gymnasium compatibility.

        Returns
        -------
        obs : numpy.ndarray
            Initial stacked observation.
        info : dict
            Empty info dict.
        """
        super().reset(seed=seed)
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            initial_total_asset = self.initial_amount + np.sum(
                np.array(self.num_stock_shares) * np.array(self.state[1 : 1 + self.stock_dim])
            )
            self.asset_memory = [float(initial_total_asset)]
        else:
            prev_total_asset = self.previous_state[0] + np.sum(
                np.array(self.state[1 : 1 + self.stock_dim])
                * np.array(self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )
            self.asset_memory = [float(prev_total_asset)]

        self.turbulence = 0.0
        self.cost = 0.0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1

        # fill frame buffer with the first window of [price, price_diff]
        self._frames.clear()
        start = max(0, self.day - self.window_size + 1)
        end   = self.day + 1  # exclusive

        for idx in range(start, end):
            p   = float(self.df.iloc[idx]["close"])
            prv = float(self.df.iloc[idx - 1]["close"]) if idx > 0 else p
            d   = p - prv
            self._frames.append(np.array([p, d], dtype=np.float32))

        while len(self._frames) < self.window_size:
            self._frames.appendleft(self._frames[0].copy())

        return self._get_obs(), {}

    def render(self, mode: str = "human", close: bool = False) -> np.ndarray:
        """Return the current stacked observation (for debugging/visualization)."""
        return self._get_obs()

    # ---------------------------------------------------------------------
    # State helpers
    # ---------------------------------------------------------------------
    def _initiate_state(self) -> list[float]:
        """Build the initial **flat** state = [cash, price, position, price_diff].

        Returns
        -------
        list[float]
            Flat state vector for the current day.
        """
        price = float(self.data["close"])
        if self.day > 0:
            prev_price = float(self.df.loc[self.day - 1, "close"])
            price_diff = price - prev_price
        else:
            price_diff = 0.0  # no previous day
        if self.initial:
            state = (
                [self.initial_amount]
                + [price]
                + self.num_stock_shares
                + [price_diff]
            )
        else:
            state = (
                [self.previous_state[0]]
                + [price]
                + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                + [price_diff]
            )
        return state

    def _update_state(self) -> list[float]:
        """Update the **flat** state for the next day.

        Returns
        -------
        list[float]
            Flat state vector for the next day.
        """
        price = float(self.data["close"])
        if self.day > 0:
            prev_price = float(self.df.loc[self.day - 1, "close"])
            price_diff = price - prev_price
        else:
            price_diff = 0.0  # no previous day
        state = (
            [self.state[0]]
            + [price]
            + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            + [price_diff]
        )
        return state

    def _get_obs(self) -> np.ndarray:
        """Concatenate the last ``window_size`` flat states into a 1D vector.

        Returns
        -------
        numpy.ndarray
            Stacked observation of shape ``(window_size * _base_obs_len,)``.
        """
        if len(self._frames) < self.window_size:
            pad = [self._frames[-1]] * (self.window_size - len(self._frames))
            frames = list(self._frames) + pad
        else:
            frames = list(self._frames)
        return np.concatenate(frames, axis=0).astype(np.float32)

    def _get_date(self):
        """Return the current date/timestamp label used for logging."""
        return self.data["date"] if "date" in self.data else self.day

    def _make_plot(self) -> None:
        """Save a simple account-value plot to the ``results/`` folder."""
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def _total_asset(self, state_vec: Sequence[float]) -> float:
        """Compute total portfolio value on a flat state.

        Parameters
        ----------
        state_vec : Sequence[float]
            Flat state: ``[cash, price, position, ...]``.

        Returns
        -------
        float
            ``cash + price * position``.
        """
        cash = float(state_vec[0])
        price = float(state_vec[1])
        position = float(state_vec[1 + self.stock_dim])
        return float(cash + price * position)

    # ---------------------------------------------------------------------
    # Logging helpers
    # ---------------------------------------------------------------------
    def save_state_memory(self) -> pd.DataFrame:
        """Return the recorded flat states as a DataFrame."""
        date_list = self.date_memory[:-1]
        state_list = self.state_memory
        return pd.DataFrame({"date": date_list, "states": state_list})

    def save_asset_memory(self) -> pd.DataFrame:
        """Return the account value time-series as a DataFrame."""
        return pd.DataFrame({"date": self.date_memory, "account_value": self.asset_memory})

    def save_action_memory(self) -> pd.DataFrame:
        """Return the executed trades per step as a DataFrame."""
        date_list = self.date_memory[:-1]
        action_list = self.actions_memory
        return pd.DataFrame({"date": date_list, "actions": action_list})

    # ---------------------------------------------------------------------
    # SB3 helper
    # ---------------------------------------------------------------------
    def get_sb_env(self) -> Tuple[DummyVecEnv, Any]:
        """Wrap this env in SB3 ``DummyVecEnv`` and return ``(env, obs)``."""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs
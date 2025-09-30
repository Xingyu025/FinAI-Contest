# Adapted from FinRL (https://github.com/AI4Finance-Foundation/FinRL-Meta/blob/master/meta/env_stock_trading/env_stock_trading.py)

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv

matplotlib.use("Agg")


class StockTradingEnv_FinRLMeta(gym.Env):
    """A stock trading environment for Gymnasium compatible with SB3.

    The environment tracks cash, per-asset prices, held shares, and optional
    technical indicators. Actions are continuous in ``[-1, 1]`` for each asset
    and are scaled by ``hmax`` internally to discrete share counts.

    Parameters
    ----------
    df : pandas.DataFrame
        Panel-like market data indexed by day, with columns including at least
        ``date``, ``close`` (per asset) and entries for all names in
        ``tech_indicator_list``. For multi-asset data, rows for the same day
        should be grouped and accessible via ``df.loc[day, :]``.
    stock_dim : int
        Number of tradable assets.
    hmax : int
        Max shares to buy/sell per asset at each step (used to scale actions).
    initial_amount : int
        Starting cash.
    num_stock_shares : list[int]
        Initial shares held per asset (length = ``stock_dim``).
    buy_cost_pct : list[float]
        Proportional buy fee per asset (length = ``stock_dim``).
    sell_cost_pct : list[float]
        Proportional sell fee per asset (length = ``stock_dim``).
    reward_scaling : float
        Scalar multiplier applied to the per-step reward.
    state_space : int
        Flattened observation dimension.
    action_space : int
        Number of action dimensions (should equal ``stock_dim``).
    tech_indicator_list : list[str]
        Names of technical-indicator columns to include in the state.
    turbulence_threshold : float, optional
        If not ``None`` and current turbulence ≥ threshold, force liquidations
        and block new buys that step.
    risk_indicator_col : str, default "turbulence"
        Column name used as the risk indicator for the threshold logic.
    make_plots : bool, default False
        If True, save a simple account-value plot at the end of each episode.
    print_verbosity : int, default 10
        Print summary every N episodes (episode % N == 0).
    day : int, default 0
        Initial day index in the dataframe.
    initial : bool, default True
        If True, start from ``initial_amount`` and ``num_stock_shares``; if False,
        reuse ``previous_state`` as starting portfolio.
    previous_state : list, default []
        Previous state to resume from when ``initial=False``.
    model_name : str, default ""
        Optional tag used when saving episode logs.
    mode : str, default ""
        Optional tag used when saving episode logs.
    iteration : str, default ""
        Optional tag used when saving episode logs.

    Attributes
    ----------
    action_space : gymnasium.spaces.Box
        Continuous actions in ``[-1, 1]`` per asset (shape = ``(stock_dim,)``).
    observation_space : gymnasium.spaces.Box
        Observation vector of length ``state_space``.
    state : list[float]
        Current flattened observation.
    reward : float
        Last scaled reward returned by :meth:`step`.
    episode : int
        Episode counter (incremented in :meth:`reset`).
    asset_memory : list[float]
        Portfolio value time-series across the episode.
    actions_memory : list[np.ndarray]
        Actions taken each step.
    rewards_memory : list[float]
        Unscaled per-step rewards (delta in total asset).
    date_memory : list
        Recorded dates per step for logging.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        stock_dim: int,
        hmax: int,
        initial_amount: int,
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
    ) -> None:
        """Initialize the trading environment.

        Notes
        -----
        The observation is constructed as:

        - cash (1)
        - close prices for each asset (``stock_dim``)
        - shares held for each asset (``stock_dim``)
        - technical indicators for each asset, flattened in the order of
          ``tech_indicator_list``

        Total length equals ``state_space`` which must match this layout.
        """
        self.day = day
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.num_stock_shares = num_stock_shares
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space  # numeric count; Box below overwrites attr
        self.tech_indicator_list = tech_indicator_list

        self.action_space = spaces.Box(low=-1, high=1, shape=(self.action_space,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )

        self.data = self.df.loc[self.day, :]
        self.terminal = False
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.initial = initial
        self.previous_state = previous_state or []
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        # initialize state & episode trackers
        self.state = self._initiate_state()
        self.reward = 0.0
        self.turbulence = 0.0
        self.cost = 0.0
        self.trades = 0
        self.episode = 0

        # initial portfolio value: cash + sum(shares_i * price_i)
        self.asset_memory = [
            self.initial_amount
            + np.sum(
                np.array(self.num_stock_shares)
                * np.array(self.state[1 : 1 + self.stock_dim])
            )
        ]
        self.rewards_memory: list[float] = []
        self.actions_memory: list[np.ndarray] = []
        # keep intermediate states for logging
        self.state_memory: list[list[float]] = []
        self.date_memory = [self._get_date()]

        self._seed()

    # ---------------------------------------------------------------------
    # Trading helpers
    # ---------------------------------------------------------------------
    def _sell_stock(self, index: int, action: int) -> int:
        """Execute a sell order for a single asset.

        Parameters
        ----------
        index : int
            Asset index in ``[0, stock_dim)``.
        action : int
            Intended number of shares to sell (non-positive raw action mapped to size).

        Returns
        -------
        int
            Executed number of shares sold (≥ 0).
        """
        def _do_sell_normal() -> int:
            #Check if stock position is positive which means it is tradable
            if self.state[index + 1] > 0:
                if self.state[index + self.stock_dim + 1] > 0:
                    sell_num_shares = min(
                        abs(action), self.state[index + self.stock_dim + 1]
                    )
                    sell_amount = (
                        self.state[index + 1]
                        * sell_num_shares
                        * (1 - self.sell_cost_pct[index])
                    )
                    self.state[0] += sell_amount
                    self.state[index + self.stock_dim + 1] -= sell_num_shares
                    self.cost += (
                        self.state[index + 1]
                        * sell_num_shares
                        * self.sell_cost_pct[index]
                    )
                    self.trades += 1
                else:
                    sell_num_shares = 0
            else:
                print("not tradable")
                sell_num_shares = 0
            return sell_num_shares

        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            if self.state[index + 1] > 0 and self.state[index + self.stock_dim + 1] > 0:
                sell_num_shares = self.state[index + self.stock_dim + 1]
                sell_amount = (
                    self.state[index + 1]
                    * sell_num_shares
                    * (1 - self.sell_cost_pct[index])
                )
                self.state[0] += sell_amount
                self.state[index + self.stock_dim + 1] = 0
                self.cost += (
                    self.state[index + 1]
                    * sell_num_shares
                    * self.sell_cost_pct[index]
                )
                self.trades += 1
            else:
                sell_num_shares = 0
        else:
            sell_num_shares = _do_sell_normal()
        return sell_num_shares

    def _buy_stock(self, index: int, action: int) -> int:
        """Execute a buy order for a single asset.

        Parameters
        ----------
        index : int
            Asset index in ``[0, stock_dim)``.
        action : int
            Intended number of shares to buy (non-negative raw action mapped to size).

        Returns
        -------
        int
            Executed number of shares bought (≥ 0).
        """
        def _do_buy() -> int:
            #Check if stock position is positive which means it is tradable
            if self.state[index + 1] > 0:
                # integer shares we can afford including fees
                available_amount = self.state[0] // (
                    self.state[index + 1] * (1 + self.buy_cost_pct[index])
                )
                buy_num_shares = int(min(available_amount, action))
                buy_amount = (
                    self.state[index + 1]
                    * buy_num_shares
                    * (1 + self.buy_cost_pct[index])
                )
                self.state[0] -= buy_amount
                self.state[index + self.stock_dim + 1] += buy_num_shares
                self.cost += (
                    self.state[index + 1] * buy_num_shares * self.buy_cost_pct[index]
                )
                self.trades += 1
            else:
                buy_num_shares = 0
            return buy_num_shares

        if self.turbulence_threshold is None or self.turbulence < self.turbulence_threshold:
            buy_num_shares = _do_buy()
        else:
            buy_num_shares = 0
        return buy_num_shares

    def _make_plot(self) -> None:
        """Save a simple plot of account value over the episode to ``results/``."""
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    # ---------------------------------------------------------------------
    # Gymnasium API
    # ---------------------------------------------------------------------
    def step(self, actions: np.ndarray) -> Tuple[list[float], float, bool, bool, Dict[str, Any]]:
        """Run one environment step.

        Parameters
        ----------
        actions : numpy.ndarray
            Continuous actions in ``[-1, 1]`` per asset; internally scaled by
            ``hmax`` and cast to integer share counts.

        Returns
        -------
        obs : list[float]
            Next observation (flattened).
        reward : float
            Scaled reward = ``reward_scaling * (Δ portfolio value)``.
        terminated : bool
            Episode termination flag (end of data).
        truncated : bool
            Always ``False`` in this implementation.
        info : dict
            Extra diagnostics (empty here).
        """
        self.terminal = self.day >= len(self.df.index.unique()) - 1
        if self.terminal:
            if self.make_plots:
                self._make_plot()

            end_total_asset = self.state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
            )

            df_total_value = pd.DataFrame(self.asset_memory, columns=["account_value"])
            df_total_value["date"] = self.date_memory
            df_total_value["daily_return"] = df_total_value["account_value"].pct_change(1)

            if df_total_value["daily_return"].std() != 0:
                sharpe = (
                    (252 ** 0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
                )

            tot_reward = end_total_asset - self.asset_memory[0]
            df_rewards = pd.DataFrame(self.rewards_memory, columns=["account_rewards"])
            df_rewards["date"] = self.date_memory[:-1]

            if self.episode % self.print_verbosity == 0:
                print(f"day: {self.day}, episode: {self.episode}")
                print(f"begin_total_asset: {self.asset_memory[0]:0.2f}")
                print(f"end_total_asset: {end_total_asset:0.2f}")
                print(f"total_reward: {tot_reward:0.2f}")
                print(f"total_cost: {self.cost:0.2f}")
                print(f"total_trades: {self.trades}")
                if df_total_value["daily_return"].std() != 0:
                    print(f"Sharpe: {sharpe:0.3f}")
                print("=================================")

            if (self.model_name != "") and (self.mode != ""):
                df_actions = self.save_action_memory()
                df_actions.to_csv(
                    f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv"
                )
                df_total_value.to_csv(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                df_rewards.to_csv(
                    f"results/account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv",
                    index=False,
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.png"
                )
                plt.close()

            return self.state, self.reward, self.terminal, False, {}

        # not terminal: apply actions
        actions = (actions * self.hmax).astype(int)
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            actions = np.array([-self.hmax] * self.stock_dim)

        begin_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        )

        argsort_actions = np.argsort(actions)
        sell_index = argsort_actions[: np.where(actions < 0)[0].shape[0]]
        buy_index = argsort_actions[::-1][: np.where(actions > 0)[0].shape[0]]

        for index in sell_index:
            actions[index] = -self._sell_stock(index, actions[index])

        for index in buy_index:
            actions[index] = self._buy_stock(index, actions[index])

        self.actions_memory.append(actions)

        # s -> s+1
        self.day += 1
        self.data = self.df.loc[self.day, :]
        if self.turbulence_threshold is not None:
            if len(self.df.tic.unique()) == 1:
                self.turbulence = float(self.data[self.risk_indicator_col])
            else:
                self.turbulence = float(self.data[self.risk_indicator_col].values[0])

        self.state = self._update_state()

        end_total_asset = self.state[0] + sum(
            np.array(self.state[1 : (self.stock_dim + 1)])
            * np.array(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
        )
        self.asset_memory.append(end_total_asset)
        self.date_memory.append(self._get_date())

        unscaled_reward = end_total_asset - begin_total_asset
        self.rewards_memory.append(unscaled_reward)
        self.reward = float(unscaled_reward * self.reward_scaling)
        self.state_memory.append(self.state)

        return self.state, self.reward, self.terminal, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, Any] | None = None,
    ) -> Tuple[list[float], Dict[str, Any]]:
        """Reset the environment.

        Parameters
        ----------
        seed : int, optional
            RNG seed.
        options : dict, optional
            Unused; present for Gymnasium API compatibility.

        Returns
        -------
        obs : list[float]
            Initial observation.
        info : dict
            Empty info dict.
        """
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            self.asset_memory = [
                self.initial_amount
                + np.sum(
                    np.array(self.num_stock_shares)
                    * np.array(self.state[1 : 1 + self.stock_dim])
                )
            ]
        else:
            previous_total_asset = self.previous_state[0] + sum(
                np.array(self.state[1 : (self.stock_dim + 1)])
                * np.array(
                    self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                )
            )
            self.asset_memory = [previous_total_asset]

        self.turbulence = 0.0
        self.cost = 0.0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1
        return self.state, {}

    def render(self, mode: str = "human", close: bool = False) -> list[float]:
        """Return the current state for debugging/visualization."""
        return self.state

    # ---------------------------------------------------------------------
    # State construction
    # ---------------------------------------------------------------------
    def _initiate_state(self) -> list[float]:
        """Build the initial state vector (cash, prices, shares, indicators).

        Returns
        -------
        list[float]
            Flattened observation.
        """
        if self.initial:
            # Initial State
            if len(self.df.tic.unique()) > 1:
                # multiple assets
                state = (
                    [self.initial_amount]
                    + self.data.close.values.tolist()
                    + self.num_stock_shares
                    + sum(
                        (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                        [],
                    )
                )
            else:
                # single asset
                state = (
                    [self.initial_amount]
                    + [float(self.data.close)]
                    + [0] * self.stock_dim
                    + sum(([float(self.data[tech])] for tech in self.tech_indicator_list), [])
                )
        else:
            # Using previous state
            if len(self.df.tic.unique()) > 1:
                state = (
                    [self.previous_state[0]]
                    + self.data.close.values.tolist()
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    + sum(
                        (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                        [],
                    )
                )
            else:
                state = (
                    [self.previous_state[0]]
                    + [float(self.data.close)]
                    + self.previous_state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)]
                    + sum(([float(self.data[tech])] for tech in self.tech_indicator_list), [])
                )
        return state

    def _update_state(self) -> list[float]:
        """Update the state vector for the next timestep.

        Returns
        -------
        list[float]
            Flattened next observation.
        """
        if len(self.df.tic.unique()) > 1:
            state = (
                [self.state[0]]
                + self.data.close.values.tolist()
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(
                    (self.data[tech].values.tolist() for tech in self.tech_indicator_list),
                    [],
                )
            )
        else:
            state = (
                [self.state[0]]
                + [float(self.data.close)]
                + list(self.state[(self.stock_dim + 1) : (self.stock_dim * 2 + 1)])
                + sum(([float(self.data[tech])] for tech in self.tech_indicator_list), [])
            )
        return state

    def _get_date(self):
        """Return the current date label from the dataframe (per asset layout)."""
        if len(self.df.tic.unique()) > 1:
            date = self.data.date.unique()[0]
        else:
            date = self.data.date
        return date

    # ---------------------------------------------------------------------
    # Logging helpers
    # ---------------------------------------------------------------------
    def save_state_memory(self) -> pd.DataFrame:
        """Return a DataFrame snapshot of states across the episode.

        Returns
        -------
        pandas.DataFrame
            DataFrame of recorded states (indexed by date for multi-asset).
        """
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list, columns=["date"])

            state_list = self.state_memory
            df_states = pd.DataFrame(
                state_list,
                columns=[
                    "cash",
                    "Bitcoin_price",
                    "Gold_price",
                    "Bitcoin_num",
                    "Gold_num",
                    "Bitcoin_Disable",
                    "Gold_Disable",
                ],
            )
            df_states.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            state_list = self.state_memory
            df_states = pd.DataFrame({"date": date_list, "states": state_list})
        return df_states

    def save_asset_memory(self) -> pd.DataFrame:
        """Return the account value time-series as a DataFrame."""
        date_list = self.date_memory
        asset_list = self.asset_memory
        df_account_value = pd.DataFrame({"date": date_list, "account_value": asset_list})
        return df_account_value

    def save_action_memory(self) -> pd.DataFrame:
        """Return the action time-series as a DataFrame.

        For multi-asset data, the DataFrame columns correspond to asset tickers.
        """
        if len(self.df.tic.unique()) > 1:
            date_list = self.date_memory[:-1]
            df_date = pd.DataFrame(date_list, columns=["date"])

            action_list = self.actions_memory
            df_actions = pd.DataFrame(action_list)
            df_actions.columns = self.data.tic.values
            df_actions.index = df_date.date
        else:
            date_list = self.date_memory[:-1]
            action_list = self.actions_memory
            df_actions = pd.DataFrame({"date": date_list, "actions": action_list})
        return df_actions

    # ---------------------------------------------------------------------
    # Misc
    # ---------------------------------------------------------------------
    def _seed(self, seed: int | None = None) -> List[int]:
        """Seed the internal RNG (Gymnasium-style)."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]  # noqa: F722

    def get_sb_env(self) -> Tuple[DummyVecEnv, Any]:
        """Wrap this single env in a SB3 ``DummyVecEnv``.

        Returns
        -------
        env : stable_baselines3.common.vec_env.DummyVecEnv
            Vectorized environment containing this env.
        obs : Any
            Initial observation returned by ``env.reset()``.
        """
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

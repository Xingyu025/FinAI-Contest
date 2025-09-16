# Adapted from gym-anytrading (https://github.com/AminHP/gym-anytrading)
# Original author: Amin HP (MIT License)
# Refactored by Chunlin Feng, 2025

from __future__ import annotations

from typing import List

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gymnasium import spaces
from gymnasium.utils import seeding
from stable_baselines3.common.vec_env import DummyVecEnv
from collections import deque

matplotlib.use("Agg")


class StockTradingEnv_gym_anytrading(gym.Env):
    """
        df (pandas.DataFrame): Dataframe containing data
        hmax (int): Maximum cash to be traded in each trade per asset.
        initial_amount (int): Amount of cash initially available
        buy_cost_pct (float, array): Cost for buying shares, each index corresponds to each asset
        sell_cost_pct (float, array): Cost for selling shares, each index corresponds to each asset
        turbulence_threshold (float): Maximum turbulence allowed in market for purchases to occur. If exceeded, positions are liquidated
        print_verbosity(int): When iterating (step), how often to print stats about state of env
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
        turbulence_threshold=None,
        risk_indicator_col="turbulence",
        make_plots: bool = False,
        print_verbosity=10,
        day=0,
        initial=True,
        previous_state=[],
        model_name="",
        mode="",
        iteration="",
        window_size: int = 30,
    ):
        assert stock_dim == 1, "This env is single-asset. Use stock_dim=1."
        assert hmax == np.inf, "No maximum cash restriction"
        assert window_size >= 1, "window_size must be >= 1"

        self.day = day
        self.df = df.reset_index(drop=True)
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.num_stock_shares = num_stock_shares
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.turbulence_threshold = turbulence_threshold
        self.risk_indicator_col = risk_indicator_col
        self.make_plots = make_plots
        self.print_verbosity = print_verbosity
        self.initial = initial
        self.previous_state = previous_state
        self.model_name = model_name
        self.mode = mode
        self.iteration = iteration

        self.window_size = int(window_size)
        self._frames: deque[np.ndarray] = deque(maxlen=self.window_size)


        self.action_space = gym.spaces.MultiDiscrete([2]*stock_dim) #0 for sell, and 1 for buy
        # base single-timestep obs len: [cash, price, position, techs...]
        self._base_obs_len = 1 + self.stock_dim + self.stock_dim + len(self.tech_indicator_list)
        obs_len = self.window_size * self._base_obs_len
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,))
        self.position = [0]*self.stock_dim # 0 for short and 1 for long

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
        self.asset_memory = [initial_total_asset]
        self.rewards_memory: List[float] = []
        self.actions_memory: List[np.ndarray] = []
        self.state_memory: List[list] = []
        self.date_memory: List = [self._get_date()]

    def _sell_stock(self, index: int, shares_to_sell: int) -> int:
        price = self.state[1]
        position = self.state[1 + self.stock_dim]
        if price <= 0 or position <= 0:
            return 0
        actual = int(position)
        gross = price * actual
        fee = gross * self.sell_cost_pct[index]
        proceeds = gross - fee
        # update
        self.state[0] += proceeds
        self.state[1 + self.stock_dim] -= actual
        self.cost += fee
        self.trades += 1
        return actual

    def _buy_stock(self, index: int, shares_to_buy: int) -> int:
        price = self.state[1]
        if price <= 0:
            return 0
        # consider fee in affordability
        unit_cost = price * (1 + self.buy_cost_pct[index])
        affordable = int(self.state[0] // unit_cost)
        actual = int(affordable)
        if actual <= 0:
            return 0
        gross = price * actual
        fee = gross * self.buy_cost_pct[index]
        total_cost = gross + fee
        # update
        self.state[0] -= total_cost
        self.state[1 + self.stock_dim] += actual
        self.cost += fee
        self.trades += 1
        return actual

    def step(self, actions):
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
                if df_total_value["daily_return"].std() not in (None, 0):
                    sharpe = (252**0.5) * df_total_value["daily_return"].mean() / df_total_value["daily_return"].std()
                    print(f"Sharpe: {sharpe:.3f}")
                print("=================================")


            if (self.model_name != "") and (self.mode != ""):
                self.save_action_memory().to_csv(
                    f"results/actions_{self.mode}_{self.model_name}_{self.iteration}.csv", index=False
                )
                df_total_value.to_csv(
                    f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.csv", index=False
                )
                pd.DataFrame({"date": self.date_memory[:-1], "account_rewards": self.rewards_memory}).to_csv(
                    f"results/account_rewards_{self.mode}_{self.model_name}_{self.iteration}.csv", index=False
                )
                plt.plot(self.asset_memory, "r")
                plt.savefig(f"results/account_value_{self.mode}_{self.model_name}_{self.iteration}.png")
                plt.close()

            # return the stacked observation at terminal as well
            return self._get_obs(), self.reward, self.terminal, False, {}

        acts = np.asarray(actions, dtype=int).reshape(-1)
        pos  = np.asarray(self.position, dtype=int).reshape(-1)

        # Optional turbulence override: force flat/short
        if self.turbulence_threshold is not None and self.turbulence >= self.turbulence_threshold:
            acts = np.zeros_like(acts)  # force side=0

        begin_total_asset = self._total_asset(self.state)

        sell_hits = np.where((acts == 0) & (pos == 1))[0]  # long -> close
        buy_hits  = np.where((acts == 1) & (pos == 0))[0]  # flat/short -> open

        did_trade = (sell_hits.size > 0) or (buy_hits.size > 0)
        executed = np.zeros_like(acts, dtype=int)
        reward = 0.0

        if did_trade:
            # SELL (close longs)
            for i in sell_hits:
                # choose a quantity policy: sell all current shares or up to hmax
                qty = min(self.hmax, self.state[1 + self.stock_dim])
                traded = self._sell_stock(i, qty) 
                executed[i] = -traded
                self.position[i] = 0

            # BUY (open longs)
            for i in buy_hits:
                qty = self.hmax
                traded = self._buy_stock(i, qty) 
                executed[i] = traded
                self.position[i] = 1
            
        self.day += 1
        self.data = self.df.loc[self.day, :]
        if self.turbulence_threshold is not None and (self.risk_indicator_col in self.data):
            self.turbulence = float(self.data[self.risk_indicator_col])

        self.state = self._update_state()

        end_total_asset = self._total_asset(self.state) 
        reward = (end_total_asset - begin_total_asset) * self.reward_scaling

        self.actions_memory.append(executed.copy())          # zeros if no trade
        self.asset_memory.append(self._total_asset(self.state))
        self.date_memory.append(self._get_date())
        self.rewards_memory.append(reward)
        self.state_memory.append(self.state)
        self._frames.append(np.asarray(self.state, dtype=np.float32))

        self.reward = reward
        return self._get_obs(), self.reward, False, False, {}


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.state = self._initiate_state()

        if self.initial:
            initial_total_asset = self.initial_amount + np.sum(
                np.array(self.num_stock_shares) * np.array(self.state[1 : 1 + self.stock_dim])
            )
            self.asset_memory = [initial_total_asset]
        else:
            prev_total_asset = self.previous_state[0] + np.sum(
                np.array(self.state[1 : 1 + self.stock_dim]) *
                np.array(self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)])
            )
            self.asset_memory = [prev_total_asset]

        self.turbulence = 0.0
        self.cost = 0.0
        self.trades = 0
        self.terminal = False
        self.rewards_memory = []
        self.actions_memory = []
        self.date_memory = [self._get_date()]
        self.episode += 1

        # fill frame buffer with the initial flat state (or previous_state, if provided)
        self._frames.clear()
        base = np.asarray(self.state, dtype=np.float32)
        # If someone passes a full stacked previous_state, we could detect and use it.
        for _ in range(self.window_size):
            self._frames.append(base.copy())

        return self._get_obs(), {}

    def render(self, mode="human", close=False):
        return self._get_obs()

    # ----- helpers -----

    def _initiate_state(self):
        # single-timestep flat state = [cash] + [price] + [position] + techs...
        price = float(self.data["close"])
        if self.initial:
            state = (
                [self.initial_amount] + [price] + self.num_stock_shares +
                [float(self.data[t]) for t in self.tech_indicator_list]
            )
        else:
            state = (
                [self.previous_state[0]] + [price] +
                self.previous_state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)] +
                [float(self.data[t]) for t in self.tech_indicator_list]
            )
        return state

    def _update_state(self):
        price = float(self.data["close"])
        state = (
            [self.state[0]] + [price] +
            list(self.state[(self.stock_dim + 1):(self.stock_dim * 2 + 1)]) +
            [float(self.data[t]) for t in self.tech_indicator_list]
        )
        return state

    def _get_obs(self) -> np.ndarray:
        """Concatenate last `window_size` flat states."""
        if len(self._frames) < self.window_size:
            # pad just in case
            pad = [self._frames[-1]] * (self.window_size - len(self._frames))
            frames = list(self._frames) + pad
        else:
            frames = list(self._frames)
        return np.concatenate(frames, axis=0).astype(np.float32)

    def _get_date(self):
        return self.data["date"] if "date" in self.data else self.day

    def _make_plot(self):
        plt.plot(self.asset_memory, "r")
        plt.savefig(f"results/account_value_trade_{self.episode}.png")
        plt.close()

    def _total_asset(self, state_vec):
        # cash + price*position  (computed on single flat state)
        cash = state_vec[0]
        price = state_vec[1]
        position = state_vec[1 + self.stock_dim]  # after prices
        return float(cash + price * position)



    def save_state_memory(self):
        date_list = self.date_memory[:-1]
        state_list = self.state_memory
        df_states = pd.DataFrame({"date": date_list, "states": state_list})
        return df_states

    def save_asset_memory(self):
        return pd.DataFrame({"date": self.date_memory, "account_value": self.asset_memory})

    def save_action_memory(self):
        date_list = self.date_memory[:-1]
        action_list = self.actions_memory
        return pd.DataFrame({"date": date_list, "actions": action_list})
    
    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

# Adapted from Trading_Gym (https://github.com/Yvictor/TradingGym)
# Original author: Yvictor (MIT License)
# Refactored by Xiangyu Ji, 2025

from __future__ import annotations

from typing import Any, Dict, Tuple, List

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


class TradingEnvStandardized(gym.Env):
    """
    Single-asset trading environment with discrete actions and rolling-window observations.

    ----------------------------------------------------
    Action Space
    ----------------------------------------------------
    Discrete(3):
        0 -> Hold (do nothing)
        1 -> Buy  (open / increase long)
        2 -> Sell (open / increase short)

    The agent cannot exceed the maximum allowed position `max_position`.
    - If a Buy at +max_position is attempted, it becomes Hold.
    - If a Sell at -max_position is attempted, it becomes Hold.

    ----------------------------------------------------
    State Space (internal)
    ----------------------------------------------------
    Market Data
    -----------
    - df_sample
        Sub-DataFrame representing one continuous trading session.
    - price
        1-D array of prices used for reward & PnL calculations.
    - obs_features
        2-D array of selected market features at each time step.

    Position State
    --------------
    - posi_arr
        1-D array of position (number of shares/contracts) at each time step.
    - posi_variation_arr
        1-D array of position changes due to trades.
    - posi_entry_cover_arr
        1-D array encoding type of trade:
            +1 -> new entry
            +2 -> increase existing position
            -1 -> partial cover
            -2 -> full close
             0 -> no trade

    Price Statistics
    ----------------
    - price_mean_arr
        1-D array of average entry price of current position at each time.

    Reward and P&L
    --------------
    - reward_fluctuant_arr
        Unrealized P&L:
            (price - price_mean_arr) * posi_arr
    - reward_makereal_arr
        Indicator whether P&L at a step is realized (1) or not (0).
    - reward_arr
        1-D array of realized reward per time step.

    Control / Bookkeeping
    ---------------------
    - step_st
        Start index of current observation window.
    - info
        Auxiliary info (e.g. joined df + transaction log) at episode end.
    - transaction_details
        DataFrame logging position, PnL, etc. at the end of the episode.

    ----------------------------------------------------
    Observation Space (agent-facing)
    ----------------------------------------------------
    The agent sees a rolling window of length `obs_len`.

    Base market window:
        obs_state = obs_features[step_st : step_st + obs_len]
        shape: (obs_len, feature_len)

    If `return_transaction=False`:
        observation = obs_state

    If `return_transaction=True`:
        For each timestep in the window, we concatenate:
          [market features,
           posi_arr,
           posi_variation_arr,
           posi_entry_cover_arr,
           price,
           price_mean_arr,
           reward_fluctuant_arr,
           reward_makereal_arr,
           reward_arr]
        so the number of columns becomes:
          feature_len + 8

    In both cases, the Gym observation space is a Box with shape:
        (obs_len, obs_feature_len)

    ----------------------------------------------------
    Reward (Option A)
    ----------------------------------------------------
    At each step, after trades and updates, the environment returns:

        step_reward = obs_reward.sum()

    where `obs_reward` is the slice of `reward_arr` over the current
    observation window. (Unrealized reward is not directly returned.)

    ----------------------------------------------------
    Initial Condition (reset)
    ----------------------------------------------------
    At reset():
    - df_sample           <- random session from full df
    - step_st             <- 0
    - price               <- df_sample[price_name].values
    - obs_features        <- df_sample[feature_names].values
    - posi_arr            <- zeros
    - posi_variation_arr  <- zeros
    - posi_entry_cover_arr<- zeros
    - price_mean_arr      <- copy of price
    - reward_fluctuant_arr<- zeros
    - reward_makereal_arr <- zeros
    - reward_arr          <- zeros
    - info                <- None
    - transaction_details <- empty DataFrame
    - t_index             <- 0
    - First observation window is taken from index 0 and padded if needed.
    """
    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        obs_len: int,
        step_len: int,
        fee: float,
        max_position: int = 5,
        deal_col_name: str = "price",
        feature_names: List[str] | None = None,
        return_transaction: bool = False,
        fluc_div: float = 100.0,
        gameover_limit: float | None = None,
    ) -> None:
        super().__init__()

        assert obs_len >= 1, "obs_len must be >= 1"
        assert step_len >= 1, "step_len must be >= 1"
        feature_names = feature_names or ["price", "volume"]

        self.df = df.reset_index(drop=True)
        self.obs_len = int(obs_len)
        self.step_len = int(step_len)
        self.fee = float(fee)
        self.max_position = int(max_position)
        self.price_name = deal_col_name
        self.using_feature = feature_names
        self.feature_len = len(self.using_feature)
        self.return_transaction = return_transaction
        self.fluc_div = float(fluc_div)
        self.gameover_limit = gameover_limit

        # Precompute session boundaries:
        # we assume 'serial_number' marks session start (==0)
        if "serial_number" not in self.df.columns:
            raise ValueError("DataFrame must contain 'serial_number' column.")
        self.begin_fs = self.df[self.df["serial_number"] == 0]
        self.date_leng = len(self.begin_fs)

        # ---- Gym spaces ----
        # Actions: 0=Hold, 1=Buy, 2=Sell
        self.action_space = spaces.Discrete(3)

        # Observation: rolling window with optional transaction features
        self.transaction_feature_len = 8 if self.return_transaction else 0
        self.obs_feature_len = self.feature_len + self.transaction_feature_len

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_len, self.obs_feature_len),
            dtype=np.float32,
        )

        # Internal episode state placeholders
        self.df_sample: pd.DataFrame | None = None
        self.price: np.ndarray | None = None
        self.obs_features: np.ndarray | None = None

        self.posi_arr: np.ndarray | None = None
        self.posi_variation_arr: np.ndarray | None = None
        self.posi_entry_cover_arr: np.ndarray | None = None
        self.price_mean_arr: np.ndarray | None = None
        self.reward_fluctuant_arr: np.ndarray | None = None
        self.reward_makereal_arr: np.ndarray | None = None
        self.reward_arr: np.ndarray | None = None

        self.step_st: int = 0
        self.t_index: int = 0
        self.info: Any = None
        self.transaction_details: pd.DataFrame = pd.DataFrame()

        # Observation caches
        self.obs_state: np.ndarray | None = None
        self.obs_posi: np.ndarray | None = None
        self.obs_posi_var: np.ndarray | None = None
        self.obs_posi_entry_cover: np.ndarray | None = None
        self.obs_price: np.ndarray | None = None
        self.obs_price_mean: np.ndarray | None = None
        self.obs_reward_fluctuant: np.ndarray | None = None
        self.obs_makereal: np.ndarray | None = None
        self.obs_reward: np.ndarray | None = None
        self.obs_return: np.ndarray | None = None


"""Gymnasium Environment for M5 Walmart Inventory Replenishment.

Compliant with Gymnasium API v0.26+ / v1.0.
Formulated to match exactly the operational mechanics, timeline, and cost structure
of the baseline M5 simulator (m5_baseline.ipynb) under Lost Sales and Lead Time L=2.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class M5SingleItemEnv(gym.Env):
    """Single-item inventory replenishment environment driven by empirical M5 demand series.

    Operational Timeline at day t (matching simulate() in m5_baseline.ipynb):
        1. Receive arriving orders placed at t - L:
           stock = stock_{t-1} + received_t
        2. Policy observes state (stock, pipeline, history) and decides order quantity q_t.
           If q_t > 0, order is scheduled for arrival at t + L.
        3. Customer demand D_t is realized from empirical M5 series:
           sales_t = min(stock, D_t)
           lost_sales_t = max(0, D_t - stock)
           end_inventory_{t+1} = stock - sales_t >= 0
        4. Operational cost accounting:
           holding_cost = h * end_inventory
           lost_sales_cost = p * lost_sales
           ordering_cost = k if q_t > 0 else 0.0
           total_cost = holding_cost + lost_sales_cost + ordering_cost
           reward = -total_cost

    Observation Space (Box, shape=(4,)):
        [on_hand_stock, pipeline_due_t+1, pipeline_due_t+2, last_observed_sales]

    Action Space:
        By default, Box(low=0.0, high=max_order, shape=(1,), dtype=np.float32)
        Rounded to integer q_t = np.rint(np.clip(action, 0, max_order))
        Supports discrete actions if discrete_actions=True.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        demand_series: Union[List[float], np.ndarray],
        h: float = 0.1,
        p: float = 5.0,
        k: float = 0.0,
        unit_cost: float = 1.0,
        prices: Optional[Union[List[float], np.ndarray]] = None,
        lead_time: int = 2,
        review_period: int = 1,
        warmup_days: int = 0,
        initial_stock: float = 0.0,
        initial_pipeline: Optional[Dict[int, float]] = None,
        max_order: float = 200.0,
        max_inventory: float = 500.0,
        discrete_actions: bool = False,
        action_levels: Optional[List[int]] = None,
        # Aliases for compatibility
        holding_cost: Optional[float] = None,
        lost_sales_penalty: Optional[float] = None,
        fixed_ordering_cost: Optional[float] = None,
    ) -> None:
        super().__init__()

        # Demand series
        self.demand_series = np.asarray(demand_series, dtype=np.float32)
        assert self.demand_series.ndim == 1, "demand_series must be a 1D array"
        assert len(self.demand_series) > warmup_days >= 0, "demand_series length must exceed warmup_days"
        assert np.isfinite(self.demand_series).all(), "demand_series must contain finite numbers"
        assert (self.demand_series >= 0).all(), "demand_series must be non-negative"

        # Prices series
        if prices is not None:
            self.prices = np.asarray(prices, dtype=np.float32)
            assert len(self.prices) == len(self.demand_series), "prices must match demand length"
        else:
            self.prices = np.full(len(self.demand_series), np.nan, dtype=np.float32)

        # Operational parameters (supporting aliases)
        self.h: float = float(holding_cost if holding_cost is not None else h)
        self.p: float = float(lost_sales_penalty if lost_sales_penalty is not None else p)
        self.k: float = float(fixed_ordering_cost if fixed_ordering_cost is not None else k)
        self.unit_cost: float = float(unit_cost)
        self.lead_time: int = int(lead_time)
        self.review_period: int = int(review_period)
        self.warmup_days: int = int(warmup_days)
        self.initial_stock: float = float(initial_stock)
        self.initial_pipeline: Dict[int, float] = dict(initial_pipeline or {})
        self.max_order: float = float(max_order)
        self.max_inventory: float = float(max_inventory)

        # Action space configuration
        self.discrete_actions = discrete_actions
        if self.discrete_actions:
            self.action_levels = action_levels or [0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 100]
            self.action_space = spaces.Discrete(len(self.action_levels))
        else:
            self.action_space = spaces.Box(
                low=np.array([0.0], dtype=np.float32),
                high=np.array([self.max_order], dtype=np.float32),
                dtype=np.float32,
            )

        # Observation space: [stock, pipeline_t+1, pipeline_t+2, last_sales]
        obs_high = np.array(
            [self.max_inventory] + [self.max_order] * self.lead_time + [self.max_order],
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(
            low=np.zeros(len(obs_high), dtype=np.float32),
            high=obs_high,
            dtype=np.float32,
        )

        # Dynamic state variables
        self.t: int = 0
        self.stock: float = 0.0
        self.pipeline: Dict[int, float] = {}
        self.observed_sales: List[float] = []
        self.last_sales: float = 0.0
        self.step_history: List[Dict[str, Any]] = []

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset environment to day t=0."""
        super().reset(seed=seed)

        self.t = 0
        self.pipeline = dict(self.initial_pipeline)
        self.observed_sales = []
        self.last_sales = 0.0
        self.step_history = []

        # Arriving orders at day 0
        received = float(self.pipeline.pop(self.t, 0.0))
        self.stock = float(self.initial_stock) + received

        obs = self._get_obs()
        outstanding = self._get_outstanding()
        info = {
            "day": self.t,
            "on_hand": self.stock,
            "pipeline": outstanding,
            "ip": self.stock + sum(outstanding),
            "state": {
                "stock": self.stock,
                "pipeline": outstanding,
                "history": tuple(self.observed_sales),
            },
        }

        return obs, info

    def step(self, action: Union[int, float, np.ndarray]) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one simulation step for day t."""
        assert self.t < len(self.demand_series), "Episode already finished, call reset()."

        # 1. Determine order quantity q
        if self.discrete_actions:
            if isinstance(action, (int, np.integer)):
                act_idx = int(action)
            elif isinstance(action, np.ndarray) and action.size == 1:
                act_idx = int(action.item())
            else:
                act_idx = int(action)
            q = float(self.action_levels[act_idx])
        else:
            if isinstance(action, np.ndarray):
                raw_act = float(action.item() if action.size == 1 else action[0])
            else:
                raw_act = float(action)
            q = float(np.rint(np.clip(raw_act, 0.0, self.max_order)))

        # 2. Schedule order arrival at t + lead_time if within review period
        review_trigger = (self.t % self.review_period == 0)
        actual_order = q if review_trigger else 0.0
        if actual_order > 0.0:
            arr_day = self.t + self.lead_time
            self.pipeline[arr_day] = self.pipeline.get(arr_day, 0.0) + actual_order

        # 3. Realize demand and fulfill from available stock
        demand = float(self.demand_series[self.t])
        sales = min(self.stock, demand)
        lost = demand - sales
        end_stock = self.stock - sales

        # 4. Cost accounting
        holding_cost = self.h * end_stock
        lost_sales_cost = self.p * lost
        ordering_cost = self.k if actual_order > 0.0 else 0.0
        total_cost = holding_cost + lost_sales_cost + ordering_cost
        reward = -total_cost

        # Margin calculation
        price_t = self.prices[self.t]
        if np.isfinite(price_t):
            gross_margin = sales * (price_t - self.unit_cost)
        else:
            gross_margin = 0.0 if sales == 0.0 else np.nan

        # Record step logging matching simulate()
        outstanding = self._get_outstanding()
        scored = (self.t >= self.warmup_days)
        step_log = {
            "day": self.t,
            "review": review_trigger,
            "cycle": self.t // self.review_period,
            "review_period": self.review_period,
            "received": float(self.pipeline.get(self.t, 0.0)),
            "on_hand": self.stock,
            "ip": self.stock + sum(outstanding),
            "pipeline": outstanding,
            "order": actual_order,
            "arrival_day": (self.t + self.lead_time) if actual_order > 0 else np.nan,
            "demand": demand,
            "sales": sales,
            "lost": lost,
            "end_inventory": end_stock,
            "on_order": sum(self.pipeline.values()),
            "holding_cost": holding_cost,
            "lost_sales_cost": lost_sales_cost,
            "ordering_cost": ordering_cost,
            "total_cost": total_cost,
            "inventory_value": end_stock * self.unit_cost,
            "gross_margin": gross_margin,
            "scored": scored,
        }
        self.step_history.append(step_log)

        # 5. Advance state to next period
        self.observed_sales.append(sales)
        self.last_sales = sales
        self.t += 1

        # Check termination
        terminated = (self.t >= len(self.demand_series))
        truncated = False

        if not terminated:
            # Receive orders arriving at the beginning of day t
            next_received = float(self.pipeline.pop(self.t, 0.0))
            self.stock = end_stock + next_received
        else:
            self.stock = end_stock

        obs = self._get_obs()
        next_outstanding = self._get_outstanding()
        info = {
            **step_log,
            "state": {
                "stock": self.stock,
                "pipeline": next_outstanding,
                "history": tuple(self.observed_sales),
            },
        }

        return obs, reward, terminated, truncated, info

    def _get_outstanding(self) -> Tuple[float, ...]:
        """Return pipeline orders arriving in the next [1, lead_time] days."""
        return tuple(self.pipeline.get(self.t + j, 0.0) for j in range(1, self.lead_time + 1))

    def _get_obs(self) -> np.ndarray:
        """Construct normalized/clipped Markovian observation."""
        outstanding = self._get_outstanding()
        # [stock, pipeline_due_t+1, pipeline_due_t+2, last_sales]
        obs_list = [self.stock] + list(outstanding) + [self.last_sales]
        return np.array(obs_list, dtype=np.float32)

    def get_history_df(self):
        """Return history log as pandas DataFrame matching simulate()."""
        import pandas as pd
        return pd.DataFrame(self.step_history)
